"""Calibration-only directions and held-out finite-difference causal probes."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import digest, evaluate, tensors, future_correct
from ..tasks import encode, make_evalmath_case, PROGRAMS, TOPOLOGIES


@torch.no_grad()
def calibrate(model, cases):
    device = next(model.parameters()).device
    x, _ = tensors(cases, device)
    out = model(x, policy='joint')
    usage = torch.stack([c.sum((0,1)).cpu() for c in out.codes]).sum(0)
    order = usage.argsort(descending=True)
    axes = model.sae.decoder.weight[:,order[:4]].detach().cpu().double()
    axes = torch.linalg.qr(axes).Q.float()
    pca, means = [], []
    for state in out.states:
        rows = state.detach().cpu().reshape(-1,model.hidden).double()
        mean = rows.mean(0)
        # full right basis permits an identity control at hidden dimension.
        _, _, vh = torch.linalg.svd(rows-mean, full_matrices=True)
        pca.append(vh.T.float())
        means.append(mean.float())
    return {'axes':axes, 'feature_order':order, 'usage':usage,
            'pca':pca, 'means':means,
            'calibration_signatures':[c.signature for c in cases]}


def _hook_input(delta):
    def hook(v,h):
        if v != -1:
            return h
        h = h.clone()
        h[:,0] += delta
        return h
    return hook


def _rank(s):
    p = s/s.sum().clamp_min(1e-20)
    return float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum()))


@torch.no_grad()
def geometry(model, cases, frame, config, oracle, policy='joint'):
    device = next(model.parameters()).device
    axes = frame['axes'].to(device)
    q = model.vpd.readout_basis.detach().cpu().double()
    rows = []
    for task in cases:
        x = encode(task)[None].to(device)
        clean = model(x, policy=policy)
        for eps in config['geometry_epsilons']:
            derivatives, curvature = [], []
            effects = []
            for axis in range(4):
                plus = model(x, policy=policy, hook=_hook_input(eps*axes[:,axis]))
                minus = model(x, policy=policy, hook=_hook_input(-eps*axes[:,axis]))
                derivatives.append([((p-m)/(2*eps)).detach().cpu().double().reshape(-1)
                                    for p,m in zip(plus.states,minus.states)])
                curvature.append([float((p-2*c+m).norm()/(eps*eps))
                                  for p,c,m in zip(plus.states,clean.states,minus.states)])
                effects.append({'axis':axis, 'logit_delta_norm':float((plus.logits-minus.logits).norm()),
                                'plus_top1_future_correct':future_correct(oracle,task,plus.logits.argmax(),model.meter),
                                'minus_top1_future_correct':future_correct(oracle,task,minus.logits.argmax(),model.meter)})
            for dimensions in (2,4):
                previous = None
                for v in range(model.visits):
                    j = torch.stack([d[v] for d in derivatives[:dimensions]], -1)
                    u,s,_ = torch.linalg.svd(j, full_matrices=False)
                    basis = u[:,s>s.max()*1e-6]
                    angles = [] if previous is None else [float(a) for a in torch.acos(
                        torch.linalg.svdvals(previous.T@basis).clamp(0,1))*180/math.pi]
                    previous = basis
                    by_token = j.reshape(4,model.hidden,dimensions)
                    visible = torch.einsum('thd,hk->tkd',by_token,q)
                    support = clean.codes[v][0].detach().cpu()>0
                    prev_support = clean.codes[max(0,v-1)][0].detach().cpu()>0
                    jac = float((support&prev_support).sum()/((support|prev_support).sum().clamp_min(1)))
                    rows.append({'case_id':task.case_id, 'visit':v, 'dimensions':dimensions, 'epsilon':eps,
                                 'singular_values':s.tolist(), 'effective_rank':_rank(s),
                                 'principal_angles_degrees':angles,
                                 'head_readable_jacobian_energy':float(visible.square().sum()),
                                 'head_orthogonal_jacobian_energy':float(j.square().sum()-visible.square().sum()),
                                 'support_jaccard_previous':jac,
                                 'curvature_axes':[c[v] for c in curvature[:dimensions]],
                                 'individual_direction_effects':effects[:dimensions]})
    return {'status':'exploratory_finite_differences_no_continuum_certificate', 'records':rows}


@torch.no_grad()
def fiber_transport(model, cases):
    device = next(model.parameters()).device
    q = model.vpd.readout_basis
    direction = torch.linspace(-1,1,model.hidden,device=device)
    direction -= (direction@q)@q.T
    direction = direction/direction.norm()
    rows = []
    for task in cases:
        x = encode(task)[None].to(device)
        clean = model(x,policy='joint').logits
        for at in range(model.visits):
            def hook(v,h):
                if v!=at:
                    return h
                h=h.clone();h[:,3]+=.05*direction
                return h
            changed = model(x,policy='joint',hook=hook).logits
            rows.append({'id':task.case_id,'visit':at,'logit_effect_norm':float((changed-clean).norm())})
    return {'patch_norm':.05,'rows':rows,'exact_linear_head_only':True}


def compression_hook(model, frame, method, dimension, at):
    device=next(model.parameters()).device
    order=frame['feature_order'].to(device)
    q=model.vpd.readout_basis
    pca=frame['pca'][at][:,:dimension].to(device)
    mean=frame['means'][at].to(device)
    feature_basis=torch.linalg.qr(model.sae.decoder.weight[:,order[:min(dimension,model.hidden)]].detach()).Q
    def hook(v,h):
        if v!=at:
            return h
        if method=='pca':
            return mean+((h-mean)@pca)@pca.T
        if method=='sae_groups':
            return (h@feature_basis)@feature_basis.T
        if method=='sae_reconstruction':
            # Retain calibration-selected features; no task-specific ranking.
            code=model.sae.encode(F.layer_norm(h,(model.hidden,)))
            kept=torch.zeros_like(code);kept[...,order[:dimension]]=code[...,order[:dimension]]
            return model.sae.decoder(kept)
        visible=(h@q)@q.T
        if method=='head_readable':
            qq=q[:,:min(dimension,q.shape[1])]
            return (h@qq)@qq.T
        if method=='fiber_preservation':
            candidate=pca-(q@(q.T@pca))
            u,s,_=torch.linalg.svd(candidate,full_matrices=False)
            ff=u[:,s>1e-5]
            return visible+(h@ff)@ff.T
        raise ValueError(method)
    return hook


@torch.no_grad()
def compactification(model,cases,frame,config,oracle):
    rows=[]
    for at in range(model.visits):
        for d in config['retained_dimensions']:
            if d>model.hidden:
                continue
            for method in ('pca','sae_groups','sae_reconstruction','head_readable','fiber_preservation'):
                result=evaluate(model,cases,oracle,config['budget'],policy='joint',
                                hook=compression_hook(model,frame,method,d,at))
                rows.append({'visit':at,'retained_dimension_requested':d,'method':method,
                             'dimension_note': 'fiber method additionally preserves the full head-readable space; SAE count is feature count',
                             'overall':result['overall']})
    return {'calibration_frozen':True,'records':rows}


@torch.no_grad()
def interchange(model,oracle,frame,config,count=16):
    """Swap the first source operator, holding public input series fixed."""
    device=next(model.parameters()).device
    pairs=[]
    for i in range(count):
        graph=PROGRAMS[(i*7)%64]
        changed=(('N' if graph[0]!='N' else 'R'),graph[1],graph[2])
        a=make_evalmath_case(oracle,1870000+i,case_id=f'causal_recipient:{i}',family='lacunary',graph=graph,
                             topology=TOPOLOGIES[i%4])
        b=make_evalmath_case(oracle,1870000+i,case_id=f'causal_donor:{i}',family='lacunary',graph=changed,topology=a.topology)
        assert a.query==b.query and all(x[0]==y[0] for x,y in zip(a.demos,b.demos))
        oa=model(encode(a)[None].to(device),policy='joint')
        ob=model(encode(b)[None].to(device),policy='joint')
        valid=(a.hidden_target!=b.hidden_target and future_correct(oracle,a,oa.logits.argmax(),model.meter)
               and future_correct(oracle,b,ob.logits.argmax(),model.meter))
        pairs.append((a,b,tuple(s.cpu() for s in oa.states),tuple(s.cpu() for s in ob.states),valid))
        del oa,ob
    rows=[]
    for at in range(model.visits):
        for d in (2,4,8,16,model.hidden):
            basis=frame['pca'][at][:,:d].to(device)
            valid=correct=unconditional=0
            for a,b,oa,ob,is_valid in pairs:
                # CPU-cached donor and recipient traces are copied for this one test.
                difference=(ob[at]-oa[at]).to(device)
                patch=(difference@basis)@basis.T
                def hook(v,h):
                    return h+patch if v==at else h
                out=model(encode(a)[None].to(device),policy='joint',hook=hook)
                success=future_correct(oracle,b,out.logits.argmax(),model.meter)
                unconditional+=int(success)
                valid+=int(is_valid);correct+=int(is_valid and success)
            rows.append({'visit':at,'dimension':d,'valid_counterfactual_tests':valid,
                         'correct_counterfactual_outcomes':correct,
                         'causal_fidelity':correct/valid if valid else None,
                         'unconditional_successes':unconditional,'attempted_pairs':count})
    return {'named_variable':'first formal-series unary source operator, others fixed',
            'validity':'both clean top1 futures exact and oracle futures differ', 'records':rows}


@torch.no_grad()
def topology_diagnostics(model,task,frame,certificate_root):
    """Invoke existing exact-rational routines with explicitly sampled premises."""
    sys.path.insert(0,str(Path(certificate_root).resolve()))
    from scm12_rank_one import rank_one_null_certificate
    from scm13_affine_disk import affine_disk_certificate
    device=next(model.parameters()).device
    axes=frame['axes'][:,:2].to(device)
    x=encode(task)[None].to(device)
    clean=model(x,policy='joint')
    radius=.01
    plus=[];minus=[]
    for d in range(2):
        plus.append(model(x,policy='joint',hook=_hook_input(radius*axes[:,d])))
        minus.append(model(x,policy='joint',hook=_hook_input(-radius*axes[:,d])))
    circle=[]
    for k in range(16):
        theta=2*math.pi*k/16
        u=torch.tensor([math.cos(theta),math.sin(theta)],device=device)
        circle.append((u,model(x,policy='joint',hook=_hook_input(radius*(axes@u)))))
    rows=[]
    for v in range(model.visits):
        # Four token strands, fixed two-dimensional chart, same intervention disk.
        base=(clean.states[v][0]@axes).cpu().double()
        j=torch.stack([((plus[d].states[v][0]-minus[d].states[v][0])@axes/(2*radius)).cpu().double()
                       for d in range(2)],-1)
        affine_errors=torch.zeros(4,dtype=torch.double)
        rank_one_errors=affine_errors.clone()
        for u,out in circle:
            actual=(out.states[v][0]@axes).cpu().double()
            surrogate=base+radius*(j@u.cpu().double())
            rank_one=base+radius*j[:,:,0]*float(u[0])
            affine_errors=torch.maximum(affine_errors,(actual-surrogate).norm(dim=-1))
            rank_one_errors=torch.maximum(rank_one_errors,(actual-rank_one).norm(dim=-1))
        a=affine_disk_certificate(base.tolist(),j.tolist(),str(radius),affine_errors.tolist())
        r=rank_one_null_certificate(base.tolist(),j[:,:,0].tolist(),[-radius,radius],rank_one_errors.tolist())
        rows.append({'visit':v,'affine_disk':a,'rank_one':r,
                     'premises':'finite differences and 16 sampled circle points only',
                     'neural_certificate':'inconclusive_missing_uniform_continuum_bounds'})
    return {'status':'exploratory_conditional_surrogate_only','records':rows}
