"""SHAFT-COMP-14: preregistered task-weighted directional observability test.

Local synthetic 3,265-param GRUs only. Distinct random diagnostic and outcome
continuations. Frozen protocol in SHAFT_COMP_14_PROTOCOL.md. No networking,
model replication or persistence mechanism.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from research.shaft_compactification.experiments.trained_recurrence import (
    fit, GOALS, HIDDEN, TOKENS
)
from research.shaft_compactification.experiments.causal_probe import train_probe, counterfactual
from research.shaft_compactification.experiments.gramian_probe import analytic_jacobian
from research.shaft_compactification.experiments.observability_prediction import (
    sample_pair, model_digest, auc
)

CALIBRATION_SEEDS=(101,202)
HELDOUT_SEEDS=(505,606)
DEPTHS=(12,40)
PER_GOAL=16
DIAGNOSTIC_STEPS=5
OUTCOME_STEPS=32
RANK_REL=1e-3
PERMUTATIONS=1000
BOOTSTRAPS=1000


def native_logits_grad(model, states):
    """Native four counterfactual logits and their exact first derivatives.

    h_next=GRUCell(onehot(q), h), z=readout([h_next,onehot(q)]).
    Native output head is Linear -> tanh -> Linear.
    """
    n=len(states)
    q=F.one_hot(torch.arange(TOKENS), TOKENS).float().repeat(n,1)
    s=states.repeat_interleave(TOKENS,dim=0)
    next_state=model.cell(q,s)
    derivative_cell=analytic_jacobian(model.cell,q,s)
    first=model.readout[0]
    second=model.readout[2]
    cat=torch.cat((next_state,q),dim=-1)
    u=torch.tanh(first(cat))
    z=second(u).reshape(n,TOKENS)
    grad_inside=((1-u.square())*second.weight) @ first.weight[:,:HIDDEN]
    grad=(grad_inside.unsqueeze(1) @ derivative_cell).squeeze(1)
    return z, grad.reshape(n,TOKENS,HIDDEN)


def check_grad(model):
    generator=torch.Generator().manual_seed(141401)
    x=torch.randn(HIDDEN,generator=generator)
    with torch.no_grad():
        _,gradient=native_logits_grad(model,x.unsqueeze(0))
    def fn(h):
        q=F.one_hot(torch.arange(TOKENS),TOKENS).float()
        after=model.cell(q,h.expand(TOKENS,-1))
        return model.readout(torch.cat((after,q),dim=-1)).flatten()
    with torch.enable_grad():
        truth=torch.autograd.functional.jacobian(fn,x.detach().requires_grad_(True))
    difference=(gradient[0]-truth).abs().max().item()
    assert difference<1e-5,difference
    return difference


def directional_scores(model,probe,states_a,states_b,inputs):
    """Same diagnostic inputs for both states. Tangent contrast fixed at t=0.

    S_T = sqrt(0.5 * delta' * (Q_a + Q_b) * delta), where
    Q_s sums Phi' C' diag(w(z)^2) C Phi over diagnostic times.
    """
    n=len(states_a)
    W=probe.weight.detach()
    contrast=(states_a-states_b).unsqueeze(-1)
    init_phi=torch.eye(HIDDEN).unsqueeze(0).expand(n,HIDDEN,HIDDEN)
    weighted=[]
    unweighted=[]
    ranks=[]
    current_native_margins=[]
    with torch.no_grad():
        for start in (states_a,states_b):
            h=start.detach()
            phi=init_phi.clone()
            blocks=[W@phi]
            weighted_energy=torch.zeros(n)
            unweighted_energy=torch.zeros(n)
            for step in range(DIAGNOSTIC_STEPS+1):
                native_logits,C=native_logits_grad(model,h)
                if step==0:
                    current_native_margins.append(native_logits.abs().min(dim=1).values)
                v=(phi@contrast).squeeze(-1)
                directional=(C@v.unsqueeze(-1)).squeeze(-1)
                weights=(1+native_logits.abs()).reciprocal()
                weighted_energy+=(directional*weights).square().sum(dim=1)
                unweighted_energy+=directional.square().sum(dim=1)
                if step<DIAGNOSTIC_STEPS:
                    inp=F.one_hot(inputs[:,step],TOKENS).float()
                    J=analytic_jacobian(model.cell,inp,h)
                    phi=J@phi
                    h=model.cell(inp,h)
                    blocks.append(W@phi)
            O=torch.cat(blocks,dim=1).double()
            s=torch.linalg.svdvals(O)
            ranks.append((s > RANK_REL*s[:,[0]]).sum(dim=1).float())
            weighted.append(weighted_energy)
            unweighted.append(unweighted_energy)
        task=torch.sqrt(.5*(weighted[0]+weighted[1]).clamp_min(0))
        generic=torch.sqrt(.5*(unweighted[0]+unweighted[1]).clamp_min(0))
        rank_deficit=HIDDEN-.5*(ranks[0]+ranks[1])
        near_boundary=-.5*(current_native_margins[0]+current_native_margins[1])
        probe_margin=-.5*(probe(states_a).abs().min(dim=1).values +
                          probe(states_b).abs().min(dim=1).values)
        separation=(states_a-states_b).norm(dim=1)
    return {k:v.numpy() for k,v in {
        'task_weighted':task,
        'native_unweighted':generic,
        'rank_deficit':rank_deficit,
        'near_native_margin':near_boundary,
        'low_probe_margin':probe_margin,
        'state_distance':separation,
    }.items()}


def pair_rows(model,probe,seed,depth,per_goal):
    goals,ha,hb,_=sample_pair(model,seed,depth,per_goal)
    n=len(goals)
    diag_gen=torch.Generator().manual_seed(1400000+seed*1000+depth)
    outcome_gen=torch.Generator().manual_seed(1500000+seed*1000+depth)
    diagnostic=torch.randint(TOKENS,(n,DIAGNOSTIC_STEPS),generator=diag_gen)
    outcome=torch.randint(TOKENS,(n,OUTCOME_STEPS),generator=outcome_gen)
    with torch.no_grad():
        za=counterfactual(model,ha,torch.empty((n,0),dtype=torch.long))
        zb=counterfactual(model,hb,torch.empty((n,0),dtype=torch.long))
        teacher_same=((za>=0)==(zb>=0)).all(dim=1)
        probe_same=((probe(ha)>=0)==(probe(hb)>=0)).all(dim=1)
        outcome_a=counterfactual(model,ha,outcome)>=0
        outcome_b=counterfactual(model,hb,outcome)>=0
        event=(outcome_a!=outcome_b).any(dim=1)
    scores=directional_scores(model,probe,ha,hb,diagnostic)
    records=[]
    for idx in range(n):
        row=dict(seed=seed,depth=depth,pair=idx,goal_for_audit_only=int(goals[idx]),
                 teacher_same=bool(teacher_same[idx]),probe_same=bool(probe_same[idx]),
                 eligible=bool(teacher_same[idx]&probe_same[idx]),
                 diverged_32=int(event[idx]))
        for key,values in scores.items(): row[key]=float(values[idx])
        records.append(row)
    return records


def summaries(rows):
    e=[r for r in rows if r['eligible']]
    keys=['task_weighted','native_unweighted','rank_deficit',
          'near_native_margin','low_probe_margin','state_distance']
    labels=[r['diverged_32'] for r in e]
    return {'total':len(rows),'eligible':len(e),'positives':sum(labels),
            'auroc':{k:(None if (a:=auc([r[k] for r in e],labels)) is None else round(a,6))
                     for k in keys},
            'mean_score_by_label':{k: {
                str(y): round(float(np.mean([r[k] for r in e if r['diverged_32']==y])),6)
                if any(r['diverged_32']==y for r in e) else None for y in (0,1)} for k in keys}}


def stats_test(rows, permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    e=[r for r in rows if r['eligible'] and r['seed'] in HELDOUT_SEEDS]
    if not e:return {'error':'no eligible heldout data'}
    group={}
    for i,r in enumerate(e):group.setdefault((r['seed'],r['depth']),[]).append(i)
    y=np.asarray([r['diverged_32'] for r in e])
    new=np.asarray([r['task_weighted'] for r in e])
    base=np.asarray([r['rank_deficit'] for r in e])
    an=auc(new,y);ab=auc(base,y)
    rng=np.random.default_rng(141414)
    per=[]
    for _ in range(permutations):
        labels=y.copy()
        for ids in group.values():labels[ids]=rng.permutation(labels[ids])
        a=auc(new,labels)
        if a is not None:per.append(a)
    center=float(np.mean(per)) if per else None
    p=(1+sum(abs(v-center)>=abs(an-center)-1e-12 for v in per))/(1+len(per)) if per else None
    boot_new=[];boot_delta=[]
    for _ in range(bootstraps):
        idx=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in group.values()])
        a=auc(new[idx],y[idx]);b=auc(base[idx],y[idx])
        if a is not None and b is not None:
            boot_new.append(a);boot_delta.append(a-b)
    return {'registered_task_auc':None if an is None else round(an,6),
            'matched_rank_auc':None if ab is None else round(ab,6),
            'task_minus_rank_auc':None if an is None or ab is None else round(an-ab,6),
            'permutation_p_two_sided':None if p is None else round(p,6),
            'permutation_null_mean':None if center is None else round(center,6),
            'permutation_count':len(per),
            'stratified_bootstrap_count':len(boot_new),
            'task_bootstrap95':[round(float(x),6) for x in np.quantile(boot_new,[.025,.975])],
            'difference_bootstrap95':[round(float(x),6) for x in np.quantile(boot_delta,[.025,.975])],
            'note':'within-model uncertainty only; two held-out model seeds'}


def run(per_goal=PER_GOAL,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    rows=[];cells={};hashes={};parity={}
    for seed in CALIBRATION_SEEDS+HELDOUT_SEEDS:
        model,_=fit(seed)
        probe,_=train_probe(model,seed)
        hashes[str(seed)]={'base':model_digest(model),'probe':model_digest(probe)}
        parity[str(seed)]=check_grad(model)
        for depth in DEPTHS:
            cell=pair_rows(model,probe,seed,depth,per_goal)
            cells[f'{seed}_d{depth}']=summaries(cell)
            rows.extend(cell)
    cal=[r for r in rows if r['seed'] in CALIBRATION_SEEDS]
    test=[r for r in rows if r['seed'] in HELDOUT_SEEDS]
    return {
        'experiment':'SHAFT-COMP-14',
        'registered_protocol':'SHAFT_COMP_14_PROTOCOL.md',
        'torch':torch.__version__,'numpy':np.__version__,
        'calibration_seeds':list(CALIBRATION_SEEDS),'heldout_seeds':list(HELDOUT_SEEDS),
        'depths':list(DEPTHS),'per_goal':per_goal,
        'diagnostic_horizon':DIAGNOSTIC_STEPS,'outcome_horizon':OUTCOME_STEPS,
        'score_direction':'larger task-weighted directional Gramian predicts divergence',
        'boundary_weight':'w(z)=1/(1+abs(z))',
        'rank_relative_threshold':RANK_REL,
        'analytic_native_jacobian_autograd_max_abs':parity,
        'model_sha256':hashes,
        'cells':cells,
        'calibration_pooled':summaries(cal),
        'heldout_pooled':summaries(test),
        'uncertainty':stats_test(rows,permutations,bootstraps),
        'rows':rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--per-goal',type=int,default=PER_GOAL)
    ap.add_argument('--permutations',type=int,default=PERMUTATIONS)
    ap.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if not 1<=a.per_goal<=64:raise ValueError('per-goal must be in 1..64')
    result=run(a.per_goal,a.permutations,a.bootstraps)
    output=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(output)
    else:
        print(output,end='')

if __name__=='__main__':
    main()
