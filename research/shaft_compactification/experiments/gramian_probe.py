"""SHAFT-COMP-12: finite-horizon LOCAL observability of trained toy GRUs.

For a frozen recurrent trajectory, evaluate the Jacobian cocycle
Phi_0=I, Phi_(k+1)=J_k Phi_k; O_L=[W Phi_0;...;W Phi_L].
The right singular space of O_L yields local linearized observation directions.
A nonlinear counterfactual control tests whether these local diagnostics carry
predictive value away from the linearization point.

Only local synthetic four-bit lookup tasks and PyTorch in-memory inference.
No network, model replication, or any persistent agent operations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from research.shaft_compactification.experiments.trained_recurrence import fit, TOKENS, HIDDEN
from research.shaft_compactification.experiments.causal_probe import train_probe, capture, counterfactual

SEEDS = (101, 202)
DEPTHS = (12, 40)
HORIZONS = tuple(range(8))
THRESHOLDS = (1e-3, 1e-5)
RADII = (0.02, 0.2)


def analytic_jacobian(cell, inputs, states):
    """Analytic GRUCell derivative (PyTorch gates reset, update, new)."""
    weight_in, weight_hidden = cell.weight_ih, cell.weight_hh
    bias_in, bias_hidden = cell.bias_ih, cell.bias_hh
    pre_in = F.linear(inputs, weight_in, bias_in)
    pre_hidden = F.linear(states, weight_hidden, bias_hidden)
    ir, iz, inn = pre_in.chunk(3, dim=1)
    hr, hz, hn = pre_hidden.chunk(3, dim=1)
    r = torch.sigmoid(ir + hr)
    z = torch.sigmoid(iz + hz)
    n = torch.tanh(inn + r * hn)
    R, Z, N = weight_hidden.chunk(3, dim=0)
    outer_update = ((states - n) * z * (1 - z)).unsqueeze(-1) * Z
    dnew = r.unsqueeze(-1) * N + (hn * r * (1 - r)).unsqueeze(-1) * R
    outer_new = ((1 - z) * (1 - n*n)).unsqueeze(-1) * dnew
    return torch.diag_embed(z) + outer_update + outer_new


def jacobian_parity(cell, seed=8088):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(HIDDEN, generator=g)
    token = F.one_hot(torch.tensor(2), TOKENS).float()
    with torch.no_grad():
        exact = analytic_jacobian(cell, token[None], h[None])[0]
    autograd = torch.autograd.functional.jacobian(lambda hh: cell(token, hh), h)
    return float((exact-autograd).abs().max())


def trajectory(model, state, tokens):
    h=state.unsqueeze(0)
    path=[h[0]]
    jac=[]
    with torch.no_grad():
        for token in tokens:
            inp=F.one_hot(torch.tensor([int(token)]),TOKENS).float()
            jac.append(analytic_jacobian(model.cell,inp,h)[0])
            h=model.cell(inp,h)
            path.append(h[0])
    return path, jac


def observability_blocks(W, jacobians):
    n=W.shape[1]
    phi=torch.eye(n, dtype=W.dtype)
    blocks=[W@phi]
    for J in jacobians:
        phi=J @ phi
        blocks.append(W@phi)
    return blocks


def rank_at(s, threshold):
    return int((s > threshold * s[0]).sum()) if s[0] > 0 else 0


def linear_stats(blocks):
    out=[]
    for k in HORIZONS:
        matrix=torch.cat(blocks[:k+1],dim=0).double()
        singular=torch.linalg.svdvals(matrix)
        Gram = matrix.T @ matrix
        ev=torch.linalg.eigvalsh(Gram).clamp_min(0).flip(0)
        vals=singular.cpu().numpy()
        out.append({
            'horizon':k,
            'algebraic_rank_at_relative_1e_6':rank_at(vals,1e-6),
            'rank_at_relative_1e_3':rank_at(vals,1e-3),
            'rank_at_relative_1e_5':rank_at(vals,1e-5),
            'stable_rank':float(ev.sum()/ev[0]) if ev[0]>0 else 0.,
            'max_singular':float(singular.max()),
            'min_positive_singular':float(singular[singular > singular[0] * 1e-6].min()),
            'gramian_consistency':float((ev[:len(singular)] - singular.square()).abs().max()) if len(singular)==HIDDEN else None,
        })
    return out


def interventions(model, probe, state, tokens, blocks, noise_seed):
    g = torch.Generator().manual_seed(noise_seed)
    W=probe.weight.detach()
    O=torch.cat(blocks[:5],dim=0).double()
    _,singular,Vh=torch.linalg.svd(O,full_matrices=True)
    blind=Vh[-1].float()
    assert float((O@blind.double()).norm()) < 1e-5
    _,_,v0=torch.linalg.svd(W.double(),full_matrices=True)
    base_null=v0[4:].T.float()
    arbitrary=torch.randn(HIDDEN,generator=g)
    random_null=base_null @ (base_null.T @ arbitrary)
    random_null=random_null/random_null.norm()
    random=torch.randn(HIDDEN,generator=g)
    random=random/random.norm()
    directions={'horizon4_linear_blind':blind/blind.norm(),
                'instantaneous_null_random':random_null,
                'isotropic_random':random}
    baseline=[]
    h=state.unsqueeze(0)
    with torch.no_grad():
        baseline.append((probe(h).detach()[0],h.detach()[0]))
        for t in tokens:
            inp=F.one_hot(torch.tensor([int(t)]),TOKENS).float()
            h=model.cell(inp,h)
            baseline.append((probe(h).detach()[0],h.detach()[0]))
    report={}
    for radius in RADII:
        rr={}
        for label, direction in directions.items():
            perturb=radius*direction
            nonlinear=[]
            changed=(state+perturb).unsqueeze(0)
            with torch.no_grad():
                nonlinear.append(probe(changed).detach()[0])
                for t in tokens:
                    inp=F.one_hot(torch.tensor([int(t)]),TOKENS).float()
                    changed=model.cell(inp,changed)
                    nonlinear.append(probe(changed).detach()[0])
            diff=torch.stack([x-baseline[i][0] for i,x in enumerate(nonlinear)])
            linear=torch.stack([block@perturb for block in blocks])
            discrepancy=diff-linear
            teacher_changed=0
            for k in (0,4,7):
                seq=torch.tensor(tokens[:k],dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    orig = counterfactual(model,state.unsqueeze(0),seq) >= 0
                    altered = counterfactual(model,(state+perturb).unsqueeze(0),seq) >= 0
                teacher_changed += int(bool((orig!=altered).any()))
            rr[label]={
                'initial_probe_l2_shift':float(diff[0].norm()),
                'linearized_l2_until_4':float(linear[:5].norm()),
                'observed_l2_until_4':float(diff[:5].norm()),
                'linearization_error_l2_until_4':float(discrepancy[:5].norm()),
                'linearized_l2_until_7':float(linear.norm()),
                'observed_l2_until_7':float(diff.norm()),
                'teacher_query_signature_changed_at_0_4_7':teacher_changed,
                'max_probe_logit_change_until_7':float(diff.abs().max()),
            }
        report[str(radius)]=rr
    return report


def run(samples_per_seed_depth=16):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    if samples_per_seed_depth <= 0 or samples_per_seed_depth > 128:
        raise ValueError('samples_per_seed_depth must lie in 1..128')
    report={
        'experiment':'SHAFT-COMP-12',
        'evidence':'trained_small_gru_local_jacobian_cocycle',
        'torch':str(torch.__version__),
        'numpy':str(np.__version__),
        'seeds':list(SEEDS),
        'state_depths':list(DEPTHS),
        'horizons':list(HORIZONS),
        'samples_per_seed_depth':samples_per_seed_depth,
        'relative_rank_thresholds':[1e-3,1e-5,1e-6],
        'radii':list(RADII),
        'results':{},
    }
    for seed in SEEDS:
        model,loss=fit(seed)
        probe,probe_loss=train_probe(model,seed)
        parity=jacobian_parity(model.cell,seed+10)
        assert parity<1e-5,parity
        by_depth={}
        for depth in DEPTHS:
            samples_per_goal=max(1,(samples_per_seed_depth+15)//16)
            captured,_=capture(model,(depth,),samples_per_goal,seed+30000+depth)[depth]
            captured=captured[:samples_per_seed_depth]
            records=[]
            for i, state in enumerate(captured):
                gen=torch.Generator().manual_seed(940000 + seed * 1000 + depth * 100 + i)
                tokens=torch.randint(TOKENS,(len(HORIZONS)-1,),generator=gen).tolist()
                _,js=trajectory(model,state,tokens)
                blocks=observability_blocks(probe.weight.detach(),js)
                stats=linear_stats(blocks)
                intervention=interventions(model,probe,state,tokens,blocks,
                                           noise_seed=seed*1048576 + depth*65536+i)
                records.append({'ranks':stats,'interventions':intervention})
            by_depth[str(depth)]={
                'samples':len(records),
                'rank_profiles':{},
                'interventions':{},
            }
            for horizon in HORIZONS:
                rank3=[x['ranks'][horizon]['rank_at_relative_1e_3'] for x in records]
                rank5=[x['ranks'][horizon]['rank_at_relative_1e_5'] for x in records]
                rank6=[x['ranks'][horizon]['algebraic_rank_at_relative_1e_6'] for x in records]
                sr=[x['ranks'][horizon]['stable_rank'] for x in records]
                by_depth[str(depth)]['rank_profiles'][str(horizon)]={
                    'rank_rel_1e3_median':float(np.median(rank3)),
                    'rank_rel_1e5_median':float(np.median(rank5)),
                    'rank_rel_1e6_median':float(np.median(rank6)),
                    'stable_rank_median':float(np.median(sr)),
                    'rank_rel_1e3_min':int(min(rank3)),
                    'rank_rel_1e3_max':int(max(rank3)),
                    'rank_rel_1e5_min':int(min(rank5)),
                    'rank_rel_1e5_max':int(max(rank5)),
                    'rank_rel_1e5_full_rank_count':int(sum(r == HIDDEN for r in rank5)),
                }
            for radius in RADII:
                by_depth[str(depth)]['interventions'][str(radius)]={}
                for label in ('horizon4_linear_blind','instantaneous_null_random','isotropic_random'):
                    rows=[x['interventions'][str(radius)][label] for x in records]
                    by_depth[str(depth)]['interventions'][str(radius)][label]={
                        key+'_median':float(np.median([row[key] for row in rows]))
                        for key in rows[0] if key != 'teacher_query_signature_changed_at_0_4_7'
                    }
                    by_depth[str(depth)]['interventions'][str(radius)][label][
                        'teacher_signature_change_frequency_0_4_7']=float(np.mean([
                        row['teacher_query_signature_changed_at_0_4_7']>0 for row in rows]))
            assert all(x['ranks'][0]['algebraic_rank_at_relative_1e_6']==4 for x in records)
        report['results'][str(seed)]={
            'base_final_train_loss':float(loss),'probe_train_loss':float(probe_loss),
            'analytic_vs_autograd_jacobian_max_abs':parity,'depths':by_depth}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--out')
    args=p.parse_args()
    result=json.dumps(run(args.samples),indent=2,sort_keys=True)+'\n'
    if args.out:
        path=Path(args.out)
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(result)
    else:
        print(result,end='')
if __name__=='__main__':main()
