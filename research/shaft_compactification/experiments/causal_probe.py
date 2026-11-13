"""SHAFT-COMP-11: controlled causal test for compact predictive-state codes.

Test native frozen toy GRUs in memory only. A linear behavioral probe can be
exact on training-depth states while its kernel is not dynamically invariant.
Compare length-matched perturbations in probe-nullspace, rowspace and random
state directions at several future horizons, measuring counterfactual outputs.

No network, replication, secret-sharing or persistence mechanism.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from research.shaft_compactification.experiments.trained_recurrence import fit, HIDDEN, TOKENS

SEEDS=(101,202)
TRAIN_DEPTHS=(4,8,12)
TEST_DEPTHS=(12,40)
HORIZONS=(0,1,4,12)
RADII=(0.75,1.5,3.0)
TASKS=16
PER_TASK=32

def capture(model, depths, per_task, seed):
    gen=torch.Generator().manual_seed(seed)
    goals=torch.arange(TASKS).repeat_interleave(per_task)
    perm=torch.randperm(len(goals),generator=gen)
    goals=goals[perm]
    seq=torch.randint(TOKENS,(len(goals),max(depths)),generator=gen)
    out={}
    with torch.no_grad():
        h=model.embedding(goals)
        for step in range(1,max(depths)+1):
            h=model.cell(F.one_hot(seq[:,step-1], TOKENS).float(),h)
            if step in depths:
                out[step]=(h.detach().clone(),goals.detach().clone())
    return out

def counterfactual(model,h,continuations):
    with torch.no_grad():
        for i in range(continuations.shape[1]):
            h=model.cell(F.one_hot(continuations[:,i],TOKENS).float(),h)
        q=F.one_hot(torch.arange(TOKENS),TOKENS).float().repeat(len(h),1)
        h4=h.repeat_interleave(TOKENS,dim=0)
        next_h=model.cell(q,h4)
        logits=model.readout(torch.cat((next_h,q),dim=1)).reshape(len(h),TOKENS)
        return logits

def train_probe(model,seed):
    x,y=[],[]
    data=capture(model,TRAIN_DEPTHS,48,seed+70000)
    for depth,(state,goals) in data.items():
        x.append(state)
        y.append((counterfactual(model,state,torch.empty(len(state),0,dtype=torch.long))>=0).float())
    x=torch.cat(x);y=torch.cat(y)
    torch.manual_seed(seed+90000)
    probe=nn.Linear(HIDDEN,TOKENS)
    optim=torch.optim.Adam(probe.parameters(),lr=.04)
    for _ in range(400):
        loss=F.binary_cross_entropy_with_logits(probe(x),y)
        optim.zero_grad();loss.backward();optim.step()
    return probe.eval(),float(loss.item())

def perturbation_subspaces(probe,gaussian):
    # W=4x24; right singular vectors span row-space and nullspace.
    with torch.no_grad():
        U,S,Vh=torch.linalg.svd(probe.weight.detach(),full_matrices=True)
        assert (S>1e-5).sum().item()==TOKENS
        row=Vh[:TOKENS].T
        null=Vh[TOKENS:].T
        result={}
        for label,basis in [('row',row),('null',null)]:
            result[label]=gaussian @ basis @ basis.T
        result['random']=gaussian
        return result

def evaluate(seed):
    model,loss=fit(seed)
    probe,probe_loss=train_probe(model,seed)
    report={'base_training_loss':round(loss,8),'probe_training_loss':round(probe_loss,8),
            'depths':{}}
    for depth in TEST_DEPTHS:
        h,g=capture(model,(depth,),PER_TASK,seed+100000+depth)[depth]
        with torch.no_grad():
            base_code=probe(h)>=0
        generator=torch.Generator().manual_seed(seed+200000+depth)
        gaussian=torch.randn(len(h),HIDDEN,generator=generator)
        vectors=perturbation_subspaces(probe,gaussian)
        continuations={k:torch.randint(TOKENS,(len(h),k),generator=generator) for k in HORIZONS}
        baseline={k:(counterfactual(model,h,seq)>=0) for k,seq in continuations.items()}
        out={'samples':len(h),'probe_code_exact_vs_synthetic_truth':round(float((base_code==((g[:,None]>>torch.arange(TOKENS))&1).bool()).all(dim=1).float().mean()),6), 'radii':{}}
        for radius in RADII:
            res={}
            for arm,vec in vectors.items():
                unit=vec/torch.linalg.vector_norm(vec,dim=1,keepdim=True).clamp_min(1e-9)
                h_changed=h+radius*unit
                with torch.no_grad():
                    changed_code=probe(h_changed)>=0
                    original_logits=probe(h)
                    changed_logits=probe(h_changed)
                entry={
                    'code_preserved_fraction':round(float((base_code==changed_code).all(dim=1).float().mean()),6),
                    'code_logit_delta_max':round(float((original_logits-changed_logits).abs().max()),8),
                    'mean_applied_l2':round(float((h_changed-h).norm(dim=1).mean()),6),
                    'future':{}
                }
                for k,seq in continuations.items():
                    changed=(counterfactual(model,h_changed,seq)>=0)
                    base=baseline[k]
                    entry['future'][str(k)]={
                        'exact_signature_preservation':round(float((base==changed).all(dim=1).float().mean()),6),
                        'bit_preservation':round(float((base==changed).float().mean()),6),
                        'changed_despite_same_probe_code_fraction':round(float((((base!=changed).any(dim=1)) & ((base_code==changed_code).all(dim=1))).float().mean()),6),
                    }
                res[arm]=entry
            out['radii'][str(radius)]=res
        report['depths'][str(depth)]=out
    return report

def run():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    return {'status':'trained_local_toy_gru_causal_probe',
            'torch':torch.__version__,
            'model_seeds':list(SEEDS),'train_probe_depths':list(TRAIN_DEPTHS),
            'test_depths':list(TEST_DEPTHS),'future_horizons':list(HORIZONS),
            'perturbation_radii':list(RADII),'samples_per_depth_per_model':TASKS*PER_TASK,
            'method':'fixed-norm rowspace/nullspace/random perturbations of frozen recurrent state; source probe not retrained after intervention',
            'results':{str(seed):evaluate(seed) for seed in SEEDS}}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out')
    opts=parser.parse_args()
    result=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if opts.out:
        Path(opts.out).write_text(result)
    else:print(result,end='')
if __name__=='__main__':main()
