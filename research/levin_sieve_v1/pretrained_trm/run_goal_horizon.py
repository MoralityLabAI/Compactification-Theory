"""LSV-10: local tangent goal-horizon rank on a frozen pretrained Sudoku model.

Exploratory diagnostic only: autograd at the clean FP64 trajectory supplies
point Jacobians. Their singular-value ranks are not certified global quotients.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster

HERE = Path(__file__).resolve().parent


def quantiles(vals):
    v=torch.as_tensor(vals,dtype=torch.float64).flatten()
    if not torch.isfinite(v).all():
        raise ValueError('nonfinite tangent diagnostics')
    return {key:round(torch.quantile(v,torch.tensor(q,dtype=torch.float64)).item(),8)
            for key,q in (('min',0),('p10',.1),('median',.5),('p90',.9),('max',1))}


def tangent_profile(gradients:torch.Tensor,rank_relative_tolerance=1e-3):
    """Per-case tangent rank of H goal-history gradients, [N,H,D]."""
    if gradients.ndim != 3 or not all(s>0 for s in gradients.shape):
        raise ValueError('expected nonempty [N,H,D] gradient stack')
    if not gradients.is_floating_point() or not torch.isfinite(gradients).all():
        raise ValueError('finite floating-point gradients required')
    if not 0 < rank_relative_tolerance < 1:
        raise ValueError('invalid relative rank threshold')
    # Use double precision even when the caller provides FP32 gradients.
    g=gradients.double()
    singular=torch.linalg.svdvals(g)
    relative=singular / singular[:, :1].clamp_min(1e-300)
    rank=(relative > rank_relative_tolerance).sum(-1)
    gram=torch.bmm(g,g.transpose(1,2))
    norm=torch.linalg.vector_norm(g,dim=-1)
    adjacent_cos = (torch.diagonal(gram,offset=1,dim1=-2,dim2=-1) /
                    (norm[:,:-1]*norm[:,1:]).clamp_min(1e-300))
    return {'rank':rank,'normalized_singular_values':relative,
            'adjacent_goal_gradient_cosine':adjacent_cos,
            'gradient_norms':norm}


def analysis(model,stage1,inputs,targets,relative_tolerance=1e-3):
    """Freeze classes using final step then differentiate their margin at each depth."""
    if stage1.ndim!=3 or stage1.shape[1:]!=(81,48) or stage1.dtype!=torch.float64:
        raise ValueError('expected 81x48 FP64 stage-1 hidden state')
    if len(stage1)!=len(inputs) or len(targets)!=len(inputs):
        raise ValueError('roster size mismatch')
    h=stage1.detach().clone().requires_grad_(True)
    states=[h]
    for _ in range(3):states.append(model.block(states[-1]))
    B=len(inputs)
    ar=torch.arange(B)
    missing=inputs.eq(0)
    if not missing.any(-1).all():
        raise ValueError('one missing cell required per puzzle')
    indices=missing.long().argmax(-1)
    final=model.head(states[-1])[ar,indices]
    top2=final.topk(2,dim=-1).indices.detach()
    winner,competitor=top2[:,0],top2[:,1]
    margins=[];gradients=[]
    for i,z in enumerate(states):
        logits=model.head(z)[ar,indices]
        gap=logits.gather(1,winner[:,None]).flatten()-logits.gather(1,competitor[:,None]).flatten()
        margins.append(gap.detach())
        gradients.append(torch.autograd.grad(gap.sum(),h,retain_graph=i<3)[0].detach().flatten(1))
    stack=torch.stack(gradients,dim=1)
    profile=tangent_profile(stack,relative_tolerance)
    ranks=profile['rank']
    # The (N,4,81,48) gradient tensor supports a local spatial sensitivity measure.
    energy=stack.reshape(B,4,81,48).square().sum(-1)
    active=(energy > energy.amax(-1,keepdim=True)*.01).sum(-1)
    outcomes={
      'n_puzzles':B,'n_hidden_coordinates':81*48,
      'goal':'top-two final-step digit margin for the first missing cell, competitors frozen before differentiation',
      'n_steps_from_stage1':3,'rank_relative_singular_value_tolerance':relative_tolerance,
      'tangent_history_rank_distribution':{str(k):int((ranks==k).sum()) for k in range(5)},
      'mean_rank':round(ranks.double().mean().item(),8),
      'smallest_to_largest_singular_value_ratio_quantiles':quantiles(profile['normalized_singular_values'][:,-1]),
      'adjacent_grad_cosine_quantiles':quantiles(profile['adjacent_goal_gradient_cosine']),
      'gradient_norm_quantiles_by_depth':{str(i):quantiles(profile['gradient_norms'][:,i]) for i in range(4)},
      'local_spatial_support_token_count_quantiles_by_depth':{str(i):quantiles(active[:,i]) for i in range(4)},
      'final_pair_gap_quantiles':quantiles(margins[-1]),
      'selected_goal_class_correct':int((winner==targets[ar,indices]).sum()),
      'identity_recurrence_tangent_rank':1,
      'notes':'local tangent-space observational rank only, not a nonlinear finite-region invariant or a finite-radius proof',
    }
    if outcomes['tangent_history_rank_distribution']['0']:
        raise AssertionError('zero rank in a margin with nonzero affine head')
    return outcomes


def run(lock_path,checkpoint=None,fetch=False):
    lock=json.loads(Path(lock_path).read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':
        raise ValueError('unexpected source lock')
    data,digest=locked_bytes(lock,checkpoint,fetch)
    torch.set_num_threads(1)
    raw=load_pretrained(lock,data).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    with torch.no_grad():z1=raw.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(raw).double().eval()
    return {'schema':'LSV-10/goal-horizon-tangent-v1','status':'observational_point_autograd_not_global_certificate',
            'checkpoint_sha256':digest,'roster_sha256':roster,'source_commit':lock['revision'],
            'torch_version':torch.__version__,'measurements':analysis(model,z1,inputs,targets)}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    a.add_argument('--checkpoint',type=Path)
    a.add_argument('--fetch',action='store_true')
    a.add_argument('--out',type=Path)
    args=a.parse_args()
    rec=run(args.lock,args.checkpoint,args.fetch)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps(rec,sort_keys=True,indent=2))

if __name__=='__main__':main()
