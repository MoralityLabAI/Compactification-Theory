"""LSV-12: find compact 4x4 minors for a numerical rank-obstruction witness.

Observed autograd entries still need a separately *verified* error bound before
this becomes a formal numerical certificate for the pretrained network.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import torch
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from run_goal_horizon import quantiles
from sparse_minor import greedy_minor

HERE=Path(__file__).resolve().parent


def gradient_history(model,z1,inputs):
    h=z1.detach().clone().requires_grad_(True)
    B=len(inputs)
    ar=torch.arange(B)
    first=inputs.eq(0).long().argmax(-1)
    states=[h]
    for _ in range(3):states.append(model.block(states[-1]))
    finals=model.head(states[-1])[ar,first]
    pair=finals.topk(2,dim=-1).indices.detach()
    grads=[]
    for k,z in enumerate(states):
        y=model.head(z)[ar,first]
        margin=y.gather(-1,pair[:,0:1]).squeeze(-1)-y.gather(-1,pair[:,1:2]).squeeze(-1)
        grads.append(torch.autograd.grad(margin.sum(),h,retain_graph=k<3)[0].detach().flatten(1))
    return torch.stack(grads,dim=1)


def run(lock_path,checkpoint=None,fetch=False):
    lock=json.loads(Path(lock_path).read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':
        raise ValueError('unexpected source lock')
    data,sha=locked_bytes(lock,checkpoint,fetch)
    raw=load_pretrained(lock,data).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    torch.set_num_threads(1)
    with torch.no_grad():z1=raw.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(raw).double().eval()
    J=gradient_history(model,z1,inputs)
    records=[]
    for i in range(len(inputs)):
        w=greedy_minor(J[i])
        records.append({'case_index':i,'hidden_coordinate_indices':w['columns'],
            'observed_sigma_min':round(w['observed_sigma_min'],12),
            'required_verified_abs_entry_error_lt':round(w['required_verified_abs_entry_error_lt'],12),
            'max_abs_selected_derivative':round(w['observed_max_abs_entry'],12)})
    eps=[v['required_verified_abs_entry_error_lt'] for v in records]
    min_sv=[v['observed_sigma_min'] for v in records]
    return {'schema':'LSV-12/sparse-obstruction-minor-v1',
       'status':'OBSERVATIONAL_MINOR; VERIFIED_ENTRYWISE_ERROR_BUDGET_UNAVAILABLE',
       'checkpoint_sha256':sha,'roster_sha256':roster,
       'source_commit':lock['revision'],'torch_version':torch.__version__,
       'n_cases':len(records),'n_gradients_per_case':4,'latent_coordinates':3888,
       'entries_requiring_validation_per_case':16,
       'required_verified_max_abs_entry_error_quantiles':quantiles(eps),
       'observed_minor_sigma_min_quantiles':quantiles(min_sv),
       'cases_with_positive_observed_minor':sum(s>0 for s in min_sv),
       'cases_certified_true_rank_four':0,
       'caution':'A rigorous row-gradient entry error enclosure is required before any observed minor certifies true rank.',
       'cases':records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    rec=run(a.lock,a.checkpoint,a.fetch)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:rec[key] for key in ('schema','status','checkpoint_sha256','roster_sha256',
           'n_cases','cases_with_positive_observed_minor','cases_certified_true_rank_four',
           'required_verified_max_abs_entry_error_quantiles','observed_minor_sigma_min_quantiles')},indent=2))

if __name__=='__main__':main()
