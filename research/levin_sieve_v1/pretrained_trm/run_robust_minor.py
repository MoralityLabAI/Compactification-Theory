"""LSV-14 source-locked execution of positive-radius recurrent rank witnesses."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from robust_minor import robust_minor,check_full_receipt
from robust_inverse import inverse_perturbation_witness


def run(radius='1e-12',max_cases=64,precision=160,fetch=False,
        checkpoint=None,lock_path=HERE/'source_lock.json',
        baseline_path=HERE/'results'/'lsv13_arb_validated_64_20261104.json'):
    source=json.loads(Path(lock_path).read_text())
    baseline=json.loads(Path(baseline_path).read_text())
    raw_bytes,sha=locked_bytes(source,checkpoint,fetch)
    (inputs,_,_),roster=prepare_roster(source)
    if baseline['schema']!='LSV-13/arb-validated-goal-minors-v1' or baseline['cases_checked']!=64:
        raise ValueError('not a validated LSV-13 64-case receipt')
    if baseline['checkpoint_sha256']!=sha or baseline['roster_sha256']!=roster:
        raise ValueError('checkpoint or roster mismatch')
    if baseline['rigorously_verified_rank_four_cases']!=64:
        raise ValueError('LSV-13 baseline not all verified')
    if type(max_cases)!=int or not 1<=max_cases<=64:
        raise ValueError('max_cases out of range')
    torch.set_num_threads(1)
    model=load_pretrained(source,raw_bytes)
    with torch.no_grad():z1=model.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(model).double().eval()
    records=[]
    start=time.perf_counter()
    for case in baseline['cases'][:max_cases]:
        index=len(records)
        if case['case_index']!=index:
            raise ValueError('prior case ordering changed')
        result=robust_minor(model,z1[index],case['columns'],case['selected_goal_cell'],
                            case['winner_digit'],case['rival_digit'],radius,precision)
        result['case_index']=index
        result['neumann_certificate']=inverse_perturbation_witness(
            result['entry_intervals'],case['observed_autograd_binary64_hex'])
        records.append(result)
        print('CASE',index,'radius',radius,'rank4',result['independent_rational_det_excludes_zero'],
              'top1',result['top1_stable_across_box'],
              'elapsed',round(time.perf_counter()-start,3),flush=True)
    rec={
        'schema':'LSV-14/positive-radius-goal-rank-v1',
        'status':'rigorous interval enclosure, sound when Arb and source contracts hold',
        'radius':radius,'precision_bits':precision,
        'state_space':'entire 3888-dimensional L-infinity ball about each exact binary32 stage-one state',
        'mathematical_model':'exact real lift of frozen binary32 weights, exact erf-GELU and LayerNorm',
        'source_revision':source['revision'],'checkpoint_sha256':sha,'roster_sha256':roster,
        'cases':records,'cases_checked':len(records),'wall_seconds':round(time.perf_counter()-start,3),
        'disclaimer':'does not certify hardware rounding, task correctness, future trajectory convergence, or class stability unless separately reported'
    }
    rec['replay']=check_full_receipt(rec)
    rec['neumann_certified_cases']=sum(c['neumann_certificate']['rank_four_by_neumann'] for c in records)
    rec['rank_four_cases_by_either_method']=sum(
        c['neumann_certificate']['rank_four_by_neumann'] or c['independent_rational_det_excludes_zero']
        for c in records)
    rec['top1_stable_cases']=sum(c['top1_stable_across_box'] for c in records)
    rec['top2_stable_cases']=sum(c['top2_stable_across_box'] for c in records)
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--radius',default='1e-12')
    ap.add_argument('--max-cases',type=int,default=64)
    ap.add_argument('--precision',type=int,default=160)
    ap.add_argument('--fetch',action='store_true')
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--out',type=Path,default=Path('runs/lsv14/robust.json'))
    args=ap.parse_args()
    result=run(radius=args.radius,max_cases=args.max_cases,precision=args.precision,fetch=args.fetch,checkpoint=args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('RESULT',json.dumps({k:result[k] for k in ('radius','cases_checked','replay','neumann_certified_cases','rank_four_cases_by_either_method','top1_stable_cases','top2_stable_cases','wall_seconds')}))

if __name__=='__main__':main()
