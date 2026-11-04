"""LSV-13 checkpoint-locked rigorous real-arithmetic Jacobian minor verifier.

For each selected case, Arb ball forward differentiation checks 16 entries of
four goal-history gradients on four prior-chosen coordinates. A determinant
ball excluding zero proves local true Jacobian rank >= 4 for that case.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from flint import arb, ctx

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from validated_minor import validated_minor
from run_minor_witness import gradient_history


def run(max_cases=1,precision=128,fetch=False,checkpoint_path=None,
        lock_path=HERE/'source_lock.json',minor_path=HERE/'results'/'lsv12_sparse_minor_20261104.json'):
    lock=json.loads(Path(lock_path).read_text())
    minor=json.loads(Path(minor_path).read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':raise ValueError('bad source schema')
    data,sha=locked_bytes(lock,checkpoint_path,fetch)
    model=load_pretrained(lock,data)
    (inputs,_,_),rosterhash=prepare_roster(lock)
    if minor['checkpoint_sha256']!=sha or minor['roster_sha256']!=rosterhash:
        raise ValueError('LSV-12 source/roster hash mismatch')
    if len(minor['cases'])!=64:raise ValueError('expected exactly 64 frozen cases')
    if not 1<=max_cases<=64:raise ValueError('max_cases out of range')
    torch.set_num_threads(1)
    with torch.no_grad():
        z1=model.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(model).double().eval()
    with torch.no_grad():
        s=z1
        for _ in range(3): s=model.block(s)
        cell=inputs.eq(0).long().argmax(-1)
        ar=torch.arange(len(inputs))
        pairs=model.head(s)[ar,cell].topk(2,dim=1).indices
    # Independent point-autograd matrix. It is only a numerical approximation;
    # ball differences below certify its absolute error against exact derivatives.
    J=gradient_history(model,z1[:max_cases],inputs[:max_cases])
    records=[]
    start=time.perf_counter()
    for case in minor['cases'][:max_cases]:
        idx=case['case_index']
        indices=case['hidden_coordinate_indices']
        if idx != len(records) or len(indices)!=4:raise ValueError('case order/index mismatch')
        winner,rival=(int(pairs[idx,0]),int(pairs[idx,1]))
        pred=validated_minor(model,z1[idx],indices,int(cell[idx]),winner,rival,precision)
        det=pred['det']
        observed=J[idx,:,indices]
        entry_errors=[]
        for row in range(4):
            for col in range(4):
                approx=arb(float(observed[row,col]))
                validated_value=pred['rows'][row][col]
                entry_errors.append((validated_value-approx).abs_upper())
        worst=max(entry_errors)
        previous_budget=arb(str(case['required_verified_abs_entry_error_lt']))
        budget_proved=bool(worst < previous_budget)
        record={
          'case_index':idx,'columns':indices,'selected_goal_cell':int(cell[idx]),
          'winner_digit':winner,'rival_digit':rival,
          'observed_autograd_sigma_min':case['observed_sigma_min'],
          'previous_required_entry_error_lt':case['required_verified_abs_entry_error_lt'],
          'determinant_enclosure':det.str(26),
          'determinant_excludes_zero':pred['rank_four_proved'],
          'max_rigorous_autograd_entry_error_upper':worst.str(18,radius=False),
          'strict_lsv12_entrywise_budget_proved':budget_proved,
          'winner_verified_in_exact_real_model':pred['winning_class_verified'],
          'rival_verified_in_exact_real_model':pred['runner_up_verified'],
          'verified_jacobian_entry_enclosures':[[v.str(20) for v in row] for row in pred['rows']],
          'elapsed_seconds':round(time.perf_counter()-start,3),
        }
        if not all(v.is_finite() for row in pred['rows'] for v in row) or not det.is_finite():
            raise ArithmeticError('nonfinite numerical interval')
        records.append(record)
        print('CASE',idx,'rank4',record['determinant_excludes_zero'],
              'winner',record['winner_verified_in_exact_real_model'],
              'rival',record['rival_verified_in_exact_real_model'],'autograd_budget',budget_proved,
              'det',det.str(12),'elapsed',record['elapsed_seconds'],flush=True)
    certificate_count=sum(v['determinant_excludes_zero'] for v in records)
    return {
      'schema':'LSV-13/arb-validated-goal-minors-v1',
      'evidence':'rigorous ball-arithmetic enclosure of exact-real model derivatives, conditioned on Arb implementation correctness',
      'model':'external 44,001-parameter four-step TRM-inspired Sudoku model',
      'checkpoint_sha256':sha,'source_commit':lock['revision'],'roster_sha256':rosterhash,
      'input_state_contract':'exact binary FP32 stage-one state embedded in real numbers',
      'real_model_contract':'exact binary FP32 weights embedded in real numbers, exact erf-GELU and LayerNorm with exact binary64 eps',
      'not_certified':'machine-code FP32/FP64 rounding, finite-radius robustness, correct Sudoku answers, or semantic 4D latent quotient',
      'arb_precision_bits':precision,'cases_checked':len(records),
      'rigorously_verified_rank_four_cases':certificate_count,
      'cases_discharge_original_lsv12_entry_error_budget':sum(v['strict_lsv12_entrywise_budget_proved'] for v in records),
      'total_rigorously_checked_autograd_derivative_entries':16*len(records),
      'all_exact_real_top1_verified':all(v['winner_verified_in_exact_real_model'] for v in records),
      'all_exact_real_top2_verified':all(v['rival_verified_in_exact_real_model'] for v in records),
      'wall_seconds':round(time.perf_counter()-start,3),
      'cases':records,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fetch',action='store_true')
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--max-cases',type=int,default=1)
    ap.add_argument('--precision',type=int,default=128)
    ap.add_argument('--out',type=Path,default=Path('runs/lsv13/verified.json'))
    args=ap.parse_args()
    r=run(max_cases=args.max_cases,precision=args.precision,fetch=args.fetch,checkpoint_path=args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('RESULT',json.dumps({k:r[k] for k in ('cases_checked','rigorously_verified_rank_four_cases','cases_discharge_original_lsv12_entry_error_budget','all_exact_real_top1_verified','all_exact_real_top2_verified','wall_seconds')}),flush=True)
    if r['rigorously_verified_rank_four_cases']!=r['cases_checked'] or r['cases_discharge_original_lsv12_entry_error_budget']!=r['cases_checked']:
        raise SystemExit('some determinant intervals or derivative error budgets were not certified; see receipt')

if __name__=='__main__':main()
