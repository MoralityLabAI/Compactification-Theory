"""LSV-14 independent exact rational replay of a positive-radius Jacobian receipt.

Depends only on the Python standard library, and the pre-existing LSV-13
rational determinant arithmetic. Checks every case from saved decimal balls;
it does not trust claimed success flags or source-order metadata without hashes.
This verifies finite witness arithmetic, conditional on trusted Arb evaluation.
"""
from __future__ import annotations
import argparse
import json
from fractions import Fraction
from pathlib import Path
from verify_serialized_minors import parse_ball, independent_det
from robust_inverse import inverse_perturbation_witness


def verify(robust:dict, baseline:dict):
    if robust.get('schema')!='LSV-14/positive-radius-goal-rank-v1':
        raise ValueError('bad robust receipt schema')
    if baseline.get('schema')!='LSV-13/arb-validated-goal-minors-v1':
        raise ValueError('bad locked baseline schema')
    if robust.get('checkpoint_sha256')!=baseline['checkpoint_sha256'] or robust.get('roster_sha256')!=baseline['roster_sha256']:
        raise ValueError('mismatched checkpoint or roster')
    if len(baseline['cases'])!=64 or baseline['rigorously_verified_rank_four_cases']!=64:
        raise ValueError('baseline lacks complete 64-case proof')
    if not isinstance(robust.get('radius'),str) or Fraction(robust['radius'])<=0:
        raise ValueError('radius must be strictly positive')
    if robust.get('cases_checked')!=len(robust['cases']) or len(robust['cases'])>64:
        raise ValueError('bad cases count')
    verified=0;neumann=0;either=0;class1=class2=0
    for idx,record in enumerate(robust['cases']):
        old=baseline['cases'][idx]
        if (record['case_index']!=idx or record['radius']!=robust['radius']
             or record['initial_box_dimensions']!=3888 or record['selected_columns']!=old['columns']
             or record['goal_cell']!=old['selected_goal_cell']
             or record['winner_digit']!=old['winner_digit'] or record['rival_digit']!=old['rival_digit']):
            raise AssertionError(f'case {idx}: source-locked selection changed')
        entries=record['entry_intervals']
        if len(entries)!=4 or any(len(row)!=4 for row in entries):
            raise ValueError('missing 4-by-4 interval rows')
        lo,hi=independent_det([[parse_ball(s) for s in row] for row in entries])
        success=lo>0 or hi<0
        if bool(record['independent_rational_det_excludes_zero'])!=success:
            raise AssertionError(f'case {idx}: claimed determinant result disagrees')
        lower=(lo if lo>0 else -hi if hi<0 else Fraction(0))
        if lower!=Fraction(record['independent_rational_abs_det_lower_bound']):
            raise AssertionError(f'case {idx}: claimed strict lower bound differs')
        exact_ineq=inverse_perturbation_witness(
            entries,old['observed_autograd_binary64_hex'])
        claimed=record.get('neumann_certificate')
        if claimed!=exact_ineq:
            raise AssertionError(f'case {idx}: exact Neumann witness disagrees')
        verified+=int(success)
        neumann+=int(exact_ineq['rank_four_by_neumann'])
        either+=int(success or exact_ineq['rank_four_by_neumann'])
        class1+=int(record['top1_stable_across_box']);class2+=int(record['top2_stable_across_box'])
    if class1!=robust['top1_stable_cases'] or class2!=robust['top2_stable_cases']:
        raise AssertionError('class claim totals disagree')
    if verified!=robust['replay']['positive_radius_rank_four_cases']:
        raise AssertionError('count disagrees with exact rational replay')
    if neumann!=robust['neumann_certified_cases'] or either!=robust['rank_four_cases_by_either_method']:
        raise AssertionError('Neumann or union totals disagree')
    return {'radius':robust['radius'],'cases_checked':len(robust['cases']),
            'neumann_rank_four_cases':neumann,'rank_four_cases_by_either_method':either,
            'positive_radius_rank_four_cases':verified,'top1_stable_cases_reported':class1,
            'top2_stable_cases_reported':class2,'locked_selection_matches':True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('receipt',type=Path)
    ap.add_argument('--baseline',type=Path,default=Path(__file__).resolve().parent/'results'/'lsv13_arb_validated_64_20261104.json')
    args=ap.parse_args()
    print(json.dumps(verify(json.loads(args.receipt.read_text()), json.loads(args.baseline.read_text())),sort_keys=True,indent=2))

if __name__=='__main__':main()
