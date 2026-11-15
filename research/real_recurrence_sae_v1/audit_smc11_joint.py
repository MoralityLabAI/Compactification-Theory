"""SMC-11: independent receipt-level consistency and frozen-weight audit.

Uses only standard-library Python. This does not rerun Arb or the source
teacher; CI already does so. It independently re-evaluates every exact
Fraction feature-identity guard and the joint causal-policy decision,
checks complete puzzle/seed/radius/budget coverage, and verifies all
three serialized SAE binary snapshot hashes.

A certificate's source/rounding scope remains as declared in OLM-11/SMC-10.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

SCHEMA='SMC-11C/same-fit-SAE-identity-and-Arb-local-causal-v1'
CHECKPOINT='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
ROSTER='eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611'
PUZZLES=tuple(range(3,64,4))
SEEDS=(71,89,113)
RADII=('1/50','1/5')
BUDGETS=('0','1/1000000000000','1/10000000000','1/100000000','1/1000000')
CONTROLS=('isotropic','sae_inactive','jacobian_optimal')
LABELS=('sae_active_greater','control_greater','inconclusive','abstain_selection')


def guard(radius_sq,budget):
    """Exact rational strict inequality; None represents infinite radius."""
    r=Q(budget)
    if r<0:
        raise ValueError('negative uncertainty')
    return radius_sq is None or r*r<Q(radius_sq)


def verify_selection(row):
    if row['status']!='certified':
        if any(v for g in row['budget_guards'].values() for v in g.values()):
            raise ValueError('abstaining selection has an accepted guard')
        return
    if not row.get('torch_same_top1_and_top4'):
        raise ValueError('nominal Torch and exact model differ')
    if len(row['top4_features'])!=4 or len(set(row['top4_features']))!=4:
        raise ValueError('top-k identity malformed')
    if row['top1_feature'] not in row['top4_features']:
        raise ValueError('top1 feature not in top4')
    for budget in BUDGETS:
        actual=row['budget_guards'][budget]
        for key,field in (('top1','top1_radius_sq_exact'),
                          ('top4','top4_radius_sq_exact')):
            expected=guard(row[field],budget)
            if actual[key] is not expected:
                raise ValueError('incorrect exact ball guard '+key)
    if 'rays' in row:
        for sign in ('positive','negative'):
            ray=row['rays'][sign]
            for radius in ('1/50','1/5','4/5'):
                for key in ('top1','topk'):
                    value=ray[key]['radius_squared_exact']
                    if ray['guards'][radius][key] is not guard(value,radius):
                        raise ValueError('incorrect sharp signed ray guard')


def check_joint(report,snapshot_dir=None,full=True):
    if report['schema']!=SCHEMA:
        raise ValueError('wrong receipt schema')
    if report['checkpoint_sha256']!=CHECKPOINT or report['roster_sha256']!=ROSTER:
        raise ValueError('untrusted source checkpoint or roster')
    if report['sae_steps']!=300 or report['arb_precision_bits']<100:
        raise ValueError('unexpected fitted SAE or precision')
    if full:
        if tuple(report['test_puzzle_ids'])!=PUZZLES:
            raise ValueError('test roster differs')
        if tuple(report['seeds'])!=SEEDS:
            raise ValueError('SAE seed list differs')
        if tuple(report['radii'])!=RADII or tuple(report['common_l2_budget_grid'])!=BUDGETS:
            raise ValueError('intervention grammar differs')
    keys=set()
    snaps={}
    for row in report['frozen_sae_snapshots']:
        seed=row['seed']
        if seed in snaps:
            raise ValueError('repeated fitted SAE seed')
        for field in ('trained_sae_sha256','snapshot_file_sha256'):
            value=row[field]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid frozen weight SHA256')
        snaps[seed]=row
        if snapshot_dir is not None:
            p=Path(snapshot_dir)/('sae_seed'+str(seed)+'.pt')
            if not p.is_file():
                raise ValueError('missing frozen SAE file')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row['snapshot_file_sha256']:
                raise ValueError('frozen SAE file hash mismatch')
    if set(snaps)!=set(report['seeds']):
        raise ValueError('SAE seed/snapshot coverage differs')
    selections={}
    for row in report['exact_goal_selection_cases']:
        key=(row['seed'],row['puzzle_id'])
        if key in selections:
            raise ValueError('duplicated goal-token selection')
        verify_selection(row)
        selections[key]=row
    if len(selections)!=len(report['seeds'])*len(report['test_puzzle_ids']):
        raise ValueError('missing goal-token cases')
    summary=defaultdict(Counter)
    for row in report['cases']:
        if row['seed'] not in report['seeds'] or row['puzzle_id'] not in report['test_puzzle_ids']:
            raise ValueError('unregistered SAE seed/puzzle tuple')
        if (row['radius'] not in report['radii'] or
                row['common_l2_uncertainty'] not in report['common_l2_budget_grid']):
            raise ValueError('unregistered causal radius or uncertainty')
        key=(row['seed'],row['puzzle_id'],row['radius'],row['common_l2_uncertainty'])
        if key in keys:
            raise ValueError('duplicated causal tuple')
        keys.add(key)
        sel=selections[(row['seed'],row['puzzle_id'])]
        if row['goal_token']!=sel['goal_token']:
            raise ValueError('causal goal-token mismatch')
        if row['sae_snapshot_sha256']!=snaps[row['seed']]['trained_sae_sha256']:
            raise ValueError('causal analysis used different SAE tensor fingerprint')
        budget=row['common_l2_uncertainty']
        if row['exact_top1_guard'] is not sel['budget_guards'][budget]['top1']:
            raise ValueError('top1 proof used inconsistent selection certificate')
        if row['exact_top4_guard'] is not sel['budget_guards'][budget]['top4']:
            raise ValueError('top4 proof used inconsistent selection certificate')
        for control in CONTROLS:
            original=row['comparisons'][control]
            composed=row['composed_policy_comparisons'][control]
            if original not in ('sae_active_greater','control_greater','inconclusive'):
                raise ValueError('unknown causal proof label')
            needed='top4' if control=='sae_inactive' else 'top1'
            expected=(original if sel['status']=='certified'
                      and sel['budget_guards'][budget][needed]
                      else 'abstain_selection')
            if expected!=composed:
                raise ValueError('causal/identity composition disagrees')
            summary[row['radius']+'|'+budget+'|'+control][composed]+=1
    num_expected=len(report['seeds'])*len(report['test_puzzle_ids'])*len(report['radii'])*len(report['common_l2_budget_grid'])
    if len(keys)!=num_expected:
        raise ValueError('missing causal proof contexts')
    for tag in sorted(report['same_fit_composed_summary']):
        total=report['same_fit_composed_summary'][tag]
        counts=summary[tag]
        for stored,base in [('sae_active_greater','sae_active_greater'),
                            ('control_greater','control_greater'),
                            ('inconclusive_original','inconclusive'),
                            ('abstain_selection','abstain_selection')]:
            if total[stored]!=counts[base]:
                raise ValueError('summary mismatches independently recomputed proof labels')
        if total['total']!=sum(counts.values()):
            raise ValueError('incorrect summary denominator')
    if len(report['same_fit_composed_summary'])!=len(report['radii'])*len(report['common_l2_budget_grid'])*3:
        raise ValueError('incorrect summary coverage')
    if full and len(keys)!=480:
        raise ValueError('unexpected full source test case count')
    rays={}
    for sign in ('positive','negative'):
        for radius in ('1/50','1/5','4/5'):
            rays[sign+'|'+radius]={
                key:sum(bool(row.get('rays',{}).get(sign,{}).get('guards',{}).get(radius,{}).get(key))
                        for row in selections.values())
                for key in ('top1','topk')}
    return {
        'schema':'SMC-11C/independent-receipt-and-snapshot-audit-v1',
        'rechecked_exact_selection_cells':len(selections),
        'rechecked_arb_causal_contexts':len(keys),
        'rechecked_control_decisions':len(keys)*3,
        'validated_snapshots':len(snaps),
        'all_source_fingerprint_provenance_matched':True,
        'selection_abstentions':sum(row['status']!='certified' for row in selections.values()),
        'independent_signed_ray_guard_counts':rays,
        'composed_iso_at_r02_eta1e8':report['same_fit_composed_summary'].get('1/5|1/100000000|isotropic'),
        'scope':'Independent proof label, exact guard and frozen file hash audit of hosted source-locked Arb receipt. Does not replace its interval forward verification.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--snapshot-dir',type=Path)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--out',type=Path,default=Path('runs/smc11/joint-audit.json'))
    args=p.parse_args()
    content=args.receipt.read_bytes()
    data=json.loads(content)
    result=check_joint(data,args.snapshot_dir,full=not args.pilot)
    result['source_receipt_sha256']=hashlib.sha256(content).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
