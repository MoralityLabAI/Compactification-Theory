"""SMC-15 independent stdlib auditor for frozen native SAE erasure receipts.

Recomputes the declared calibration signs, eligibility, per-cluster signed
concordance, bootstrap interval, and success gate. It does not re-execute
PyTorch neural models and must not be read as validation of network weights.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

BASES = {'smc15_novel_30':(202615030000,30),
         'smc15_novel_34':(202615040000,34)}
SOURCE_SHA='7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087'
INDEPENDENT_SHA='1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3'
CONDITIONED_SHA='3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07'
PAIRS=[('pretrainedA_71',f'newB_{seed}_SAE_{sae}') for seed in (501,701) for sae in (71,89)]
EPS=1e-8


def close(a,b,tol=2e-8):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol*max(1,abs(a),abs(b))


def sign(x):
    return 0 if abs(x)<=EPS else (1 if x>0 else -1)


def validate_rosters(r,full):
    seen_solution=set(); seen_input=set()
    for fam,(base,holes) in BASES.items():
        rows=r['fresh_cohort_rosters'][fam]
        n=64 if full else 4
        if len(rows)!=n:
            raise ValueError('fresh cohort size changed')
        if r['preregistered_test_bases'][fam]!=base:
            raise ValueError('fresh seed base changed')
        fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if fingerprint!=r['fresh_cohort_manifest_sha256'][fam]:
            raise ValueError('fresh cohort manifest hash differs')
        for i,row in enumerate(rows):
            if row['seed']!=base+i or row['holes']!=holes:
                raise ValueError('unregistered fresh seed or hole count')
            x,y=row['x'],row['target']
            if len(x)!=81 or len(y)!=81 or x.count(0)!=holes:
                raise ValueError('malformed fresh Sudoku puzzle')
            for indices in ( [[9*rr+cc for cc in range(9)] for rr in range(9)] +
                             [[9*rr+cc for rr in range(9)] for cc in range(9)] +
                             [[9*rr+cc for rr in range(br,br+3) for cc in range(bc,bc+3)]
                              for br in (0,3,6) for bc in (0,3,6)]):
                if {y[t] for t in indices}!=set(range(1,10)):
                    raise ValueError('fresh target is not complete Sudoku')
            if any(v and v!=y[j] for j,v in enumerate(x)):
                raise ValueError('fresh clue conflicts with target')
            if row['rectangles']<=0:
                raise ValueError('fresh target violates noncanonical invariant')
            tup_y,tup_x=tuple(y),tuple(x)
            if tup_y in seen_solution or tup_x in seen_input:
                raise ValueError('duplicate fresh puzzle or target')
            seen_solution.add(tup_y);seen_input.add(tup_x)


def validate_raw(rows,roster,require_eligible):
    if len(rows)!=len(roster):raise ValueError('puzzle count changed')
    for i,(row,puzzle) in enumerate(zip(rows,roster)):
        if row['puzzle_index']!=i or row['seed']!=puzzle['seed']:
            raise ValueError('puzzle seed/order changed')
        slots=row['goal_positions']
        blanks=[j for j,v in enumerate(puzzle['x']) if v==0]
        expected=[blanks[(j*len(blanks))//8] for j in range(8)]
        if slots!=expected:
            raise ValueError('target-dependent or changed selected goal positions')
        keys=('eligible_mask','source_selected_ids','target_matched_ids',
              'target_other_active_ids','target_other_active_coefficients',
              'target_match_clean_coefficients','target_match_after_erase_coefficients',
              'matched_physical_dose','other_physical_dose',
              'source_drop','matched_drop','other_active_drop',
              'source_clean_correct','target_clean_correct',
              'source_correct_to_wrong_erase','target_matched_correct_to_wrong_erase',
              'target_other_correct_to_wrong_erase')
        if any(len(row[k])!=8 for k in keys):
            raise ValueError('raw native erasure token count differs')
        for j,e in enumerate(row['eligible_mask']):
            if e and require_eligible:
                if not (0<=row['target_matched_ids'][j]<96 and
                        0<=row['target_other_active_ids'][j]<96 and
                        row['target_matched_ids'][j]!=row['target_other_active_ids'][j] and
                        row['target_match_clean_coefficients'][j]>0 and
                        row['target_other_active_coefficients'][j]>0):
                    raise ValueError('eligible target feature not naturally active and distinct')
            for column in ('source_drop','matched_drop','other_active_drop','matched_physical_dose','other_physical_dose'):
                if not math.isfinite(row[column][j]):
                    raise ValueError('nonfinite native erasure result')
            if not close(row['matched_physical_dose'][j],row['other_physical_dose'][j],1e-9):
                raise ValueError('control physical dose not matched')
            if row['matched_physical_dose'][j]<0:
                raise ValueError('negative physical dose')
            for col in ('source','target_matched','target_other'):
                if row[col+'_correct_to_wrong_erase'][j] and not row[('source' if col=='source' else 'target')+'_clean_correct'][j]:
                    raise ValueError('wrong clean-to-wrong flip definition')


def orient_replay(rows):
    count=0;matched=0.;other=0.
    for r in rows:
        for i,eligible in enumerate(r['eligible_mask']):
            if eligible:
                count+=1
                matched+=r['source_drop'][i]*r['matched_drop'][i]
                other+=r['source_drop'][i]*r['other_active_drop'][i]
    return {name:{'sign':1 if x>=0 else -1,
                  'identified':count>0 and abs(x)>1e-14,
                  'eligible_calibration_goals':count,
                  'calibration_products_sum':x}
            for name,x in (('matched',matched),('other',other))}


def score_replay(rows,orientation,seed,draws=2000):
    signs={key:orientation[key]['sign'] for key in ('matched','other')}
    counts=[]; s_m=[];s_o=[]
    for row in rows:
        k=0; a=0;b=0
        for i,e in enumerate(row['eligible_mask']):
            if e:
                k+=1
                s=sign(row['source_drop'][i])
                a+=s*sign(row['matched_drop'][i])*signs['matched']
                b+=s*sign(row['other_active_drop'][i])*signs['other']
        counts.append(k);s_m.append(a);s_o.append(b)
    total=sum(counts);n=sum(k>0 for k in counts)
    if total==0 or not all(orientation[key]['identified'] for key in signs):
        return {'status':'insufficient_coverage_or_unidentified_calibration',
                'eligible_goals':total,'nonempty_puzzle_clusters':n,
                'registered_success':False}
    rng=random.Random(seed)
    values=[]
    for _ in range(draws):
        indices=[rng.randrange(len(rows)) for _ in rows]
        den=sum(counts[i] for i in indices)
        if den:
            values.append(sum(s_m[i]-s_o[i] for i in indices)/den)
    values.sort()
    ok=len(values)==draws
    interval=[values[int(.025*draws)],values[int(.975*draws)-1]] if ok else None
    return {
      'status':'measured' if ok else 'bootstrap_abstained',
      'eligible_goals':total,'nonempty_puzzle_clusters':n,
      'mean_matched_signed_concordance':sum(s_m)/total,
      'mean_other_active_signed_concordance':sum(s_o)/total,
      'matched_minus_other_active':sum(s_m[i]-s_o[i] for i in range(len(rows)))/total,
      'unweighted_mean_puzzle_difference':sum((s_m[i]-s_o[i])/counts[i] for i in range(len(rows)) if counts[i])/n,
      'puzzle_cluster_bootstrap95':interval,
      'bootstrap_valid_resamples':len(values),
      'bootstrap_requested_resamples':draws,
      'registered_success':total>=24 and n>=20 and ok and interval[0]>0}


def audit(receipt):
    if receipt['schema']!='SMC-15/fresh-cohort-active-native-erasure-v1':
        raise ValueError('unrecognized SMC15 receipt schema')
    if (receipt['source_receipt_sha256']!=SOURCE_SHA or
        receipt['independent_training_receipt_sha256']!=INDEPENDENT_SHA or
        receipt['conditioned_mapping_receipt_sha256']!=CONDITIONED_SHA):
        raise ValueError('source/model/calibration receipt provenance mismatch')
    full=receipt['full_protocol']
    if full!=(receipt['fresh_puzzles_per_family']==64 and len(receipt['compare'])==4):
        raise ValueError('full-protocol flag inconsistent')
    validate_rosters(receipt,full)
    for ix,pr in enumerate(receipt['compare']):
        if (pr['source'],pr['target'])!=PAIRS[ix]:
            raise ValueError('unregistered model/SAE comparison')
        if not pr['calibration']['feature_assignment_sha256']['matched_to_original_smc13c_receipt']:
            raise ValueError('feature matching source SHA not verified')
        cal=pr['calibration']
        if len(cal['raw'])!=64:raise ValueError('original calibration cohort must be 64 puzzles')
        # Original calibration puzzle IDs are source-registered; x is not needed for signs.
        for i,row in enumerate(cal['raw']):
            if row['puzzle_index']!=i or row['seed']!=202613020000+i:
                raise ValueError('calibration puzzle changed')
        replay=orient_replay(cal['raw'])
        for key in ('matched','other'):
            for scalar in ('sign','identified','eligible_calibration_goals'):
                if replay[key][scalar]!=cal['orientation'][key][scalar]:
                    raise ValueError('calibration orientation changed')
            if not close(replay[key]['calibration_products_sum'],cal['orientation'][key]['calibration_products_sum'],1e-8):
                raise ValueError('calibration product sum differs')
        for fam in BASES:
            data=pr['heldout'][fam]
            raw=data['per_puzzle'];roster=receipt['fresh_cohort_rosters'][fam]
            validate_raw(raw,roster,True)
            replay_stats=score_replay(raw,replay,seed=202615130+100*ix+len(fam))
            prev=data['statistics']
            for field,val in replay_stats.items():
                old=prev.get(field)
                if isinstance(val,list):
                    if not isinstance(old,list) or len(old)!=len(val) or not all(close(a,b) for a,b in zip(val,old)):
                        raise ValueError('bootstrap confidence bound changed')
                elif isinstance(val,float):
                    if not close(val,old):raise ValueError('native erasure score changed: '+field)
                elif val!=old:
                    raise ValueError('native erasure decision changed: '+field)
            coverage=data['coverage']
            if coverage['eligible_goals']!=replay_stats['eligible_goals'] or coverage['selected_goals']!=8*len(raw):
                raise ValueError('reported active feature coverage changed')
        joint=full and all(pr['heldout'][fam]['statistics']['registered_success'] for fam in BASES)
        if joint!=pr['registered_success_on_both_fresh_families']:
            raise ValueError('pair success misreported')
    return {'schema':'SMC-15/independent-stdlib-replay-v1',
            'exact_fresh_cohort_manifest_checked':True,
            'fresh_sudoku_solution_and_goal_masks_checked':True,
            'calibration_native_erasures_reoriented_from_raw':True,
            'physical_dose_and_natural_activity_checked':True,
            'paired_puzzle_bootstraps_and_preregistered_gates_recomputed':True,
            'n_primary_pairs':len(receipt['compare']),
            'n_fresh_puzzle_pair_family_cases':sum(len(pr['heldout'][f]['per_puzzle']) for pr in receipt['compare'] for f in BASES),
            'full_protocol':full,
            'scope':'Replay audits saved FP64 effect observations, not the neural computations producing them.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc15/audit.json'))
    args=p.parse_args()
    raw=args.receipt.read_bytes()
    report=audit(json.loads(raw))
    report['original_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'file':str(args.out),'puzzle_cases':report['n_fresh_puzzle_pair_family_cases'],
                      'full':report['full_protocol'],'receipt_sha256':report['original_receipt_sha256']},indent=2))


if __name__=='__main__':main()
