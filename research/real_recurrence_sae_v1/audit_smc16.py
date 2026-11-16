"""SMC-16 independent stdlib replay for frozen selective-SAE-kill receipts.

Recomputes matched assignments, all seven prior mapping SHA fingerprints,
new Sudoku manifests, goal selection, pre-outcome eligibility, geometric
re-encoding/dose guards, paired heldout margin statistics and clustered
bootstraps. It does not execute the source PyTorch neural checkpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc14 import ORIGINAL_CALIB_SHA
from smc13_family import valid_solution, rectangle_count, count_solutions

SOURCES={
 'source_receipt_sha256':'7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087',
 'independent_training_receipt_sha256':'1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3',
 'conditioned_mapping_receipt_sha256':'3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07',
}
PAIRS=(('pretrainedA_71','newB_501_SAE_71'),
       ('pretrainedA_71','newB_501_SAE_89'),
       ('pretrainedA_71','newB_701_SAE_71'),
       ('pretrainedA_71','newB_701_SAE_89'))
FAMILIES={'smc16_novel_30':(202616030000,30),
          'smc16_novel_34':(202616040000,34)}


def equal(a,b,tolerance=1e-10):
    if a is None or b is None:
        return a is None and b is None
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a-b)<=tolerance*max(1,abs(a),abs(b))


def check_rosters(r,full):
    cohort=r['fresh_rosters']
    seen_solutions=set();seen_inputs=set()
    for name,(base,holes) in FAMILIES.items():
        items=cohort[name]
        if len(items)!=(64 if r['test_puzzles_per_cohort']==64 else 4):
            raise ValueError('fresh cohort size mismatch')
        if r['test_roster_seed_bases'][name]!=base:
            raise ValueError('new cohort seed base differs from preregistration')
        sha=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if sha!=r['fresh_manifest_sha256'][name]:
            raise ValueError('fresh-roster manifest hash mismatch')
        for k,item in enumerate(items):
            if item['seed']!=base+k or item['holes']!=holes:
                raise ValueError('unregistered puzzle seed or hole count')
            x,y=item['x'],item['target']
            if len(x)!=81 or len(y)!=81 or x.count(0)!=holes:
                raise ValueError('incorrect puzzle dimensions or blank count')
            if not valid_solution(y) or rectangle_count(y)!=item['rectangles'] or item['rectangles']<=0:
                raise ValueError('invalid or canonical Sudoku completion')
            if any(d!=0 and d!=y[j] for j,d in enumerate(x)) or count_solutions(x,2)!=1:
                raise ValueError('invalid or nonunique Sudoku puzzle')
            xt,yt=tuple(x),tuple(y)
            if xt in seen_inputs or yt in seen_solutions:
                raise ValueError('repeat SMC16 puzzle or full solution')
            seen_inputs.add(xt);seen_solutions.add(yt)


def check_mappings(r):
    mappings=r['all_seven_original_smc13c_mapping_sha256']
    if len(mappings)!=7 or not r['all_seven_original_smc13c_hungarian_assignments_sha_verified']:
        raise ValueError('not all frozen prior Hungarian assignments verified')
    for i,entry in enumerate(mappings):
        for arm in ('matched_assignment_sha256','clue_shuffled_assignment_sha256'):
            expected=ORIGINAL_CALIB_SHA['matched' if arm=='matched_assignment_sha256' else 'clue_shuffled'][i]
            if entry[arm]!=expected:
                raise ValueError('original SMC13C assignment fingerprint changed')
        if not entry['matched_to_original_smc13c_receipt']:
            raise ValueError('original correspondence provenance missing')


def check_raw(rows,roster,mapping):
    if len(rows)!=len(roster):
        raise ValueError('puzzle data count mismatch')
    reasons=Counter()
    eligible_total=0
    for index,(row,puzzle) in enumerate(zip(rows,roster)):
        if row['puzzle_index']!=index or row['seed']!=puzzle['seed']:
            raise ValueError('source roster order changed')
        blanks=[j for j,v in enumerate(puzzle['x']) if v==0]
        expected=[blanks[(t*len(blanks))//8] for t in range(8)]
        if row['goal_positions']!=expected:
            raise ValueError('goal selection changed after preregistration')
        if row['goal_truth_digits']!=[puzzle['target'][j] for j in expected]:
            raise ValueError('goal truth digits differ from source solution')
        keys=('source_top_feature_ids','target_matched_feature_ids',
              'eligible_mask','edit_diagnostics','kill_margin_drop',
              'sham_margin_drop','clean_correct',
              'kill_correct_to_wrong','sham_correct_to_wrong')
        if any(len(row[key])!=8 for key in keys):
            raise ValueError('each puzzle must contain eight predeclared goal slots')
        for j in range(8):
            a,b=row['source_top_feature_ids'][j],row['target_matched_feature_ids'][j]
            if not 0<=a<96 or not 0<=b<96 or mapping[a]!=b:
                raise ValueError('source feature mapping changed')
            info=row['edit_diagnostics'][j]
            status=info['status']
            good=status=='eligible'
            reasons[status]+=1
            if row['eligible_mask'][j]!=good:
                raise ValueError('eligibility differs from preoutcome code guard')
            if good:
                eligible_total+=1
                if (not equal(info['killed_feature_reencoded'],0,1e-7) or
                    info['other_active_max_drift']>1e-4 or
                    info['sham_active_max_drift']>1e-4 or
                    info['original_active_count']!=4):
                    raise ValueError('selective feature-kill encoder cell guard violated')
                if (info['native_decoder_dose']<=0 or
                    info['kill_dose']<=0 or
                    not equal(info['kill_dose'],info['sham_dose'],1e-8) or
                    info['native_dose_multiple']>6.00000001 or
                    not equal(info['native_dose_multiple'],
                              info['kill_dose']/info['native_decoder_dose'],1e-8)):
                    raise ValueError('sham dose and original native dose guard violated')
                k,s=row['kill_margin_drop'][j],row['sham_margin_drop'][j]
                if (k is None or s is None or
                    not math.isfinite(k) or not math.isfinite(s)):
                    raise ValueError('eligible native intervention missing margin')
                for col in ('kill_correct_to_wrong','sham_correct_to_wrong'):
                    if row[col][j] and not row['clean_correct'][j]:
                        raise ValueError('clean-incorrect token labeled as clean-to-wrong')
            else:
                for col in ('kill_margin_drop','sham_margin_drop',
                            'kill_correct_to_wrong','sham_correct_to_wrong'):
                    if row[col][j] is not None:
                        raise ValueError('ineligible goal must abstain')
    return dict(reasons),eligible_total


def independent_score(rows,seed,draws=2000):
    triples=[];clean_correct=[];flip_k=0;flip_s=0
    for row in rows:
        per=[]
        for j,good in enumerate(row['eligible_mask']):
            if not good:continue
            effect=row['kill_margin_drop'][j]-row['sham_margin_drop'][j]
            per.append(effect)
            if row['clean_correct'][j]:
                clean_correct.append(effect)
            flip_k+=int(row['kill_correct_to_wrong'][j])
            flip_s+=int(row['sham_correct_to_wrong'][j])
        triples.append((sum(per),len(per)))
    n=sum(n for _,n in triples)
    clusters=sum(c>0 for _,c in triples)
    if n==0:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'nonempty_puzzle_clusters':0,'registered_success':False}
    rng=random.Random(seed)
    boot=[]
    for _ in range(draws):
        sample=[rng.randrange(len(triples)) for _ in range(len(triples))]
        count=sum(triples[i][1] for i in sample)
        if count:
            boot.append(sum(triples[i][0] for i in sample)/count)
    boot.sort()
    covered=len(boot)==draws
    bounds=[boot[int(.025*draws)],boot[int(.975*draws)-1]] if covered else None
    return {
      'status':'measured' if covered else 'bootstrap_abstained',
      'eligible_goals':n,
      'nonempty_puzzle_clusters':clusters,
      'mean_selective_kill_minus_sham_margin_drop':sum(p[0] for p in triples)/n,
      'unweighted_mean_per_puzzle':sum(s/c for s,c in triples if c)/clusters,
      'puzzle_cluster_bootstrap95':bounds,
      'valid_bootstrap_draws':len(boot),'requested_bootstrap_draws':draws,
      'clean_correct_eligible_goals':len(clean_correct),
      'clean_correct_only_mean_margin_drop_difference':(sum(clean_correct)/len(clean_correct) if clean_correct else None),
      'kill_clean_correct_to_wrong_count':flip_k,
      'sham_clean_correct_to_wrong_count':flip_s,
      'kill_flip_rate_eligible':flip_k/n,
      'sham_flip_rate_eligible':flip_s/n,
      'registered_success':n>=24 and clusters>=20 and covered and bounds[0]>0 and flip_k>flip_s
    }


def audit(receipt):
    if receipt['schema']!='SMC-16/selected-top4-encoder-preserving-sham-v1':
        raise ValueError('wrong experiment schema')
    if receipt['pre_registration_commit']!='777ff633010898103d876b54ac5d84264c12a628':
        raise ValueError('not the pre-registered SMC16 protocol')
    for field,sha in SOURCES.items():
        if receipt[field]!=sha:
            raise ValueError('frozen source artifact differs: '+field)
    frozen=receipt['fixed_parameters']
    expected={'overshoot':0.02,'native_dose_multiple_limit':6.0,
              'encoder_recode_tolerance':1e-4,'orthogonality_tolerance':1e-8,
              'bootstrap_resamples':2000}
    if frozen!=expected:
        raise ValueError('experimental hyperparameters changed')
    check_mappings(receipt)
    full=(len(receipt['comparison_roster'])==4 and receipt['test_puzzles_per_cohort']==64)
    if receipt['full_protocol']!=full:
        raise ValueError('full-protocol/pilot label differs')
    check_rosters(receipt,full)
    outcomes=[]
    for index,comp in enumerate(receipt['comparison_roster']):
        if (comp['source'],comp['target'])!=PAIRS[index]:
            raise ValueError('primary source-target roster differs')
        assignment=comp['frozen_matched_feature_mapping']
        if sorted(assignment)!=list(range(96)):
            raise ValueError('frozen mapping not a 96-feature permutation')
        sha=hashlib.sha256(json.dumps(assignment).encode()).hexdigest()
        if (sha!=ORIGINAL_CALIB_SHA['matched'][index] or
            comp['source_mapping_sha256']!=receipt['all_seven_original_smc13c_mapping_sha256'][index]):
            raise ValueError('correspondence source SHA differs')
        for family in FAMILIES:
            case=comp['heldout'][family]
            raw=case['per_puzzle']
            counts,eligible=check_raw(raw,receipt['fresh_rosters'][family],assignment)
            if (counts!=case['coverage']['ineligibility_reason_counts'] or
                eligible!=case['coverage']['eligible_goals'] or
                case['coverage']['n_selected_blank_goals']!=8*len(raw)):
                raise ValueError('eligibility/coverage receipt differs')
            computed=independent_score(raw,202616130+index*100+len(family))
            for name,value in computed.items():
                reported=case['statistics'].get(name)
                if isinstance(value,list):
                    if not isinstance(reported,list) or len(value)!=len(reported) or any(not equal(x,y) for x,y in zip(value,reported)):
                        raise ValueError('puzzle-cluster bootstrap changed')
                elif isinstance(value,float):
                    if not equal(value,reported):
                        raise ValueError('paired margin score differs: '+name)
                elif reported!=value:
                    raise ValueError('registered outcome differs: '+name)
            outcomes.append({'target':comp['target'],'family':family,
                             'eligible_goals':eligible,'score':computed.get('mean_selective_kill_minus_sham_margin_drop'),
                             'registered_success':computed['registered_success']})
        primary=full and all(comp['heldout'][name]['statistics']['registered_success'] for name in FAMILIES)
        if comp['registered_success_both_families']!=primary:
            raise ValueError('pairwise success differs from both fresh-family gates')
    return {
      'schema':'SMC-16/independent-stdlib-replay-v1',
      'n_primary_pairs':len(receipt['comparison_roster']),
      'n_pair_puzzle_family_cases':sum(len(c['heldout'][fam]['per_puzzle'])
          for c in receipt['comparison_roster'] for fam in FAMILIES),
      'original_seven_alignment_hashes_checked':True,
      'fresh_puzzle_manifest_and_unique_solutions_checked':True,
      'feature_mapping_and_eligibility_recomputed':True,
      'encoded_feature_kill_and_sham_constraints_checked':True,
      'puzzle_bootstrap_and_preregistered_gates_recomputed':True,
      'full_protocol':full,
      'outcomes':outcomes,
      'scope':'Checks saved numerical and geometric receipts, not a separate neural checkpoint forward pass.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc16/audit.json'))
    args=p.parse_args()
    raw=args.receipt.read_bytes()
    result=audit(json.loads(raw))
    result['raw_source_sha256']=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'receipt':result['raw_source_sha256'],
                      'cases':result['n_pair_puzzle_family_cases'],
                      'outcomes':result['outcomes']},indent=2))


if __name__=='__main__':
    main()
