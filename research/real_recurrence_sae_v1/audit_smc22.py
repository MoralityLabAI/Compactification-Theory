"""Independent stdlib SMC-22 source-only weighted median + raw causal audit.

No neural/PyTorch experiment imports or GPU processing. Replays original
SMC19/SMC21 file SHA, original calibration 48D gradients+native erasures,
two weighted medians per pair, new unique Sudoku masks and decoder geometry,
three predictor paired bootstrap intervals, all registered gates. The actual
neural forward and autograd remain source experimental observations.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc21 import audit_calibration,audit_fresh_goal
from smc13_family import valid_solution,rectangle_count,count_solutions

PRECOMMIT='b208c1e7a63ac3e0d3008564d25bded5a2f80f52'
SMC19_SHA='612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b'
SMC21_SHA='e7aa12a3da5b8eaa0132b5de45083ca1b2af5c38e556fef01e1f26d2b94f90af'
FAMILIES={'smc22_novel_30':(202622030000,30),
          'smc22_novel_34':(202622040000,34)}
PAIRS=(('newB_501_SAE_71','newB_501_SAE_131'),
       ('newB_501_SAE_71','newB_501_SAE_167'),
       ('newB_701_SAE_71','newB_701_SAE_131'),
       ('newB_701_SAE_71','newB_701_SAE_167'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def close(a,b,tol=2e-7):
    if a is None or b is None:return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b) and
            abs(a-b)<=tol*max(1.,abs(a),abs(b)))


def sign_eps(a):
    return 0 if abs(a)<=1e-8 else (1 if a>0 else -1)


def replay_one_median(original_rows,key,prior_ridge,fit):
    items=[];n=0;np=0
    for row in original_rows:
        occupied=False
        for slot,m in enumerate(row['metrics']):
            if m is None:continue
            n+=1;occupied=True
            x=float(m[key]);y=float(m['target_finite_residual'])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('original calibration residual is not finite')
            if x:
                items.append({'ratio':y/x,'weight':abs(x),
                              'puzzle_index':row['puzzle_index'],'goal_slot':slot,
                              'source_residual':x,'target_residual':y})
        np+=int(occupied)
    items.sort(key=lambda r:(r['ratio'],r['puzzle_index'],r['goal_slot']))
    if len(items)!=len(fit['sorted_nonzero_ratio_weight_records']):
        raise ValueError('weighted median calibration sample count differs')
    for a,b in zip(items,fit['sorted_nonzero_ratio_weight_records']):
        for k,v in a.items():
            if isinstance(v,float):
                if not close(v,b[k],1e-8):
                    raise ValueError('reweighted original source calibration data changed')
            elif v!=b[k]:
                raise ValueError('source puzzle/goal sorted median tie-break changed')
    W=sum(r['weight'] for r in items)
    median=0.;winner=None
    degenerate=W<=1e-12
    if not degenerate:
        cumulative=0.
        for i,r in enumerate(items):
            cumulative+=r['weight']
            if 2*cumulative>=W:
                median=r['ratio'];winner=i;break
        if winner is None:raise ValueError('missing original weighted median')
    beta=max(-2.,min(2.,median))
    old=float(prior_ridge['coefficient'])
    loss=lambda slope:sum(
        abs(float(m['target_finite_residual'])-slope*float(m[key]))
        for r in original_rows for m in r['metrics'] if m is not None)
    expected={
      'source_key':key,'coefficient':beta,'unclipped_median':median,
      'winning_sorted_index':winner,'total_weight':W,
      'source_nonzero_count':len(items),
      'calibration_eligible_goals':n,'calibration_eligible_puzzles':np,
      'degenerate':degenerate,
      'identified':n>=24 and np>=20 and not degenerate,
      'calibration_absolute_error_lad':loss(beta),
      'calibration_absolute_error_original_ridge':loss(old),
      'original_202621_source_ridge_coefficient':old,'clip':[-2.,2.]
    }
    if set(fit)!=set(expected)|{'sorted_nonzero_ratio_weight_records'}:
        raise ValueError('original calibration one-scalar LAD receipt fields drifted')
    for k,v in expected.items():
        actual=fit[k]
        if isinstance(v,float):
            if not close(v,actual,1e-7):
                raise ValueError('weighted median original calibration scalar changed: '+k)
        elif v!=actual:
            raise ValueError('weighted median original source count/tie changed: '+k)


def audit_calibration_lad(original_cal,lad,freeze_dir):
    if (lad['schema']!='SMC-22/original-source-only-LAD-freeze-v1'
        or lad['pre_registration_commit']!=PRECOMMIT
        or lad['original_smc19_frozen_refit_json_sha256']!=SMC19_SHA
        or lad['original_smc21_calibration_json_sha256']!=SMC21_SHA
        or lad['source_only_calibration_frozen_before_new_tests'] is not True
        or lad['calibration_puzzles']!=64 or len(lad['pairs'])!=4):
        raise ValueError('original source-only LAD calibration schema/lock changed')
    source_bytes=(Path(freeze_dir)/'frozen.json').read_bytes()
    if sha(source_bytes)!=SMC19_SHA:
        raise ValueError('original SMC19 refit checkpoint frozen.json SHA drift')
    refits=json.loads(source_bytes)
    result=audit_calibration(original_cal,refits,freeze_dir)
    for i,(pair,orig) in enumerate(zip(lad['pairs'],original_cal['pairs'])):
        if ((pair['source'],pair['target'])!=PAIRS[i] or
            pair['frozen_SMC19_mapping_hashes']!=orig['frozen_original_mapping_hashes']
            or pair['original_ridge_source']!=orig['beta_source']
            or pair['original_ridge_other']!=orig['beta_other']):
            raise ValueError('calibration mapping/ridge source not unchanged')
        replay_one_median(orig['original_64_puzzle_calibration'],
                          'source_finite_residual',orig['beta_source'],pair['lad_source'])
        replay_one_median(orig['original_64_puzzle_calibration'],
                          'other_finite_residual',orig['beta_other'],pair['lad_other'])
    return {'calibration_schema':'SMC-22/independent-stdlib-original-LAD-audit-v1',
            'original_SMC21_ridge_gradients_and_four_SMC19_refits_reaudited':True,
            'two_exact_lower_weighted_medians_per_pair_verified':True,
            'registered_four_pair_fits':[
                {'target':p['target'],'source_LAD':p['lad_source']['coefficient'],
                 'other_LAD':p['lad_other']['coefficient']}
                for p in lad['pairs']],
            'parent_audit':result}


def audit_rosters(receipt):
    seenx=set();seeny=set()
    count=receipt['puzzles_per_family']
    if count not in (4,64):raise ValueError('new SMC22 test count outside registered limits')
    for name,(base,holes) in FAMILIES.items():
        rows=receipt['fresh_rosters'][name]
        fingerprint=sha(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())
        if (len(rows)!=count or receipt['test_seed_bases'][name]!=base or
            fingerprint!=receipt['fresh_manifest_sha256'][name]):
            raise ValueError('SMC22 fresh puzzle source manifest/seed differs')
        for j,row in enumerate(rows):
            x,y=row['x'],row['target']
            if (row['seed']!=base+j or row['holes']!=holes or
                x.count(0)!=holes or not valid_solution(y) or
                row['rectangles']!=rectangle_count(y) or rectangle_count(y)<=0 or
                count_solutions(x,2)!=1 or any(v and v!=y[i] for i,v in enumerate(x))):
                raise ValueError('new exact unique noncanonical Sudoku source invalid')
            if tuple(x) in seenx or tuple(y) in seeny:
                raise ValueError('SMC22 duplicate new puzzle input/full solution')
            seenx.add(tuple(x));seeny.add(tuple(y))


def independent_score(rows,fit,seed):
    puzzle_rows=[]
    for page in rows:
        if len(page['eligible_mask'])!=8 or len(page['metrics'])!=8:
            raise ValueError('selected-goal row size changed')
        n=0;gains=[0.,0.,0.];loss=[0.]*4;correct=[0]*4
        for active,m in zip(page['eligible_mask'],page['metrics']):
            if bool(active)!=(m is not None):
                raise ValueError('ineligible SMC22 goal has target effect')
            if not active:continue
            n+=1;y=m['target_native_drop']
            predictions=[m['direct_prediction'],m['ridge_source_prediction'],
                         m['source_calibrated_prediction'],m['other_calibrated_prediction']]
            err=[abs(y-x) for x in predictions]
            for i in range(4):
                loss[i]+=err[i];correct[i]+=int(sign_eps(y)==sign_eps(predictions[i]))
            gains[0]+=err[0]-err[2]
            gains[1]+=err[1]-err[2]
            gains[2]+=err[3]-err[2]
        puzzle_rows.append((n,*gains,*loss,*correct))
    N=sum(g[0] for g in puzzle_rows)
    cluster=sum(g[0]>0 for g in puzzle_rows)
    if not N:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed)
    boots=[[],[],[]]
    for _ in range(2000):
        sample=[rng.randrange(len(rows)) for row in rows]
        denominator=sum(puzzle_rows[i][0] for i in sample)
        if denominator:
            for j in range(3):
                boots[j].append(sum(puzzle_rows[i][1+j] for i in sample)/denominator)
    for draws in boots:draws.sort()
    valid=all(len(draws)==2000 for draws in boots)
    ci=[[x[50],x[1949]] for x in boots] if valid else [None]*3
    ld=sum(x[4] for x in puzzle_rows)
    ls=sum(x[6] for x in puzzle_rows)
    ratio=ls/ld if ld>1e-8 else None
    signs=[sum(x[8+j] for x in puzzle_rows) for j in range(4)]
    prior_ok=(fit['lad_source']['identified'] and fit['lad_other']['identified']
              and fit['original_ridge_source']['identified'])
    coverage=bool(prior_ok and N>=24 and cluster>=20 and valid)
    three=bool(valid and all(x[0]>0 for x in ci))
    mae=ratio is not None and ratio<=.95
    sign=signs[2]>=signs[0]
    return {
       'status':'measured' if valid else 'bootstrap_abstention',
       'eligible_goals':N,'eligible_puzzles':cluster,
       'mean_gain_vs_direct':sum(x[1] for x in puzzle_rows)/N,
       'mean_gain_vs_original_ridge':sum(x[2] for x in puzzle_rows)/N,
       'mean_gain_vs_other_LAD':sum(x[3] for x in puzzle_rows)/N,
       'direct_MAE':ld/N,'ridge_source_MAE':sum(x[5] for x in puzzle_rows)/N,
       'source_LAD_MAE':ls/N,'other_LAD_MAE':sum(x[7] for x in puzzle_rows)/N,
       'source_LAD_over_direct_MAE_ratio':ratio,
       'direct_sign_correct':signs[0],'ridge_sign_correct':signs[1],
       'source_LAD_sign_correct':signs[2],'other_LAD_sign_correct':signs[3],
       'direct_sign_accuracy':signs[0]/N,'ridge_sign_accuracy':signs[1]/N,
       'source_LAD_sign_accuracy':signs[2]/N,'other_LAD_sign_accuracy':signs[3]/N,
       'bootstrap95_gain_vs_direct':ci[0],
       'bootstrap95_gain_vs_original_ridge':ci[1],
       'bootstrap95_gain_vs_other_LAD':ci[2],
       'valid_bootstrap_draws':len(boots[0]),
       'bootstrap_requested_draws':2000,
       'registered_original_calibration_and_test_coverage_gate':coverage,
       'registered_three_positive_paired_CI_gate':three,
       'registered_relative_direct_MAE_gate':mae,
       'registered_sign_nondegradation_gate':sign,
       'registered_success':bool(coverage and three and mae and sign)
    }


def audit_full(receipt,original_cal,lad,freeze_dir,lad_raw):
    if (receipt['schema']!='SMC-22/weighted-median-source-residual-fresh-test-v1'
        or receipt['pre_registration_commit']!=PRECOMMIT
        or receipt['original_SMC19_refit_json_sha256']!=SMC19_SHA
        or receipt['original_SMC21_calibration_json_sha256']!=SMC21_SHA
        or receipt['SMC22_frozen_LAD_json_sha256']!=sha(lad_raw)
        or receipt['original_SMC21_source_calibration']!=original_cal
        or receipt['frozen_SMC22_LAD_calibration']!=lad
        or receipt['no_new_SAE_or_teacher_training'] is not True):
        raise ValueError('SMC22 full original frozen calibration provenance mismatch')
    calibration_audit=audit_calibration_lad(original_cal,lad,freeze_dir)
    if receipt['source_original_receipts']!=original_cal['source_original_receipts']:
        raise ValueError('original SMC19/S21 trained teacher source SHA changed')
    params=receipt['fixed_parameters']
    if params!={'bootstrap_draws':2000,'direct_MAE_ratio_gate':.95,
                'sign_zero_epsilon':1e-8,'LAD_clip':[-2.,2.],
                'degenerate_LAD_source_weight_cutoff':1e-12}:
        raise ValueError('original registered frozen SMC22 scoring threshold differs')
    full=receipt['puzzles_per_family']==64 and len(receipt['pairs'])==4
    if receipt['full_protocol']!=full:
        raise ValueError('SMC22 four-puzzle pilot/full schema altered')
    audit_rosters(receipt)
    cases=[];n_puzzles=0
    for index,pair in enumerate(receipt['pairs']):
        if index>=4 or (pair['source'],pair['target'])!=PAIRS[index]:
            raise ValueError('SMC22 source SAE reference/refit roster differs')
        fit=lad['pairs'][index]
        if (pair['original_SMC19_matched_mapping_sha256']!=fit['frozen_SMC19_mapping_hashes'] or
            not close(pair['frozen_source_LAD'],fit['lad_source']['coefficient']) or
            not close(pair['frozen_other_LAD'],fit['lad_other']['coefficient']) or
            not close(pair['original_source_ridge'],fit['original_ridge_source']['coefficient'])):
            raise ValueError('SMC22 original calibration source/ridge/LAD scalars differ')
        betas={'beta_source':{'coefficient':fit['lad_source']['coefficient']},
               'beta_other':{'coefficient':fit['lad_other']['coefficient']}}
        mapping=original_cal['pairs'][index]['frozen_original_mapping_hashes']
        original_map=original_cal['pairs'][index]
        # The complete frozen 96-permutation is read from the immutable original
        # SMC19 receipt and its SHA-verified four-refit frozen.json.
        refits=json.loads((Path(freeze_dir)/'frozen.json').read_bytes())
        assignment=refits['pair_calibration'][index]['mapping']['matched']
        for fam in FAMILIES:
            block=pair['heldout'][fam];rows=block['per_puzzle']
            roster=receipt['fresh_rosters'][fam]
            cover=block['coverage']
            ca=cover['reference_decoder_columns_96x48']
            cb=cover['refit_decoder_columns_96x48']
            if len(rows)!=len(roster):
                raise ValueError('SMC22 new puzzle/target scorer missing rows')
            n_puzzles+=len(rows)
            count=0;reasons={}
            for i,(row,source) in enumerate(zip(rows,roster)):
                missing=[j for j,v in enumerate(source['x']) if v==0]
                expected_pos=[missing[j*len(missing)//8] for j in range(8)]
                if (row['puzzle_index']!=i or row['seed']!=source['seed'] or
                    row['goal_positions']!=expected_pos or
                    row['goal_truth_digits']!=[source['target'][j] for j in expected_pos]):
                    raise ValueError('SMC22 source selection uses target label or wrong puzzle')
                for key in ('source_feature_ids','target_feature_ids','other_feature_ids',
                    'reference_top1_coefficients','reference_top2_coefficients',
                    'target_matched_coefficients','other_coefficients',
                    'active_refit_feature_ids','eligible_mask','preoutcome_reasons','metrics'):
                    if len(row[key])!=8:
                        raise ValueError('SMC22 eight selected goal raw source field missing')
                for slot in range(8):
                    aid=row['source_feature_ids'][slot]
                    tid=row['target_feature_ids'][slot]
                    if not 0<=aid<96 or assignment[aid]!=tid:
                        raise ValueError('SMC22 original frozen SMC19 feature match drifted')
                    active=row['active_refit_feature_ids'][slot]
                    strict=(row['reference_top1_coefficients'][slot]>0 and
                            row['reference_top1_coefficients'][slot]>row['reference_top2_coefficients'][slot])
                    cz=row['target_matched_coefficients'][slot]
                    expected_reason=('reference_not_strict_top1' if not strict else
                        'refit_matched_inactive' if cz<=0 else
                        'refit_missing_other_active' if len(active)<2 else 'eligible')
                    if (row['preoutcome_reasons'][slot]!=expected_reason or
                        bool(row['eligible_mask'][slot])!=(expected_reason=='eligible')):
                        raise ValueError('source original eligibility/other active altered')
                    reasons[expected_reason]=reasons.get(expected_reason,0)+1
                    item=row['metrics'][slot]
                    if expected_reason=='eligible':
                        count+=1
                        if item is None:raise ValueError('eligible SAE refit target effect missing')
                        audit_fresh_goal(item,row,slot,index,fam,ca,cb,betas)
                        ea=item['target_native_drop'];p=item['ridge_source_prediction']
                        should=item['direct_prediction']+fit['original_ridge_source']['coefficient']*item['source_finite_residual']
                        if (not close(p,should) or
                            not close(item['ridge_absolute_error'],abs(ea-p)) or
                            not close(item['gain_vs_ridge'],abs(ea-p)-item['source_absolute_error']) or
                            item['ridge_sign_correct']!=(sign_eps(ea)==sign_eps(p))):
                            raise ValueError('registered SMC21 ridge prediction baseline wrong')
                    elif item is not None:
                        raise ValueError('ineligible SMC22 feature got target neural outcome')
            if (count!=cover['eligible_goals'] or
                cover['selected_goals']!=8*len(rows) or
                cover['preoutcome_reason_counts']!=reasons):
                raise ValueError('SMC22 natural activation mask/coverage denominator drifted')
            stats=independent_score(rows,fit,202622130+100*index+len(fam))
            reported=block['statistics']
            if set(stats)!=set(reported):
                raise ValueError('SMC22 score schema differs from independent replay')
            for key,v in stats.items():
                actual=reported[key]
                if isinstance(v,list):
                    if not isinstance(actual,list) or len(v)!=len(actual) or any(not close(a,b) for a,b in zip(v,actual)):
                        raise ValueError('registered source LAD cluster bootstrap interval differs')
                elif isinstance(v,float) or v is None:
                    if not close(v,actual):raise ValueError('registered LAD paired scorer differs: '+key)
                elif v!=actual:
                    raise ValueError('registered LAD pair gate/coverage changed: '+key)
            cases.append({'target':pair['target'],'family':fam,
                          'eligible_goals':count,
                          'mean_LAD_gain_vs_direct':stats.get('mean_gain_vs_direct'),
                          'mean_LAD_gain_vs_ridge':stats.get('mean_gain_vs_original_ridge'),
                          'mean_LAD_gain_vs_other_LAD':stats.get('mean_gain_vs_other_LAD'),
                          'registered_success':stats['registered_success']})
        success=bool(full and all(pair['heldout'][f]['statistics']['registered_success'] for f in FAMILIES))
        if success!=pair['registered_both_new_families_success']:
            raise ValueError('SMC22 source-LAD both-family original gate altered')
    return {
      'schema':'SMC-22/independent-stdlib-source-LAD-and-fresh-test-audit-v1',
      'original_calibration_and_refit_checkpoint_hash_audited':True,
      'all_eight_original_lower_weighted_median_coefficients_recomputed':True,
      'new_exact_unique_Sudoku_rosters_and_visible_goal_masks_verified':True,
      'source_native_physical_edits_and_48D_predictions_recomputed':True,
      'all_24000_puzzle_cluster_paired_bootstrap_values_recomputed':True,
      'n_pair_family_puzzle_cases':n_puzzles,
      'full_protocol':full,'outcomes':cases,
      'scope':'Neural FP64 teacher forward, native erasure and clean task gradients are original experiment observations, independently re-audited only through saved numerical receipts.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('calibration','full'),required=True)
    p.add_argument('--smc21-calibration',required=True,type=Path)
    p.add_argument('--lad-calibration',required=True,type=Path)
    p.add_argument('--frozen-dir',required=True,type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    sraw=a.smc21_calibration.read_bytes()
    lraw=a.lad_calibration.read_bytes()
    if sha(sraw)!=SMC21_SHA:
        raise ValueError('original historical SMC21 calibration SHA mismatch')
    original=json.loads(sraw);lad=json.loads(lraw)
    if a.phase=='calibration':
        result=audit_calibration_lad(original,lad,a.frozen_dir)
    else:
        if a.receipt is None:raise ValueError('full audit needs new test JSON')
        raw=a.receipt.read_bytes()
        result=audit_full(json.loads(raw),original,lad,a.frozen_dir,lraw)
        result['raw_full_json_sha256']=sha(raw)
    result['frozen_source_LAD_json_sha256']=sha(lraw)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
