"""SMC-20 independent standard-library native SAE refit prediction auditor.

Reconstruct original and new source locks, four refit checkpoint file SHA256,
fresh exact-unique Sudoku and visible-only goal positions, all frozen
feature/other-active selection and 48D physical edit vectors, gradient
correction, three absolute losses, sign accuracies, 2000 puzzle-bootstrap
replicates and every registered SMC20 pass/fail gate.

No import of the PyTorch experimental runner or neural forward simulation.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc19 import (audit_original,audit_frozen_files,audit_signs,
                         SOURCES,PAIRS as ORIGINAL_PAIRS)
from smc13_family import valid_solution,rectangle_count,count_solutions

PRECOMMIT='eb4b6c419fd741c4915d105235a9979426398710'
FAMILIES={'smc20_novel_30':(202620030000,30),
          'smc20_novel_34':(202620040000,34)}
ZERO=1e-8
GATE_RATIO=.90


def close(a,b,tol=3e-7):
    if a is None or b is None:
        return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b) and
            abs(a-b)<=tol*max(1.,abs(a),abs(b)))


def dot(a,b):
    if len(a)!=len(b):
        raise ValueError('48D physical gradient and edit dimension differ')
    return sum(x*y for x,y in zip(a,b))


def norm(v):
    return math.sqrt(dot(v,v))


def sub(a,b):
    if len(a)!=len(b):raise ValueError('physical vector dimensions differ')
    return [x-y for x,y in zip(a,b)]


def sign_eps(v):
    return 0 if abs(v)<=ZERO else (1 if v>0 else -1)


def sha_json_mapping(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def roster_sha(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_rosters(receipt):
    count=receipt['puzzles_per_family']
    if count not in (4,64):
        raise ValueError('wrong registered SMC20 pilot/full puzzle count')
    sx=set();sy=set()
    for fam,(base,holes) in FAMILIES.items():
        rows=receipt['fresh_rosters'][fam]
        if (len(rows)!=count or receipt['fresh_seed_bases'][fam]!=base
            or roster_sha(rows)!=receipt['fresh_manifest_sha256'][fam]):
            raise ValueError('SMC20 new family source seed or manifest mismatch')
        for i,r in enumerate(rows):
            x=r['x'];y=r['target']
            if (r['seed']!=base+i or r['holes']!=holes or len(x)!=81 or
                x.count(0)!=holes or not valid_solution(y) or
                rectangle_count(y)<=0 or r['rectangles']!=rectangle_count(y)):
                raise ValueError('new unique noncanonical full target changed')
            if any(v and v!=y[j] for j,v in enumerate(x)) or count_solutions(x,2)!=1:
                raise ValueError('SMC20 puzzle not exact unique or clue conflict')
            tx,ty=tuple(x),tuple(y)
            if tx in sx or ty in sy:
                raise ValueError('new SMC20 cohorts have duplicate complete solutions')
            sx.add(tx);sy.add(ty)


def audit_goal(item,row,pair_index,fam,slot,source_cols,target_cols):
    if item['reference_feature']!=row['source_selected_ids'][slot]:
        raise ValueError('source top-one ID changed after predicting target')
    if item['refit_matched_feature']!=row['target_matched_ids'][slot] or item['refit_other_feature']!=row['refit_other_active_ids'][slot]:
        raise ValueError('frozen refit matched/other feature differs')
    ia=item['reference_feature'];ib=item['refit_matched_feature'];ic=item['refit_other_feature']
    if not all(0<=i<96 for i in (ia,ib,ic)) or ib==ic:
        raise ValueError('invalid source or matched/native refit feature ID')
    val_a=row['reference_top1_coefficients'][slot]
    val_b=row['refit_matched_coefficients'][slot]
    val_c=row['refit_other_coefficients'][slot]
    if (val_a<=row['reference_top2_coefficients'][slot] or val_a<=0 or
        val_b<=0 or val_c<=0 or
        not close(val_a,item['reference_selected_coefficient']) or
        not close(val_b,item['refit_selected_coefficient']) or
        not close(val_c,item['refit_other_selected_coefficient'])):
        raise ValueError('source/target original clean activation coefficient guard changed')
    active=row['refit_active_feature_ids'][slot]
    if (len(active)!=4 or sorted(active)!=active or
        ib not in active or ic not in active):
        raise ValueError('receiving refit four natural active features differ')
    choices=[q for q in active if q!=ib]
    msg=f'SMC15|{pair_index}|{fam}|{row["seed"]}|{slot}|{ib}'
    chosen=int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8],'big')%len(choices)
    if ic!=choices[chosen]:
        raise ValueError('frozen preoutcome other-active feature choice tampered')
    da=[val_a*x for x in source_cols[ia]]
    db=[val_b*x for x in target_cols[ib]]
    dc_col=target_cols[ic]
    dc_norm=norm(dc_col)
    if dc_norm<=1e-12:
        raise ValueError('zero receiving control column')
    dc=[x*norm(db)/dc_norm for x in dc_col]
    for key,pred in (('reference_delta_48',da),('target_delta_48',db),
                     ('other_active_delta_48',dc)):
        got=item[key]
        if len(got)!=48 or any(not close(x,y,2e-7) for x,y in zip(got,pred)):
            raise ValueError('physical SAE native decoder edit differs: '+key)
    if not (close(item['reference_physical_dose'],norm(da)) and
            close(item['refit_physical_dose'],norm(db)) and
            close(item['other_physical_dose'],norm(dc)) and
            abs(norm(db)-norm(dc))<=1e-9):
        raise ValueError('original refit native L2 dose failed')
    u=sub(db,da)
    cos=dot(da,db)/(norm(da)*norm(db))
    if (not close(norm(u),item['physical_edit_distance']) or
        not close(cos,item['physical_edit_cosine'])):
        raise ValueError('source/refit physical angle or difference changed')
    g=item['clean_task_gradient_48']
    if len(g)!=48 or any(not math.isfinite(t) for t in g):
        raise ValueError('nonfinite/missing original clean FP64 48D gradient')
    ea=item['observed_reference_drop']; eb=item['observed_target_drop']
    eo=item['observed_other_active_drop']
    pg=ea+dot(g,u)
    po=ea+dot(g,sub(dc,da))
    for key,expected in (
        ('gradient_anchored_prediction',pg),
        ('reference_only_prediction',ea),
        ('other_active_surrogate_prediction',po),
        ('direct_clean_linear_refit_prediction',dot(g,db)),
        ('reference_only_absolute_error',abs(eb-ea)),
        ('other_surrogate_absolute_error',abs(eb-po)),
        ('gradient_anchored_absolute_error',abs(eb-pg)),
        ('absolute_error_gain_over_reference',abs(eb-ea)-abs(eb-pg)),
        ('absolute_error_gain_over_other',abs(eb-po)-abs(eb-pg)),
    ):
        if not close(item[key],expected,2e-7):
            raise ValueError('causal prediction/paired error forged: '+key)
    for name,expected in (
        ('reference_only_predicted_sign',sign_eps(ea)),
        ('other_surrogate_predicted_sign',sign_eps(po)),
        ('gradient_anchored_predicted_sign',sign_eps(pg)),
        ('observed_target_sign',sign_eps(eb)),
    ):
        if item[name]!=expected:
            raise ValueError('SMC20 original zero-sign treatment differs')
    if any(item[x] and not item['clean_correct'] for x in
           ('source_clean_correct_to_wrong','refit_clean_correct_to_wrong',
            'other_clean_correct_to_wrong')):
        raise ValueError('clean-incorrect goal falsely counted as correct-to-wrong')


def audit_condition(comp,fam,roster,index,permutation):
    result=comp['heldout'][fam]
    rows=result['per_puzzle']
    coverage=result['coverage']
    if len(rows)!=len(roster):
        raise ValueError('incomplete original pair/puzzle goal records')
    source_cols=coverage['reference_physical_decoder_columns_96x48']
    target_cols=coverage['refit_physical_decoder_columns_96x48']
    if len(source_cols)!=96 or len(target_cols)!=96 or any(len(z)!=48 for z in source_cols+target_cols):
        raise ValueError('original source/target native SAE decoder columns not archived')
    reasons={}
    n=0
    for i,(row,ro) in enumerate(zip(rows,roster)):
        if row['puzzle_index']!=i or row['seed']!=ro['seed']:
            raise ValueError('new puzzle goal row changed ordering')
        blanks=[k for k,v in enumerate(ro['x']) if v==0]
        chosen=[blanks[k*len(blanks)//8] for k in range(8)]
        if row['goal_positions']!=chosen or row['goal_truth_digits']!=[ro['target'][k] for k in chosen]:
            raise ValueError('visible-mask goal or target differs from source puzzle')
        fields=('source_selected_ids','target_matched_ids','refit_other_active_ids',
                'refit_active_feature_ids','reference_top1_coefficients',
                'reference_top2_coefficients','refit_matched_coefficients',
                'refit_other_coefficients','eligible_mask','preoutcome_reasons','metrics')
        if any(len(row[field])!=8 for field in fields):
            raise ValueError('eight original frozen goal slots missing')
        for j in range(8):
            a=row['source_selected_ids'][j];b=row['target_matched_ids'][j]
            if not 0<=a<96 or permutation[a]!=b:
                raise ValueError('original SMC19 calibration matched permutation changed')
            active=row['refit_active_feature_ids'][j]
            strict=(row['reference_top1_coefficients'][j]>0 and
                    row['reference_top1_coefficients'][j]>row['reference_top2_coefficients'][j])
            z=row['refit_matched_coefficients'][j]
            expected=('reference_not_strict_top1' if not strict else
                      'refit_matched_inactive' if z<=0 else
                      'refit_missing_distinct_active_control' if len(active)<2 or b not in active else
                      'eligible')
            why=row['preoutcome_reasons'][j]
            if why!=expected or bool(row['eligible_mask'][j])!=(why=='eligible'):
                raise ValueError('preoutcome SAE eligibility selection changed')
            reasons[why]=reasons.get(why,0)+1
            if why=='eligible':
                n+=1
                audit_goal(row['metrics'][j],row,index,fam,j,source_cols,target_cols)
            elif row['metrics'][j] is not None:
                raise ValueError('ineligible goal received target outcome')
    if n!=coverage['eligible_goals'] or reasons!=coverage['preoutcome_rejection_counts'] or coverage['selected_goal_tokens']!=len(rows)*8:
        raise ValueError('reported eligible subset or rejection count differs')
    return n


def independent_score(rows,seed):
    stats=[]
    for row in rows:
        n=0;ia=0.;io=0.;la=0.;lo=0.;lg=0.;signs=[0,0,0]
        for yes,item in zip(row['eligible_mask'],row['metrics']):
            if not yes:continue
            n+=1
            obs=item['observed_target_drop'];p=item['gradient_anchored_prediction']
            ref=item['observed_reference_drop'];q=item['other_active_surrogate_prediction']
            a=abs(obs-ref);b=abs(obs-q);c=abs(obs-p)
            ia+=a-c;io+=b-c;la+=a;lo+=b;lg+=c
            signs[0]+=int(sign_eps(obs)==sign_eps(ref))
            signs[1]+=int(sign_eps(obs)==sign_eps(q))
            signs[2]+=int(sign_eps(obs)==sign_eps(p))
        stats.append((n,ia,io,la,lo,lg,*signs))
    total=sum(x[0] for x in stats)
    puzzles=sum(x[0]>0 for x in stats)
    if not total:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed);aa=[];bb=[]
    for _ in range(2000):
        drawn=[rng.randrange(len(rows)) for row in rows]
        count=sum(stats[i][0] for i in drawn)
        if count:
            aa.append(sum(stats[i][1] for i in drawn)/count)
            bb.append(sum(stats[i][2] for i in drawn)/count)
    aa.sort();bb.sort()
    valid=len(aa)==2000 and len(bb)==2000
    ca=[aa[50],aa[1949]] if valid else None
    cb=[bb[50],bb[1949]] if valid else None
    loss_a=sum(x[3] for x in stats)
    loss_g=sum(x[5] for x in stats)
    ratio=loss_g/loss_a if loss_a>ZERO else None
    sa=sum(x[6] for x in stats);so=sum(x[7] for x in stats);sg=sum(x[8] for x in stats)
    coverage=total>=24 and puzzles>=20 and valid
    improvement=bool(valid and ca[0]>0 and cb[0]>0)
    relative=ratio is not None and ratio<=GATE_RATIO
    signs=sg>=sa
    return {
      'status':'measured' if valid else 'bootstrap_abstention',
      'eligible_goals':total,'eligible_puzzles':puzzles,
      'mean_anchor_minus_gradient_absolute_error_gain':sum(x[1] for x in stats)/total,
      'mean_other_surrogate_minus_gradient_absolute_error_gain':sum(x[2] for x in stats)/total,
      'mean_reference_only_absolute_error':loss_a/total,
      'mean_other_surrogate_absolute_error':sum(x[4] for x in stats)/total,
      'mean_gradient_anchored_absolute_error':loss_g/total,
      'relative_gradient_to_reference_absolute_error':ratio,
      'reference_only_sign_accuracy':sa/total,
      'other_surrogate_sign_accuracy':so/total,
      'gradient_anchored_sign_accuracy':sg/total,
      'reference_only_sign_correct':sa,'other_surrogate_sign_correct':so,
      'gradient_anchored_sign_correct':sg,
      'puzzle_cluster_bootstrap95_gain_over_reference':ca,
      'puzzle_cluster_bootstrap95_gain_over_other':cb,
      'valid_bootstrap_resamples':len(aa),
      'requested_bootstrap_resamples':2000,
      'registered_coverage_gate':coverage,
      'registered_both_bootstrap_improvement_gate':improvement,
      'registered_relative_mae_gate':relative,
      'registered_sign_nondegradation_gate':signs,
      'registered_success':bool(coverage and improvement and relative and signs),
    }


def audit(receipt,frozen_dir):
    frozen_dir=Path(frozen_dir)
    frozen_bytes=(frozen_dir/'frozen.json').read_bytes()
    frozen=json.loads(frozen_bytes)
    if receipt['schema']!='SMC-20/source-frozen-anchored-task-gradient-refit-v1':
        raise ValueError('unexpected SMC20 receipt format')
    if receipt['pre_registration_commit']!=PRECOMMIT:
        raise ValueError('original SMC20 preregistration SHA changed')
    if receipt['smc19_frozen_json_sha256']!='612b7bd0bd206043d0766b398e85dd3e8427ecdbf4cbd4c0db52b':
        raise ValueError('expected original frozen refit SHA differs')
    if (hashlib.sha256(frozen_bytes).hexdigest()!=receipt['smc19_frozen_json_sha256']
        or receipt['smc19_frozen_source']!=frozen):
        raise ValueError('SMC19 checkpoint and calibration source does not match frozen archive')
    if receipt['source_original_receipts']!=frozen['source_original_receipts']:
        raise ValueError('source pretrained/independent teacher SHA changed')
    audit_original(frozen)
    audit_frozen_files(frozen,frozen_dir)
    audit_signs(frozen)
    if receipt['old_original_seven_mappings_hash_checked'] is not True or receipt['refit_calibrations_locked_before_new_test'] is not True:
        raise ValueError('original mapping/refit calibration not frozen before new heldout')
    settings=receipt['fixed_parameters']
    expected={'bootstrap_draws':2000,'numerical_sign_cutoff':1e-8,
              'relative_absolute_error_gate':.9,'physical_tolerance':1e-10,
              'gradient_forward_tolerance':2e-8}
    if settings!=expected:
        raise ValueError('SMC20 registered prediction scoring parameters changed')
    full=receipt['puzzles_per_family']==64 and len(receipt['pairs'])==4
    if receipt['full_protocol']!=full:
        raise ValueError('SMC20 pilot/full classification differs')
    verify_rosters(receipt)
    outcomes=[];cases=0
    for index,pair in enumerate(receipt['pairs']):
        if index>=4 or (pair['source'],pair['target'])!=ORIGINAL_PAIRS[index]:
            raise ValueError('SMC20 original four refit comparison ordering changed')
        cal=frozen['pair_calibration'][index]
        if (pair['frozen_calibration_mapping']!=cal['mapping']['matched'] or
            pair['frozen_mapping_hashes']!=cal['mapping_hashes'] or
            pair['frozen_smc19_calibration_orientations']!=cal['calibration']['orientation']):
            raise ValueError('source-pinned SMC19 calibration/hash receipt differs')
        if sha_json_mapping(pair['frozen_calibration_mapping'])!=pair['frozen_mapping_hashes']['matched_sha256']:
            raise ValueError('SMC20 source conditioned feature assignment corrupted')
        for fam in FAMILIES:
            block=pair['heldout'][fam]
            n=audit_condition(pair,fam,receipt['fresh_rosters'][fam],index,pair['frozen_calibration_mapping'])
            rows=block['per_puzzle']
            cases+=len(rows)
            computed=independent_score(rows,202620130+100*index+len(fam))
            published=block['statistics']
            if set(published)!=set(computed):
                raise ValueError('registered SMC20 score schema/keys differ')
            for key,value in computed.items():
                actual=published[key]
                if isinstance(value,list):
                    if not isinstance(actual,list) or len(value)!=len(actual) or any(not close(a,b) for a,b in zip(value,actual)):
                        raise ValueError('puzzle-cluster bootstrap altered: '+key)
                elif isinstance(value,float) or value is None:
                    if not close(value,actual):
                        raise ValueError('source-conditioned prediction mean changed: '+key)
                elif value!=actual:
                    raise ValueError('registered gate/count changed: '+key)
            if n!=computed['eligible_goals']:
                raise ValueError('physical feature eligibility denominator differs')
            outcomes.append({'target':pair['target'],'family':fam,
                             'eligible_goals':n,
                             'gradient_vs_reference_gain':computed.get('mean_anchor_minus_gradient_absolute_error_gain'),
                             'gradient_vs_other_gain':computed.get('mean_other_surrogate_minus_gradient_absolute_error_gain'),
                             'relative_absolute_error':computed.get('relative_gradient_to_reference_absolute_error'),
                             'registered_family_success':computed['registered_success']})
        both=bool(full and all(pair['heldout'][fam]['statistics']['registered_success'] for fam in FAMILIES))
        if both!=pair['registered_both_family_success']:
            raise ValueError('SMC20 original pair two-family decision altered')
    return {
      'schema':'SMC-20/independent-stdlib-physical-gradient-refit-replay-v1',
      'source_original_frozen_training_and_fitted_SAE_file_SHA_checked':True,
      'all_seven_original_and_four_refit_calibrated_assignments_checked':True,
      'new_unique_noncanonical_Sudoku_rosters_checked':True,
      'preoutcome_native_physical_dose_and_alternative_selection_recomputed':True,
      'all_48D_physical_gradient_prediction_errors_recomputed':True,
      'all_16000_puzzle_cluster_resamples_and_gate_decisions_replayed':True,
      'n_pair_family_puzzle_cases':cases,
      'n_selected_blank_goals':cases*8,
      'full_protocol':full,'outcomes':outcomes,
      'scope':'Uses original measured native-erasure margins and FP64 clean gradients as inputs; no independent neural forward or autograd repeat.'
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--frozen-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('runs/smc20/audit.json'))
    args=parser.parse_args()
    content=args.receipt.read_bytes()
    result=audit(json.loads(content),args.frozen_dir)
    result['full_json_sha256']=hashlib.sha256(content).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'full':result['full_protocol'],
                      'pair_family_scores':result['outcomes']},indent=2))


if __name__=='__main__':main()
