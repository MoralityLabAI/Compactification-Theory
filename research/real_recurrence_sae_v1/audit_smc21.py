"""Independent standard-library SMC-21 residual calibration and heldout replay.

Both phases source-lock original SMC19 four refit checkpoints and original
seven SMC13C Hungarian assignments. Calibration replay verifies per-goal
physical gradients, original causal response, residuals and both fixed
one-dimensional ridge coefficients. Full replay checks fresh exact-unique
Sudoku/goal masks, native decoder physical contribution geometry, same-goal
prediction errors, sign convention and 2000-cluster bootstrap gates.

The independent auditor does NOT reexecute the neural teacher forward or
its FP64 autograd. Those inputs remain source-locked observations.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from audit_smc19 import audit_original,audit_frozen_files,audit_signs
from smc13_family import valid_solution,rectangle_count,count_solutions

PRECOMMIT='0f0af2dc4b34c6a85dc4e0ba5d5d1e08597bbf59'
FROZEN19_SHA='612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b'
FAMILIES={'smc21_novel_30':(202621030000,30),
          'smc21_novel_34':(202621040000,34)}
PAIRS=(('newB_501_SAE_71','newB_501_SAE_131'),
       ('newB_501_SAE_71','newB_501_SAE_167'),
       ('newB_701_SAE_71','newB_701_SAE_131'),
       ('newB_701_SAE_71','newB_701_SAE_167'))
ZERO=1e-8


def sha(x):
    return hashlib.sha256(x).hexdigest()


def close(a,b,tol=2e-7):
    if a is None or b is None:return a is None and b is None
    return (math.isfinite(a) and math.isfinite(b) and
            abs(a-b)<=tol*max(1.,abs(a),abs(b)))


def dot(a,b):
    if len(a)!=len(b):raise ValueError('physical decoder/gradient mismatch')
    return sum(x*y for x,y in zip(a,b))


def norm(a):
    return math.sqrt(dot(a,a))


def difference(a,b):
    if len(a)!=len(b):raise ValueError('physical decoder dimensions differ')
    return [x-y for x,y in zip(a,b)]


def sign_eps(x):
    return 0 if abs(x)<=ZERO else (1 if x>0 else -1)


def check_native(da,db,do,coeffs,ids,cols_a,cols_b):
    src,refit,other=ids
    za,zb,zo=coeffs
    if (not all(0<=j<96 for j in ids) or refit==other or
        za<=0 or zb<=0 or zo<=0 or
        len(cols_a)!=96 or len(cols_b)!=96 or
        any(len(q)!=48 for q in cols_a+cols_b) or
        any(len(q)!=48 for q in (da,db,do))):
        raise ValueError('invalid source/refit SAE native physical edit')
    expected_a=[za*v for v in cols_a[src]]
    expected_b=[zb*v for v in cols_b[refit]]
    base=cols_b[other];length=norm(base)
    if length<=1e-12:
        raise ValueError('zero receiving other-active physical decoder')
    expected_o=[v*norm(expected_b)/length for v in base]
    for actual,expected in zip((da,db,do),(expected_a,expected_b,expected_o)):
        if any(not close(x,y) for x,y in zip(actual,expected)):
            raise ValueError('native physical decoder contribution not reproducible')
    if abs(norm(db)-norm(do))>1e-10:
        raise ValueError('receiving same-dose target/other SAE control differs')


def audit_calibration(cal,frozen,old_dir):
    if (cal['schema']!='SMC-21/original-calibration-ridge-freeze-v1'
        or cal['pre_registration_commit']!=PRECOMMIT
        or cal['smc19_frozen_json_sha256']!=FROZEN19_SHA
        or cal['source_original_receipts']!=frozen['source_original_receipts']
        or cal['calibration_puzzles']!=64 or len(cal['pairs'])!=4
        or cal['no_fresh_heldout_states_generated'] is not True):
        raise ValueError('SMC21 original 64-puzzle source calibration schema differs')
    audit_original(frozen)
    audit_frozen_files(frozen,old_dir)
    audit_signs(frozen)
    output=[]
    for index,panel in enumerate(cal['pairs']):
        prior=frozen['pair_calibration'][index]
        if ((panel['source'],panel['target'])!=PAIRS[index] or
            (panel['source'],panel['target'])!=(prior['source'],prior['target'])
            or panel['frozen_original_mapping_hashes']!=prior['mapping_hashes']):
            raise ValueError('SMC21 refit original source feature mapping changed')
        rows=panel['original_64_puzzle_calibration']
        oldrows=prior['calibration']['raw']
        cols_a=panel['reference_decoder_columns_96x48']
        cols_b=panel['target_decoder_columns_96x48']
        if len(rows)!=64 or len(oldrows)!=64:
            raise ValueError('old original source calibration not 64 independent puzzles')
        goal_count=0;puzzle_count=0
        ra=[];rb=[];ro=[]
        mapping=prior['mapping']['matched']
        for ix,(page,original) in enumerate(zip(rows,oldrows)):
            if (page['puzzle_index']!=ix or page['seed']!=original['seed'] or
                page['goal_positions']!=original['goal_positions'] or
                page['eligible_mask']!=original['eligible_mask'] or
                len(page['metrics'])!=8):
                raise ValueError('source original SMC19 calibration eligibility/positions changed')
            any_good=False
            for slot,flag in enumerate(page['eligible_mask']):
                m=page['metrics'][slot]
                if flag!=(m is not None):
                    raise ValueError('original calibration target exposure eligibility mismatch')
                if not flag:continue
                any_good=True;goal_count+=1
                if ((m['source_feature'],m['target_feature'],m['other_feature'])!=
                    (original['source_selected_ids'][slot],
                     original['target_matched_ids'][slot],
                     original['target_other_active_ids'][slot]) or
                    mapping[m['source_feature']]!=m['target_feature']):
                    raise ValueError('calibration old/new feature identities changed')
                if (not close(m['target_clean_coefficient'],
                              original['target_match_clean_coefficients'][slot]) or
                    not close(m['other_clean_coefficient'],
                              original['target_other_active_coefficients'][slot])):
                    raise ValueError('SMC19 calibration receiving coefficients changed')
                check_native(m['source_delta_48'],m['target_delta_48'],
                             m['other_delta_48'],
                             (m['source_clean_coefficient'],
                              m['target_clean_coefficient'],
                              m['other_clean_coefficient']),
                             (m['source_feature'],m['target_feature'],m['other_feature']),
                             cols_a,cols_b)
                g=m['physical_gradient_48']
                if len(g)!=48 or any(not math.isfinite(x) for x in g):
                    raise ValueError('clean task-margin gradient missing')
                if not all(close(m[key],original[prior_key][slot])
                           for key,prior_key in (('source_native_drop','source_drop'),
                                                 ('target_native_drop','matched_drop'),
                                                 ('other_native_drop','other_active_drop'))):
                    raise ValueError('old calibration target native erasures changed')
                a=m['source_native_drop']-dot(g,m['source_delta_48'])
                b=m['target_native_drop']-dot(g,m['target_delta_48'])
                o=m['other_native_drop']-dot(g,m['other_delta_48'])
                for key,val in (('source_finite_residual',a),
                                ('target_finite_residual',b),
                                ('other_finite_residual',o)):
                    if not close(m[key],val):
                        raise ValueError('calibration original finite residual differs: '+key)
                ra.append(a);rb.append(b);ro.append(o)
            puzzle_count+=int(any_good)
        for name,source in [('beta_source',ra),('beta_other',ro)]:
            frozen_fit=panel[name]
            xx=sum(v*v for v in source)
            xy=sum(v*y for v,y in zip(source,rb))
            degenerate=xx<=1e-12
            fit=0. if degenerate else max(-2.,min(2.,xy/(1.05*xx)))
            identified=goal_count>=24 and puzzle_count>=20 and not degenerate and math.isfinite(fit)
            expected={
              'coefficient':fit,'calibration_feature':('source_finite_residual'
                    if name=='beta_source' else 'other_finite_residual'),
              'cross_product_sum':xy,'source_square_sum':xx,
              'ridge_factor':1.05,'clipping_interval':[-2.,2.],
              'degenerate':degenerate,'eligible_goals':goal_count,
              'eligible_puzzles':puzzle_count,'identified':identified}
            if set(expected)!=set(frozen_fit):
                raise ValueError('source calibration ridge schema changed')
            for key,val in expected.items():
                got=frozen_fit[key]
                if isinstance(val,float):
                    if not close(got,val):raise ValueError('fitted ridge drift: '+name+'/'+key)
                elif got!=val:
                    raise ValueError('frozen ridge source/calibration count changed')
        output.append({'source':panel['source'],'target':panel['target'],
                       'eligible_calibration_goals':goal_count,
                       'eligible_calibration_puzzles':puzzle_count,
                       'beta_source':panel['beta_source']['coefficient'],
                       'beta_other':panel['beta_other']['coefficient'],
                       'both_identified':panel['beta_source']['identified'] and panel['beta_other']['identified']})
    return {'schema':'SMC-21/independent-stdlib-original-calibration-audit-v1',
            'all_four_refit_weights_file_sha256_checked':True,
            'original_seven_source_feature_assignments_checked':True,
            'original_four_SMC19_calibration_causal_responses_replayed':True,
            'original_64_puzzle_clean_gradient_ridge_coefficients_recomputed':True,
            'comparisons':output,
            'scope':'Recomputes saved model-data gradients and native erasures from original calibration receipts; does not reexecute PyTorch teacher.'}


def audit_rosters(receipt):
    seenx=set();seeny=set()
    count=receipt['puzzles_per_family']
    if count not in (4,64):raise ValueError('unexpected fresh cohort count')
    for fam,(base,holes) in FAMILIES.items():
        rows=receipt['new_rosters'][fam]
        if (len(rows)!=count or receipt['new_seed_bases'][fam]!=base or
            sha(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())!=receipt['new_manifest_sha256'][fam]):
            raise ValueError('new SMC21 Sudoku roster SHA/seed mismatched')
        for i,row in enumerate(rows):
            x,y=row['x'],row['target']
            if (row['seed']!=base+i or len(x)!=81 or row['holes']!=holes or
                x.count(0)!=holes or not valid_solution(y) or
                rectangle_count(y)<=0 or row['rectangles']!=rectangle_count(y) or
                count_solutions(x,2)!=1 or any(v and v!=y[j] for j,v in enumerate(x))):
                raise ValueError('SMC21 new puzzle uniqueness/clues/noncanonical invariant failed')
            if tuple(x) in seenx or tuple(y) in seeny:
                raise ValueError('SMC21 test cohorts duplicate a new Sudoku state')
            seenx.add(tuple(x));seeny.add(tuple(y))


def audit_fresh_goal(m,row,slot,index,fam,cols_a,cols_b,betas):
    src=m['source_feature'];tgt=m['target_feature'];other=m['other_feature']
    if (src!=row['source_feature_ids'][slot] or
        tgt!=row['target_feature_ids'][slot] or
        other!=row['other_feature_ids'][slot]):
        raise ValueError('SMC21 original matched feature identity differs')
    selected=row['active_refit_feature_ids'][slot]
    if sorted(selected)!=selected or len(selected)!=4 or tgt not in selected or other not in selected or other==tgt:
        raise ValueError('receiving SAE active support/other control invalid')
    choices=[i for i in selected if i!=tgt]
    token=f'SMC15|{index}|{fam}|{row["seed"]}|{slot}|{tgt}'
    expected=choices[int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%len(choices)]
    if other!=expected:raise ValueError('frozen source-blind other-active choice differs')
    a=row['reference_top1_coefficients'][slot]
    b=row['target_matched_coefficients'][slot]
    o=row['other_coefficients'][slot]
    if (a<=0 or a<=row['reference_top2_coefficients'][slot] or
        b<=0 or o<=0 or
        not close(m['reference_coefficient'],a) or
        not close(m['target_coefficient'],b) or
        not close(m['other_coefficient'],o)):
        raise ValueError('SMC21 source-top1 and refit-positive cell violated')
    da=m['reference_delta_48'];db=m['target_delta_48'];do=m['other_delta_48']
    check_native(da,db,do,(a,b,o),(src,tgt,other),cols_a,cols_b)
    for name,vec in (('reference_physical_dose',da),('target_physical_dose',db),
                     ('other_physical_dose',do)):
        if not close(m[name],norm(vec)):
            raise ValueError('physical SAE native edit L2 dose changed: '+name)
    g=m['gradient_48']
    if len(g)!=48 or any(not math.isfinite(x) for x in g):
        raise ValueError('missing/nonfinite answer-conditioned FP64 task gradient')
    ea=m['reference_native_drop'];eo=m['other_native_drop'];eb=m['target_native_drop']
    ra=ea-dot(g,da);ro=eo-dot(g,do)
    direct=dot(g,db)
    source=direct+betas['beta_source']['coefficient']*ra
    control=direct+betas['beta_other']['coefficient']*ro
    for name,val in (
      ('source_finite_residual',ra),
      ('other_finite_residual',ro),
      ('target_finite_residual',eb-direct),
      ('direct_prediction',direct),
      ('source_calibrated_prediction',source),
      ('other_calibrated_prediction',control),
      ('unit_anchor_prediction',direct+ra),
      ('direct_absolute_error',abs(eb-direct)),
      ('source_absolute_error',abs(eb-source)),
      ('other_absolute_error',abs(eb-control)),
      ('gain_vs_direct',abs(eb-direct)-abs(eb-source)),
      ('gain_vs_other',abs(eb-control)-abs(eb-source))):
        if not close(m[name],val):
            raise ValueError('SMC21 heldout physical/gradient prediction arithmetic forged: '+name)
    for name,val in (
      ('direct_sign_correct',sign_eps(eb)==sign_eps(direct)),
      ('source_sign_correct',sign_eps(eb)==sign_eps(source)),
      ('other_sign_correct',sign_eps(eb)==sign_eps(control))):
        if m[name]!=val:
            raise ValueError('original signed target native effect changed: '+name)
    if any(m[name] and not m['clean_correct'] for name in
           ('reference_flip','other_flip','target_flip')):
        raise ValueError('clean-incorrect goal marked correct-to-wrong')


def independent_score(rows,betas,seed):
    clusters=[]
    for row in rows:
        n=0;ad=0.;ao=0.;d=0.;s=0.;o=0.;signs=[0,0,0]
        for flag,m in zip(row['eligible_mask'],row['metrics']):
            if flag!=(m is not None):
                raise ValueError('selected goal outcomes changed eligibility')
            if not flag:continue
            n+=1;eb=m['target_native_drop']
            pd=m['direct_prediction'];ps=m['source_calibrated_prediction'];po=m['other_calibrated_prediction']
            ed=abs(eb-pd);es=abs(eb-ps);eo=abs(eb-po)
            ad+=ed-es;ao+=eo-es;d+=ed;s+=es;o+=eo
            signs[0]+=int(sign_eps(eb)==sign_eps(pd))
            signs[1]+=int(sign_eps(eb)==sign_eps(ps))
            signs[2]+=int(sign_eps(eb)==sign_eps(po))
        clusters.append((n,ad,ao,d,s,o,*signs))
    total=sum(x[0] for x in clusters)
    puzzles=sum(x[0]>0 for x in clusters)
    if total==0:
        return {'status':'zero_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed);boot_d=[];boot_o=[]
    for _ in range(2000):
        ids=[rng.randrange(len(rows)) for row in rows]
        n=sum(clusters[i][0] for i in ids)
        if n:
            boot_d.append(sum(clusters[i][1] for i in ids)/n)
            boot_o.append(sum(clusters[i][2] for i in ids)/n)
    boot_d.sort();boot_o.sort()
    valid=len(boot_d)==2000 and len(boot_o)==2000
    ci_d=[boot_d[50],boot_d[1949]] if valid else None
    ci_o=[boot_o[50],boot_o[1949]] if valid else None
    direct=sum(x[3] for x in clusters)
    ratio=sum(x[4] for x in clusters)/direct if direct>ZERO else None
    sd=sum(x[6] for x in clusters);ss=sum(x[7] for x in clusters);so=sum(x[8] for x in clusters)
    cal=all(betas[k]['identified'] for k in ('beta_source','beta_other'))
    coverage=cal and total>=24 and puzzles>=20 and valid
    positive=valid and ci_d[0]>0 and ci_o[0]>0
    relative=ratio is not None and ratio<=.95
    signs=ss>=sd
    return {
        'status':'measured' if valid else 'bootstrap_abstention',
        'eligible_goals':total,'eligible_puzzles':puzzles,
        'mean_gain_vs_direct':sum(x[1] for x in clusters)/total,
        'mean_gain_vs_other':sum(x[2] for x in clusters)/total,
        'direct_absolute_error_mean':direct/total,
        'source_calibrated_absolute_error_mean':sum(x[4] for x in clusters)/total,
        'other_calibrated_absolute_error_mean':sum(x[5] for x in clusters)/total,
        'source_vs_direct_mae_ratio':ratio,
        'direct_sign_accuracy':sd/total,
        'source_sign_accuracy':ss/total,
        'other_sign_accuracy':so/total,
        'direct_sign_correct':sd,'source_sign_correct':ss,'other_sign_correct':so,
        'puzzle_cluster_bootstrap95_gain_vs_direct':ci_d,
        'puzzle_cluster_bootstrap95_gain_vs_other':ci_o,
        'valid_bootstrap_draws':len(boot_d),
        'requested_bootstrap_draws':2000,
        'calibration_fits_identified':cal,
        'registered_coverage_gate':coverage,
        'registered_both_positive_gain_gate':positive,
        'registered_relative_mae_gate':relative,
        'registered_sign_nondegradation_gate':signs,
        'registered_success':bool(coverage and positive and relative and signs)
    }


def audit_full(receipt,frozen_dir,calibration_data,cal_raw):
    if (receipt['schema']!='SMC-21/calibration-only-source-residual-vs-direct-gradient-v1'
        or receipt['pre_registration_commit']!=PRECOMMIT
        or receipt['smc19_frozen_json_sha256']!=FROZEN19_SHA
        or receipt['smc21_calibration_json_sha256']!=sha(cal_raw)
        or receipt['original_calibration_source']!=calibration_data):
        raise ValueError('SMC21 original registered calibration artifact differs')
    frozen=json.loads((Path(frozen_dir)/'frozen.json').read_bytes())
    audit_calibration(calibration_data,frozen,Path(frozen_dir))
    if (receipt['source_original_receipts']!=frozen['source_original_receipts'] or
        receipt['old_seven_conditioned_and_four_SMC19_mappings_verified'] is not True or
        receipt['original_refit_checkpoint_files_SHA_verified'] is not True):
        raise ValueError('SMC21 original trained source/mapping SHA integrity failed')
    if receipt['fixed_parameters']!={'ridge_factor':1.05,'ridge_clip':[-2.,2.],
           'bootstrap_draws':2000,'relative_MAE_gate':.95,'sign_epsilon':1e-8,
           'ridge_degenerate_sum_squares_cutoff':1e-12}:
        raise ValueError('SMC21 preregistered threshold/fit scorer changed')
    full=receipt['puzzles_per_family']==64 and len(receipt['pairs'])==4
    if receipt['full_protocol']!=full:
        raise ValueError('SMC21 full/pilot label differs')
    audit_rosters(receipt)
    outcomes=[]
    for ix,pair in enumerate(receipt['pairs']):
        if ix>=4 or (pair['source'],pair['target'])!=PAIRS[ix]:
            raise ValueError('SMC21 frozen same-teacher refit comparison roster altered')
        prior=frozen['pair_calibration'][ix]
        beta=calibration_data['pairs'][ix]
        if (pair['frozen_original_SMC19_mapping_hashes']!=prior['mapping_hashes']
            or pair['original_calibration_betas']!={'beta_source':beta['beta_source'],
                                                    'beta_other':beta['beta_other']}):
            raise ValueError('original matched SAE feature map/calibration beta tampered')
        for fam in FAMILIES:
            group=pair['heldout'][fam]
            rows=group['per_puzzle']
            roster=receipt['new_rosters'][fam]
            if len(rows)!=len(roster):raise ValueError('new puzzle scorer row count changed')
            cover=group['coverage']
            cols_a=cover['reference_decoder_columns_96x48']
            cols_b=cover['refit_decoder_columns_96x48']
            reasons={};count=0
            for i,(row,ro) in enumerate(zip(rows,roster)):
                missing=[j for j,v in enumerate(ro['x']) if v==0]
                pos=[missing[j*len(missing)//8] for j in range(8)]
                if (row['puzzle_index']!=i or row['seed']!=ro['seed'] or
                    row['goal_positions']!=pos or
                    row['goal_truth_digits']!=[ro['target'][j] for j in pos]):
                    raise ValueError('new Sudoku scorer input goal selection changed')
                for key in ('source_feature_ids','target_feature_ids','other_feature_ids',
                            'reference_top1_coefficients','reference_top2_coefficients',
                            'target_matched_coefficients','other_coefficients',
                            'active_refit_feature_ids','eligible_mask','preoutcome_reasons','metrics'):
                    if len(row[key])!=8:raise ValueError('missing blank-goal receipt field')
                for slot in range(8):
                    a=row['source_feature_ids'][slot];b=row['target_feature_ids'][slot]
                    if not 0<=a<96 or prior['mapping']['matched'][a]!=b:
                        raise ValueError('refit source calibrated Hungarian assignment differs')
                    active=row['active_refit_feature_ids'][slot]
                    strict=(row['reference_top1_coefficients'][slot]>0 and
                            row['reference_top1_coefficients'][slot]>row['reference_top2_coefficients'][slot])
                    coeff=row['target_matched_coefficients'][slot]
                    expected=('reference_not_strict_top1' if not strict else
                              'refit_matched_inactive' if coeff<=0 else
                              'refit_missing_other_active' if len(active)<2 else
                              'eligible')
                    if (row['preoutcome_reasons'][slot]!=expected or
                        bool(row['eligible_mask'][slot])!=(expected=='eligible')):
                        raise ValueError('SMC21 source-independent eligibility reason changed')
                    reasons[expected]=reasons.get(expected,0)+1
                    metric=row['metrics'][slot]
                    if expected=='eligible':
                        count+=1
                        if metric is None:raise ValueError('eligible calibration prediction missing')
                        audit_fresh_goal(metric,row,slot,ix,fam,cols_a,cols_b,beta)
                    elif metric is not None:
                        raise ValueError('ineligible target received finite erasure inference')
            if (count!=cover['eligible_goals'] or
                reasons!=cover['preoutcome_reason_counts'] or
                cover['selected_goals']!=len(rows)*8):
                raise ValueError('heldout condition denominators differ')
            computed=independent_score(rows,beta,202621130+100*ix+len(fam))
            reported=group['statistics']
            if set(computed)!=set(reported):
                raise ValueError('SMC21 registered score schema keys changed')
            for field,val in computed.items():
                actual=reported[field]
                if isinstance(val,list):
                    if not isinstance(actual,list) or len(actual)!=len(val) or any(
                        not close(x,y) for x,y in zip(val,actual)):
                        raise ValueError('source residual test clustered CI altered')
                elif isinstance(val,float) or val is None:
                    if not close(val,actual):
                        raise ValueError('SMC21 scientific paired MAE metric differs: '+field)
                elif val!=actual:
                    raise ValueError('registered source residual gate/coverage changed: '+field)
            outcomes.append({'target':pair['target'],'family':fam,
                             'eligible_goals':count,
                             'mean_gain_vs_direct':computed.get('mean_gain_vs_direct'),
                             'mean_gain_vs_other':computed.get('mean_gain_vs_other'),
                             'relative_mae':computed.get('source_vs_direct_mae_ratio'),
                             'registered_success':computed['registered_success']})
        both=bool(full and all(pair['heldout'][fam]['statistics']['registered_success']
                               for fam in FAMILIES))
        if both!=pair['registered_success_both_new_families']:
            raise ValueError('SMC21 both-cohort preregistered success misreported')
    return {'schema':'SMC-21/independent-stdlib-full-residual-transfer-audit-v1',
            'original_SMC19_four_SAE_checkpoint_bytes_and_seven_original_mappings_verified':True,
            'calibration_only_ridge_coefficients_recomputed':True,
            'all_new_unique_noncanonical_sudoku_rosters_checked':True,
            'new_native_physical_edits_and_48D_clean_gradients_recomputed':True,
            'all_16000_puzzle_cluster_bootstraps_and_gates_replayed':True,
            'full_protocol':full,'n_pair_family_puzzle_cases':sum(
                len(pair['heldout'][fam]['per_puzzle']) for pair in receipt['pairs'] for fam in FAMILIES),
            'outcomes':outcomes,
            'scope':'Replays saved original PyTorch model outputs and gradients; independent numerical and source SHA auditor, not another neural inference run.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('calibration','full'),required=True)
    p.add_argument('--frozen-dir',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    data=a.calibration.read_bytes()
    calibration=json.loads(data)
    frozen=json.loads((a.frozen_dir/'frozen.json').read_bytes())
    if a.phase=='calibration':
        result=audit_calibration(calibration,frozen,a.frozen_dir)
    else:
        if a.receipt is None:
            raise ValueError('full independent replay requires saved test receipt')
        raw=a.receipt.read_bytes()
        result=audit_full(json.loads(raw),a.frozen_dir,calibration,data)
        result['raw_test_json_sha256']=sha(raw)
    result['original_calibration_json_sha256']=sha(data)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
