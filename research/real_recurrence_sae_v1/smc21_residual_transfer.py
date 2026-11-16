"""SMC-21: calibration-frozen SAE residual transport against direct task gradient.

Two separated phases:
  calibrate: re-verify original teacher/SAE/S19 fit hashes and SMC13C maps;
             evaluate clean task gradients ONLY on original feature_calibration
             puzzles, combine with already source-locked SMC19 erasure receipts,
             and freeze two 1D ridge coefficients per original SAE refit pair.
  evaluate: reload the archived calibration and unchanged refit weights BEFORE
            constructing fresh heldout 64+64 Sudoku; fix physical editors,
            calculate source/other erasure and clean task-gradient predictions,
            then score the withheld new refit erasure in the same frozen teacher.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from smc13_family import (new_full_solution,dig_unique,valid_solution,
                          count_solutions,rectangle_count,fingerprint)
from smc15_active_necessity import (encoded_goals,physical_columns,
        feature_deltas,other_active_ids,dose_matched_other,ablate_margin)
from smc18_task_gradient import task_gradients
from smc17_causal_inactive import source_models
from smc19_sae_refit import load_frozen as load_smc19_frozen
from smc20_anchored_refit import fresh_rosters as smc20_rosters

PRECOMMIT='0f0af2dc4b34c6a85dc4e0ba5d5d1e08597bbf59'
FROZEN19_SHA='612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b'
FAMILIES={'smc21_novel_30':(202621030000,30),
          'smc21_novel_34':(202621040000,34)}
PAIRS=(('newB_501_SAE_71','newB_501_SAE_131'),
       ('newB_501_SAE_71','newB_501_SAE_167'),
       ('newB_701_SAE_71','newB_701_SAE_131'),
       ('newB_701_SAE_71','newB_701_SAE_167'))
GOALS=8
BOOTSTRAPS=2000
ZERO=1e-8
MAX_BETA=2.
REGULARIZER_FACTOR=1.05
MIN_VAR_SUM=1e-12


def sha(x):
    return hashlib.sha256(x).hexdigest()


def frozen_sources(original,independent,conditioned,frozen_dir,
                   allow_fetch=False,source_checkpoint=None):
    torch.set_num_threads(2)
    teachers,comps,old,original_mappings,locks=source_models(
        original,independent,conditioned,allow_fetch,source_checkpoint)
    frozen,digest=load_smc19_frozen(frozen_dir,comps,locks,old)
    if digest!=FROZEN19_SHA or frozen['source_original_seven_mappings']!=original_mappings:
        raise ValueError('SMC21 original four SAE refits/seven source assignments changed')
    return teachers,comps,old,locks,frozen,digest


def fit_ridge(calibration_rows,key):
    rows=[m for puzzle in calibration_rows for m in puzzle['metrics'] if m is not None]
    xs=[r[key] for r in rows]
    ys=[r['target_finite_residual'] for r in rows]
    x2=sum(x*x for x in xs)
    xy=sum(x*y for x,y in zip(xs,ys))
    n=sum(sum(m is not None for m in row['metrics'])>0 for row in calibration_rows)
    degenerate=x2<=MIN_VAR_SUM
    beta=(max(-MAX_BETA,min(MAX_BETA,xy/(REGULARIZER_FACTOR*x2)))
          if not degenerate else 0.)
    return {'coefficient':float(beta),'calibration_feature':key,
            'cross_product_sum':float(xy),'source_square_sum':float(x2),
            'ridge_factor':REGULARIZER_FACTOR,'clipping_interval':[-2.,2.],
            'degenerate':degenerate,'eligible_goals':len(xs),
            'eligible_puzzles':n,
            'identified':len(xs)>=24 and n>=20 and not degenerate and math.isfinite(beta)}


def prepare_geometry(ctx,comps,ref,refit,mapping,roster,family,index):
    """Original SMC19 source-top1/refit-native-active mask, before outputs."""
    co=ctx[comps[ref][0]]['cohorts'][family]
    co_refit=ctx[comps[refit][0]]['cohorts'][family]
    if comps[ref][0]!=comps[refit][0] or not bool((co['inputs']==co_refit['inputs']).all()):
        raise ValueError('frozen teacher, family or row indexing mismatch')
    stage=co['stage1'];positions=co['positions']
    za,top,strict=encoded_goals(stage,positions,*comps[ref][1:])
    zb,_,_=encoded_goals(stage,positions,*comps[refit][1:])
    matched=torch.tensor(mapping,dtype=torch.long)[top]
    alt,available=other_active_ids(zb,matched,index,family,
                                   [r['seed'] for r in roster])
    eligible=strict&available
    da_cols=physical_columns(comps[ref][1],comps[ref][3])
    db_cols=physical_columns(comps[refit][1],comps[refit][3])
    da,z_a=feature_deltas(za,top,da_cols)
    db,z_b=feature_deltas(zb,matched,db_cols)
    do=dose_matched_other(db,alt,db_cols)
    if not bool(torch.allclose(db.norm(dim=-1),do.norm(dim=-1),
                               atol=1e-12,rtol=1e-12)):
        raise ArithmeticError('SMC19 equal-dose other-active refit control lost')
    return {'cohort':co,'positions':positions,'za':za,'zb':zb,
            'top':top,'matched':matched,'other':alt,
            'eligible':eligible,'source_selected_z':z_a,
            'refit_selected_z':z_b,'da':da,'db':db,'do':do,
            'source_decoder_cols':da_cols,'refit_decoder_cols':db_cols}


def calibrate(original,independent,conditioned,frozen_dir,out,
              allow_fetch=False,source_checkpoint=None):
    teachers,comps,old,locks,frozen,fh=frozen_sources(
        original,independent,conditioned,frozen_dir,
        allow_fetch,source_checkpoint)
    from smc14_causal_transfer import source_group_context
    # Strict source calibration; no SMC20 heldout and no SMC21 novel test.
    ctx,_=source_group_context(teachers,comps,old,('feature_calibration',))
    results=[]
    for index,(ref,refit) in enumerate(PAIRS):
        historical=frozen['pair_calibration'][index]
        if (historical['source'],historical['target'])!=(ref,refit):
            raise ValueError('original SMC19 calibrated comparison reordered')
        orig=historical['calibration']['raw']
        if len(orig)!=64:
            raise ValueError('expected 64 original calibration puzzles')
        g=prepare_geometry(ctx,comps,ref,refit,historical['mapping']['matched'],
                           old['feature_calibration'],'feature_calibration',index)
        co=g['cohort']
        teacher=ctx[comps[ref][0]]['model64']
        for p in teacher.parameters():p.requires_grad_(False)
        rows=[]
        for ix,row in enumerate(orig):
            truth=co['truth_at_goals'][ix]
            positions=g['positions'][ix]
            if (row['puzzle_index']!=ix or
                row['seed']!=old['feature_calibration'][ix]['seed'] or
                row['goal_positions']!=positions.tolist() or
                row['eligible_mask']!=g['eligible'][ix].tolist() or
                row['source_selected_ids']!=g['top'][ix].tolist() or
                row['target_matched_ids']!=g['matched'][ix].tolist() or
                row['target_other_active_ids']!=g['other'][ix].tolist()):
                raise ValueError('SMC19 original preoutcome calibration selection changed')
            grad=task_gradients(teacher,co['stage1'][ix],co['clean_logits'][ix],
                                truth,positions,g['eligible'][ix])
            points=[]
            for j in range(GOALS):
                if not bool(g['eligible'][ix,j]):
                    points.append(None)
                    continue
                gr=grad[j];a=g['da'][ix,j];b=g['db'][ix,j];o=g['do'][ix,j]
                ea=float(row['source_drop'][j])
                eb=float(row['matched_drop'][j])
                eo=float(row['other_active_drop'][j])
                ra=ea-float(gr@a)
                rb=eb-float(gr@b)
                ro=eo-float(gr@o)
                if not all(math.isfinite(x) for x in (ea,eb,eo,ra,rb,ro)):
                    raise ArithmeticError('nonfinite original calibration native margin')
                points.append({
                  'source_feature':int(g['top'][ix,j]),
                  'target_feature':int(g['matched'][ix,j]),
                  'other_feature':int(g['other'][ix,j]),
                  'physical_gradient_48':gr.tolist(),
                  'source_delta_48':a.tolist(),
                  'target_delta_48':b.tolist(),
                  'other_delta_48':o.tolist(),
                  'source_native_drop':ea,
                  'target_native_drop':eb,
                  'other_native_drop':eo,
                  'source_finite_residual':ra,
                  'target_finite_residual':rb,
                  'other_finite_residual':ro})
            rows.append({
                'puzzle_index':ix,'seed':row['seed'],
                'goal_positions':positions.tolist(),
                'eligible_mask':g['eligible'][ix].tolist(),
                'metrics':points})
        b_source=fit_ridge(rows,'source_finite_residual')
        b_other=fit_ridge(rows,'other_finite_residual')
        results.append({'source':ref,'target':refit,
            'frozen_original_mapping_hashes':historical['mapping_hashes'],
            'source_calibration_replay_verified':True,
            'beta_source':b_source,'beta_other':b_other,
            'original_64_puzzle_calibration':rows})
        print('SMC21_ORIGINAL_CALIBRATION_RIDGE_FROZEN',ref,refit,
              {'beta_source':b_source,'beta_other':b_other},flush=True)
    output={
        'schema':'SMC-21/original-calibration-ridge-freeze-v1',
        'pre_registration_commit':PRECOMMIT,
        'source_original_receipts':locks,
        'smc19_frozen_json_sha256':fh,
        'original_seven_mappings_hash_verified':True,
        'original_four_SMC19_calibration_assignments_hash_verified':True,
        'original_calibration_roster_sha256':fingerprint(old['feature_calibration']),
        'original_training_roster_sha256':fingerprint(old['teacher_train']),
        'original_validation_roster_sha256':fingerprint(old['teacher_val']),
        'calibration_puzzles':64,'source_dataset':'SMC13 feature_calibration',
        'fixed_ridge_factor':REGULARIZER_FACTOR,
        'fixed_coefficient_clip':[-2.,2.],
        'pairs':results,
        'no_fresh_heldout_states_generated':True,
        'limit':'Calibration target causal erasures are from the original frozen SMC19 receipt; clean FP64 gradient was computed only on original feature_calibration.'}
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print('SMC21_CALIBRATION_FROZEN_FILE_SHA256',sha(out.read_bytes()),flush=True)
    return output


def fresh_rosters(old,limit=64):
    if limit not in (4,64):
        raise ValueError('registered SMC21 pilot four or full 64 only')
    prior,_=smc20_rosters(old,limit=64)
    xs={tuple(r['x']) for rows in list(old.values())+list(prior.values()) for r in rows}
    ys={tuple(r['target']) for rows in list(old.values())+list(prior.values()) for r in rows}
    output={}
    for fam,(base,holes) in FAMILIES.items():
        rows=[]
        for i in range(limit):
            seed=base+i
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('SMC21 noncanonical source Sudoku invalid')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1 or any(v and v!=y[j] for j,v in enumerate(x)):
                raise ValueError('SMC21 exact-unique Sudoku missing source correctness')
            if tuple(x) in xs or tuple(y) in ys:
                raise ValueError('SMC21 test solution/input overlaps SMC13-SMC20')
            xs.add(tuple(x));ys.add(tuple(y))
            rows.append({'seed':seed,'x':x,'target':y,'holes':holes,
                         'rectangles':rectangle_count(y)})
        output[fam]=rows
    return output,{f:fingerprint(rows) for f,rows in output.items()}


def sign_eps(x):
    return 0 if abs(x)<=ZERO else (1 if x>0 else -1)


def score_family(rows,betas,seed,draws=BOOTSTRAPS):
    if not rows:
        raise ValueError('one or more registered Sudoku puzzle clusters required')
    clusters=[]
    for puzzle in rows:
        if len(puzzle['eligible_mask'])!=GOALS or len(puzzle['metrics'])!=GOALS:
            raise ValueError('original eight goal slots incomplete')
        n=0;ad=0.;ao=0.;d=0.;s=0.;o=0.;scores=[0,0,0]
        for flag,x in zip(puzzle['eligible_mask'],puzzle['metrics']):
            if bool(flag)!=(x is not None):
                raise ValueError('goal eligibility and outcome mismatch')
            if not flag:continue
            n+=1
            eb=x['target_native_drop']
            pd=x['direct_prediction'];ps=x['source_calibrated_prediction']
            po=x['other_calibrated_prediction']
            ed=abs(eb-pd);es=abs(eb-ps);eo=abs(eb-po)
            ad+=ed-es;ao+=eo-es;d+=ed;s+=es;o+=eo
            scores[0]+=int(sign_eps(eb)==sign_eps(pd))
            scores[1]+=int(sign_eps(eb)==sign_eps(ps))
            scores[2]+=int(sign_eps(eb)==sign_eps(po))
        clusters.append((n,ad,ao,d,s,o,*scores))
    total=sum(x[0] for x in clusters)
    npuzzles=sum(x[0]>0 for x in clusters)
    if total==0:
        return {'status':'zero_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed);boost_direct=[];boost_other=[]
    for _ in range(draws):
        ids=[rng.randrange(len(rows)) for _ in rows]
        n=sum(clusters[i][0] for i in ids)
        if n:
            boost_direct.append(sum(clusters[i][1] for i in ids)/n)
            boost_other.append(sum(clusters[i][2] for i in ids)/n)
    boost_direct.sort();boost_other.sort()
    valid=len(boost_direct)==draws and len(boost_other)==draws
    ci_d=[boost_direct[50],boost_direct[1949]] if valid and draws==2000 else None
    ci_o=[boost_other[50],boost_other[1949]] if valid and draws==2000 else None
    direct=sum(x[3] for x in clusters)
    ratio=sum(x[4] for x in clusters)/direct if direct>ZERO else None
    sign_d=sum(x[6] for x in clusters)
    sign_s=sum(x[7] for x in clusters)
    sign_o=sum(x[8] for x in clusters)
    cal=all(betas[k]['identified'] for k in ('beta_source','beta_other'))
    coverage=cal and total>=24 and npuzzles>=20 and valid
    positive=valid and ci_d[0]>0 and ci_o[0]>0
    relative=ratio is not None and ratio<=.95
    sign=sign_s>=sign_d
    return {
        'status':'measured' if valid else 'bootstrap_abstention',
        'eligible_goals':total,'eligible_puzzles':npuzzles,
        'mean_gain_vs_direct':sum(x[1] for x in clusters)/total,
        'mean_gain_vs_other':sum(x[2] for x in clusters)/total,
        'direct_absolute_error_mean':direct/total,
        'source_calibrated_absolute_error_mean':sum(x[4] for x in clusters)/total,
        'other_calibrated_absolute_error_mean':sum(x[5] for x in clusters)/total,
        'source_vs_direct_mae_ratio':ratio,
        'direct_sign_accuracy':sign_d/total,
        'source_sign_accuracy':sign_s/total,
        'other_sign_accuracy':sign_o/total,
        'direct_sign_correct':sign_d,
        'source_sign_correct':sign_s,
        'other_sign_correct':sign_o,
        'puzzle_cluster_bootstrap95_gain_vs_direct':ci_d,
        'puzzle_cluster_bootstrap95_gain_vs_other':ci_o,
        'valid_bootstrap_draws':len(boost_direct),
        'requested_bootstrap_draws':draws,
        'calibration_fits_identified':cal,
        'registered_coverage_gate':coverage,
        'registered_both_positive_gain_gate':positive,
        'registered_relative_mae_gate':relative,
        'registered_sign_nondegradation_gate':sign,
        'registered_success':bool(coverage and positive and relative and sign)
    }


def evaluate_family(ctx,comps,reference,refit,mapping,rosters,family,index,betas):
    g=prepare_geometry(ctx,comps,reference,refit,mapping,rosters,family,index)
    co=g['cohort'];stage=co['stage1'];positions=g['positions']
    truth=co['truth_at_goals'];clean=co['clean_logits']
    eligible=g['eligible'];n=len(rosters)
    model=ctx[comps[reference][0]]['model64']
    for param in model.parameters():param.requires_grad_(False)
    print('SMC21_PREOUTCOME_GEOMETRY_FROZEN',family,index,int(eligible.sum()),flush=True)
    # These are the only extra measured native erasures, one for each residual predictor.
    ea,correct,flip_a=ablate_margin(model,stage,clean,truth,positions,g['da'])
    eo,correct_o,flip_o=ablate_margin(model,stage,clean,truth,positions,g['do'])
    if not bool((correct==correct_o).all()):
        raise ValueError('same teacher clean task score must remain unchanged')
    preds=[]
    for p in range(n):
        grad=task_gradients(model,stage[p],clean[p],truth[p],positions[p],eligible[p])
        vals=[]
        for j in range(GOALS):
            if not bool(eligible[p,j]):
                vals.append(None);continue
            a=g['da'][p,j];b=g['db'][p,j];o=g['do'][p,j];gg=grad[j]
            direct=float(gg@b)
            ra=float(ea[p,j]-gg@a)
            ro=float(eo[p,j]-gg@o)
            src=direct+betas['beta_source']['coefficient']*ra
            other=direct+betas['beta_other']['coefficient']*ro
            vals.append({'gradient_48':gg.tolist(),
                'reference_delta_48':a.tolist(),'target_delta_48':b.tolist(),
                'other_delta_48':o.tolist(),
                'reference_native_drop':float(ea[p,j]),
                'other_native_drop':float(eo[p,j]),
                'source_finite_residual':ra,'other_finite_residual':ro,
                'direct_prediction':direct,
                'source_calibrated_prediction':src,
                'other_calibrated_prediction':other,
                'unit_anchor_prediction':direct+ra})
        preds.append(vals)
    print('SMC21_ALL_CALIBRATION_ONLY_PREDICTIONS_FROZEN',
          family,index,int(eligible.sum()),flush=True)
    # Never use target native erasure to choose calibration coefficient, geometry,
    # prediction, eligibility, answer/rival, or baseline.
    eb,correct_b,flip_b=ablate_margin(model,stage,clean,truth,positions,g['db'])
    if not bool((correct==correct_b).all()):
        raise ValueError('different clean task targets after native target intervention')
    raw=[]
    for p,row in enumerate(rosters):
        metrics=[]
        reasons=[]
        for j in range(GOALS):
            z_a=float(g['source_selected_z'][p,j])
            z_b=float(g['refit_selected_z'][p,j])
            other_z=float(g['zb'][p,j,g['other'][p,j]])
            strict=(z_a>0 and float(g['za'][p,j].topk(2).values[1])<z_a)
            active=[int(v) for v in torch.nonzero(g['zb'][p,j]>0).flatten().tolist()]
            why=('reference_not_strict_top1' if not strict else
                 'refit_matched_inactive' if z_b<=0 else
                 'refit_missing_other_active' if len(active)<2 else
                 'eligible')
            if (why=='eligible')!=bool(eligible[p,j]):
                raise AssertionError('preoutcome source/refit eligibility drift')
            reasons.append(why)
            if not bool(eligible[p,j]):
                metrics.append(None);continue
            value=dict(preds[p][j])
            actual=float(eb[p,j])
            pd=value['direct_prediction'];ps=value['source_calibrated_prediction'];po=value['other_calibrated_prediction']
            value.update({
                'source_feature':int(g['top'][p,j]),
                'target_feature':int(g['matched'][p,j]),
                'other_feature':int(g['other'][p,j]),
                'reference_coefficient':z_a,
                'target_coefficient':z_b,
                'other_coefficient':other_z,
                'target_native_drop':actual,
                'target_finite_residual':actual-pd,
                'reference_physical_dose':float(g['da'][p,j].norm()),
                'target_physical_dose':float(g['db'][p,j].norm()),
                'other_physical_dose':float(g['do'][p,j].norm()),
                'direct_absolute_error':abs(actual-pd),
                'source_absolute_error':abs(actual-ps),
                'other_absolute_error':abs(actual-po),
                'gain_vs_direct':abs(actual-pd)-abs(actual-ps),
                'gain_vs_other':abs(actual-po)-abs(actual-ps),
                'direct_sign_correct':sign_eps(actual)==sign_eps(pd),
                'source_sign_correct':sign_eps(actual)==sign_eps(ps),
                'other_sign_correct':sign_eps(actual)==sign_eps(po),
                'clean_correct':bool(correct[p,j]),
                'reference_flip':bool(flip_a[p,j]),
                'other_flip':bool(flip_o[p,j]),
                'target_flip':bool(flip_b[p,j])})
            metrics.append(value)
        raw.append({
            'puzzle_index':p,'seed':row['seed'],
            'goal_positions':positions[p].tolist(),
            'goal_truth_digits':truth[p].tolist(),
            'source_feature_ids':g['top'][p].tolist(),
            'target_feature_ids':g['matched'][p].tolist(),
            'other_feature_ids':g['other'][p].tolist(),
            'reference_top1_coefficients':g['source_selected_z'][p].tolist(),
            'reference_top2_coefficients':g['za'][p].topk(2,dim=-1).values[:,1].tolist(),
            'target_matched_coefficients':g['refit_selected_z'][p].tolist(),
            'other_coefficients':g['zb'][p].gather(-1,g['other'][p,:,None]).squeeze(-1).tolist(),
            'active_refit_feature_ids':[
                torch.nonzero(g['zb'][p,j]>0).flatten().tolist() for j in range(GOALS)],
            'eligible_mask':eligible[p].tolist(),'preoutcome_reasons':reasons,
            'metrics':metrics})
    counts={}
    for page in raw:
        for key in page['preoutcome_reasons']:
            counts[key]=counts.get(key,0)+1
    return raw,{
        'eligible_goals':int(eligible.sum()),'selected_goals':n*GOALS,
        'preoutcome_reason_counts':counts,
        'reference_decoder_columns_96x48':g['source_decoder_cols'].tolist(),
        'refit_decoder_columns_96x48':g['refit_decoder_cols'].tolist(),
        'target_clean_goal_accuracy':float(correct.float().mean())}


def evaluate(original,independent,conditioned,frozen_dir,calibration_path,
             out,pilot=0,pair_limit=4,allow_fetch=False,source_checkpoint=None):
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError('registered pilot/four-pair test sizes only')
    teachers,comps,old,locks,frozen,fh=frozen_sources(
        original,independent,conditioned,frozen_dir,allow_fetch,source_checkpoint)
    raw_calibration=Path(calibration_path).read_bytes()
    cal=json.loads(raw_calibration)
    if (cal['schema']!='SMC-21/original-calibration-ridge-freeze-v1' or
        cal['pre_registration_commit']!=PRECOMMIT or
        cal['smc19_frozen_json_sha256']!=fh or
        cal['source_original_receipts']!=locks or
        cal['calibration_puzzles']!=64 or
        not cal['no_fresh_heldout_states_generated'] or
        len(cal['pairs'])!=4):
        raise ValueError('SMC21 prior independent calibration artifact not frozen')
    if (cal['original_calibration_roster_sha256']!=fingerprint(old['feature_calibration'])
        or cal['original_training_roster_sha256']!=fingerprint(old['teacher_train'])
        or cal['original_validation_roster_sha256']!=fingerprint(old['teacher_val'])):
        raise ValueError('source-only fit/validation/calibration roster changed')
    for index,entry in enumerate(cal['pairs']):
        prior=frozen['pair_calibration'][index]
        if ((entry['source'],entry['target'])!=PAIRS[index] or
            entry['frozen_original_mapping_hashes']!=prior['mapping_hashes']):
            raise ValueError('original frozen SMC19 matched assignment changed')
    print('SMC21_SOURCE_AND_PRIOR_RESIDUAL_CALIBRATION_LOCKED',
          sha(raw_calibration),flush=True)
    # Only now construct the previously unseen Sudoku states.
    rosters,manifest=fresh_rosters(old,limit=pilot or 64)
    from smc14_causal_transfer import source_group_context
    ctx,_=source_group_context(teachers,comps,rosters,tuple(FAMILIES))
    pairs=[]
    for index,(ref,refit) in enumerate(PAIRS[:pair_limit]):
        old_pair=frozen['pair_calibration'][index]
        betas=cal['pairs'][index]
        heldout={}
        for fam in FAMILIES:
            rows,coverage=evaluate_family(ctx,comps,ref,refit,
                        old_pair['mapping']['matched'],rosters[fam],fam,index,betas)
            stats=score_family(rows,betas,202621130+100*index+len(fam))
            heldout[fam]={'per_puzzle':rows,'coverage':coverage,'statistics':stats}
        both=(not bool(pilot or pair_limit!=4) and all(
            heldout[f]['statistics']['registered_success'] for f in FAMILIES))
        pairs.append({
            'source':ref,'target':refit,
            'frozen_original_SMC19_mapping_hashes':old_pair['mapping_hashes'],
            'original_calibration_betas':{
                'beta_source':betas['beta_source'],'beta_other':betas['beta_other']},
            'heldout':heldout,'registered_success_both_new_families':both})
        print('SMC21_RESIDUAL_TRANSFER_OUTCOME',ref,refit,{
          f:{'eligible':heldout[f]['statistics']['eligible_goals'],
             'gain_direct':heldout[f]['statistics'].get('mean_gain_vs_direct'),
             'gain_other':heldout[f]['statistics'].get('mean_gain_vs_other'),
             'mae_ratio':heldout[f]['statistics'].get('source_vs_direct_mae_ratio'),
             'gate':heldout[f]['statistics']['registered_success']}
          for f in FAMILIES},flush=True)
    result={
        'schema':'SMC-21/calibration-only-source-residual-vs-direct-gradient-v1',
        'pre_registration_commit':PRECOMMIT,
        'source_original_receipts':locks,
        'smc19_frozen_json_sha256':fh,
        'smc21_calibration_json_sha256':sha(raw_calibration),
        'original_calibration_source':cal,
        'old_seven_conditioned_and_four_SMC19_mappings_verified':True,
        'original_refit_checkpoint_files_SHA_verified':True,
        'new_rosters':rosters,'new_manifest_sha256':manifest,
        'new_seed_bases':{f:v[0] for f,v in FAMILIES.items()},
        'puzzles_per_family':pilot or 64,
        'full_protocol':not bool(pilot or pair_limit!=4),
        'fixed_parameters':{'ridge_factor':REGULARIZER_FACTOR,
            'ridge_clip':[-2.,2.],'bootstrap_draws':BOOTSTRAPS,
            'relative_MAE_gate':.95,'sign_epsilon':ZERO,
            'ridge_degenerate_sum_squares_cutoff':MIN_VAR_SUM},
        'pairs':pairs,
        'scope':'All coefficients from original 64-puzzle calibration; true-target clean task gradient uses the answer only as a mechanistic diagnostic; source neural weights never trained.'}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('calibrate','evaluate'),required=True)
    p.add_argument('--original-artifact',type=Path,required=True)
    p.add_argument('--independent-artifact',type=Path,required=True)
    p.add_argument('--conditioned-receipt',type=Path,required=True)
    p.add_argument('--frozen-dir',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path('runs/smc21/calibration/residual_calibration.json'))
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,))
    p.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--out',type=Path,default=Path('runs/smc21/full.json'))
    a=p.parse_args()
    if a.phase=='calibrate':
        calibrate(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
                  a.frozen_dir,a.calibration,a.fetch_source,a.source_checkpoint)
    else:
        receipt=evaluate(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
                         a.frozen_dir,a.calibration,a.out,a.pilot_puzzles or 0,
                         a.pair_limit,a.fetch_source,a.source_checkpoint)
        print(json.dumps({'out':str(a.out),'full':receipt['full_protocol'],
              'pair_gates':[{'target':p['target'],
                             'both_families':p['registered_success_both_new_families']}
                             for p in receipt['pairs']]},indent=2))


if __name__=='__main__':main()
