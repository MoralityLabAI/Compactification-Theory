"""SMC-22: source-only weighted-median residual calibration, fresh SAE refit test.

No model retraining, no novel calibration data. Reuses all SMC19/SMC21 frozen
model, SAE and original 64-puzzle calibration receipts. Two clean native
residual corrections get exactly one scalar and one observed native erasure.
Mandatory controls are strong clean-gradient direct, original SMC21 ridge
source correction and equally calibrated other-active LAD correction.
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
from smc21_residual_transfer import (
    frozen_sources,prepare_geometry,sign_eps,GOALS,
    fresh_rosters as original_smc21_rosters,ablate_margin,task_gradients
)
from smc22_lad_calibration import PRECOMMIT,SMC21_CALIB_SHA,SMC19_SHA,sha

FAMILIES={'smc22_novel_30':(202622030000,30),
          'smc22_novel_34':(202622040000,34)}
PAIRS=(('newB_501_SAE_71','newB_501_SAE_131'),
       ('newB_501_SAE_71','newB_501_SAE_167'),
       ('newB_701_SAE_71','newB_701_SAE_131'),
       ('newB_701_SAE_71','newB_701_SAE_167'))
BOOTSTRAPS=2000
ZERO=1e-8


def fresh_rosters(old,limit=64):
    if limit not in (4,64):
        raise ValueError('SMC22 accepts only four-puzzle pilot and 64-puzzle full')
    prior,_=original_smc21_rosters(old,64)
    xs={tuple(r['x']) for group in list(old.values())+list(prior.values()) for r in group}
    ys={tuple(r['target']) for group in list(old.values())+list(prior.values()) for r in group}
    rosters={}
    for fam,(base,holes) in FAMILIES.items():
        rows=[]
        for ix in range(limit):
            seed=base+ix
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('SMC22 target not valid novel noncanonical full grid')
            x=dig_unique(y,seed+1000000,holes)
            if (x.count(0)!=holes or count_solutions(x,2)!=1 or
                any(d and d!=y[j] for j,d in enumerate(x))):
                raise ValueError('SMC22 target/input not exact unique with registered holes')
            if tuple(x) in xs or tuple(y) in ys:
                raise ValueError('SMC22 fresh source overlaps original SMC13-21 cohorts')
            xs.add(tuple(x));ys.add(tuple(y))
            rows.append({'seed':seed,'x':x,'target':y,
                         'holes':holes,'rectangles':rectangle_count(y)})
        rosters[fam]=rows
    return rosters,{name:fingerprint(rows) for name,rows in rosters.items()}


def score_family(rows,fits,seed,draws=BOOTSTRAPS):
    if not rows:raise ValueError('source-locked puzzle clusters required')
    groups=[]
    for row in rows:
        if len(row['eligible_mask'])!=8 or len(row['metrics'])!=8:
            raise ValueError('eight goal tokens required')
        n=0;gain=[0.,0.,0.];loss=[0.,0.,0.,0.];signed=[0,0,0,0]
        for ok,m in zip(row['eligible_mask'],row['metrics']):
            if bool(ok)!=(m is not None):
                raise ValueError('SMC22 native erasure outcomes only on eligible tokens')
            if not ok:continue
            n+=1
            y=m['target_native_drop']
            predictors=(m['direct_prediction'],m['ridge_source_prediction'],
                        m['source_calibrated_prediction'],m['other_calibrated_prediction'])
            err=[abs(y-p) for p in predictors]
            for j in range(4):
                loss[j]+=err[j]
                signed[j]+=int(sign_eps(y)==sign_eps(predictors[j]))
            gain[0]+=err[0]-err[2]
            gain[1]+=err[1]-err[2]
            gain[2]+=err[3]-err[2]
        groups.append((n,*gain,*loss,*signed))
    total=sum(g[0] for g in groups)
    puzzles=sum(g[0]>0 for g in groups)
    if not total:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed);boots=[[],[],[]]
    for _ in range(draws):
        selection=[rng.randrange(len(groups)) for g in groups]
        n=sum(groups[i][0] for i in selection)
        if n:
            for arm in range(3):
                boots[arm].append(sum(groups[i][1+arm] for i in selection)/n)
    for arr in boots:arr.sort()
    valid=all(len(arr)==draws for arr in boots)
    intervals=[[arr[50],arr[1949]] for arr in boots] if valid and draws==2000 else [None]*3
    ld=sum(g[4] for g in groups)
    ls=sum(g[6] for g in groups)
    ratio=ls/ld if ld>ZERO else None
    signs=[sum(g[8+j] for g in groups) for j in range(4)]
    fits_good=(fits['lad_source']['identified'] and fits['lad_other']['identified']
               and fits['original_ridge_source']['identified'])
    gate_cal=bool(fits_good and total>=24 and puzzles>=20 and valid)
    gate_gains=bool(valid and all(ci[0]>0 for ci in intervals))
    gate_ratio=ratio is not None and ratio<=.95
    gate_sign=signs[2]>=signs[0]
    return {
       'status':'measured' if valid else 'bootstrap_abstention',
       'eligible_goals':total,'eligible_puzzles':puzzles,
       'mean_gain_vs_direct':sum(g[1] for g in groups)/total,
       'mean_gain_vs_original_ridge':sum(g[2] for g in groups)/total,
       'mean_gain_vs_other_LAD':sum(g[3] for g in groups)/total,
       'direct_MAE':ld/total,
       'ridge_source_MAE':sum(g[5] for g in groups)/total,
       'source_LAD_MAE':ls/total,
       'other_LAD_MAE':sum(g[7] for g in groups)/total,
       'source_LAD_over_direct_MAE_ratio':ratio,
       'direct_sign_correct':signs[0],'ridge_sign_correct':signs[1],
       'source_LAD_sign_correct':signs[2],'other_LAD_sign_correct':signs[3],
       'direct_sign_accuracy':signs[0]/total,
       'ridge_sign_accuracy':signs[1]/total,
       'source_LAD_sign_accuracy':signs[2]/total,
       'other_LAD_sign_accuracy':signs[3]/total,
       'bootstrap95_gain_vs_direct':intervals[0],
       'bootstrap95_gain_vs_original_ridge':intervals[1],
       'bootstrap95_gain_vs_other_LAD':intervals[2],
       'valid_bootstrap_draws':len(boots[0]),
       'bootstrap_requested_draws':draws,
       'registered_original_calibration_and_test_coverage_gate':gate_cal,
       'registered_three_positive_paired_CI_gate':gate_gains,
       'registered_relative_direct_MAE_gate':gate_ratio,
       'registered_sign_nondegradation_gate':gate_sign,
       'registered_success':bool(gate_cal and gate_gains and gate_ratio and gate_sign)
    }


def evaluate_family(ctx,comps,reference,refit,mapping,rosters,family,index,betas):
    g=prepare_geometry(ctx,comps,reference,refit,mapping,rosters,family,index)
    co=g['cohort'];stage=co['stage1'];positions=g['positions']
    truth=co['truth_at_goals'];clean=co['clean_logits']
    eligible=g['eligible'];n=len(rosters)
    model=ctx[comps[reference][0]]['model64']
    for param in model.parameters():param.requires_grad_(False)
    print('SMC22_PREOUTCOME_GEOMETRY_FROZEN',family,index,int(eligible.sum()),flush=True)
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
            ridge=direct+betas['ridge_source']['coefficient']*ra
            vals.append({'gradient_48':gg.tolist(),
                'reference_delta_48':a.tolist(),'target_delta_48':b.tolist(),
                'other_delta_48':o.tolist(),
                'reference_native_drop':float(ea[p,j]),
                'other_native_drop':float(eo[p,j]),
                'source_finite_residual':ra,'other_finite_residual':ro,
                'direct_prediction':direct,
                'source_calibrated_prediction':src,
                'other_calibrated_prediction':other,
                'ridge_source_prediction':ridge,
                'unit_anchor_prediction':direct+ra})
        preds.append(vals)
    print('SMC22_ALL_CALIBRATION_ONLY_PREDICTIONS_FROZEN',
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
            pd=value['direct_prediction'];ps=value['source_calibrated_prediction'];po=value['other_calibrated_prediction'];pr=value['ridge_source_prediction']
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
                'ridge_absolute_error':abs(actual-pr),
                'gain_vs_ridge':abs(actual-pr)-abs(actual-ps),
                'ridge_sign_correct':sign_eps(actual)==sign_eps(pr),
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



def run(original,independent,conditioned,frozen_dir,source_calibration,lad_calibration,
        output,fetch_source=False,source_checkpoint=None,pilot=0,pair_limit=4):
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError('registered SMC22 pilot/full only')
    teachers,comps,old,locks,s19,fh=frozen_sources(
        original,independent,conditioned,frozen_dir,fetch_source,source_checkpoint)
    if fh!=SMC19_SHA:raise ValueError('original SMC19 refit SHA drift')
    sb=Path(source_calibration).read_bytes()
    if sha(sb)!=SMC21_CALIB_SHA:raise ValueError('original SMC21 calibration source file changed')
    orig_cal=json.loads(sb)
    ladb=Path(lad_calibration).read_bytes()
    lad=json.loads(ladb)
    if (lad['schema']!='SMC-22/original-source-only-LAD-freeze-v1' or
        lad['pre_registration_commit']!=PRECOMMIT or
        lad['original_smc21_calibration_json_sha256']!=SMC21_CALIB_SHA or
        lad['original_smc19_frozen_refit_json_sha256']!=SMC19_SHA or
        lad['source_only_calibration_frozen_before_new_tests'] is not True or
        len(lad['pairs'])!=4):
        raise ValueError('original SMC22 source-only LAD calibration not frozen')
    if orig_cal['source_original_receipts']!=locks or len(orig_cal['pairs'])!=4:
        raise ValueError('SMC21 original calibration source teacher changed')
    for i,(a,b) in enumerate(PAIRS):
        fr=lad['pairs'][i];prior=s19['pair_calibration'][i]
        if ((fr['source'],fr['target'])!=(a,b) or
            fr['frozen_SMC19_mapping_hashes']!=prior['mapping_hashes'] or
            fr['original_ridge_source']!=orig_cal['pairs'][i]['beta_source']):
            raise ValueError('original SMC19 frozen calibration source mismatch')
    print('SMC22_SOURCE_AND_ORIGINAL_LAD_FITS_LOCKED',sha(ladb),flush=True)
    # Construct new heldout states only after source+calibration SHA checking.
    rosters,manifest=fresh_rosters(old,pilot or 64)
    from smc14_causal_transfer import source_group_context
    ctx,_=source_group_context(teachers,comps,rosters,tuple(FAMILIES))
    results=[]
    for i,(a,b) in enumerate(PAIRS[:pair_limit]):
        fit=lad['pairs'][i]
        betas={'beta_source':{'coefficient':fit['lad_source']['coefficient'],
                              'identified':fit['lad_source']['identified']},
               'beta_other':{'coefficient':fit['lad_other']['coefficient'],
                             'identified':fit['lad_other']['identified']},
               'ridge_source':fit['original_ridge_source']}
        heldout={}
        for family in FAMILIES:
            raw,coverage=evaluate_family(ctx,comps,a,b,
                         s19['pair_calibration'][i]['mapping']['matched'],
                         rosters[family],family,i,betas)
            stats=score_family(raw,fit,202622130+100*i+len(family))
            heldout[family]={'per_puzzle':raw,'coverage':coverage,'statistics':stats}
        both=bool(not pilot and pair_limit==4 and
                  all(heldout[family]['statistics']['registered_success'] for family in FAMILIES))
        results.append({'source':a,'target':b,
                        'original_SMC19_matched_mapping_sha256':fit['frozen_SMC19_mapping_hashes'],
                        'frozen_source_LAD':fit['lad_source']['coefficient'],
                        'frozen_other_LAD':fit['lad_other']['coefficient'],
                        'original_source_ridge':fit['original_ridge_source']['coefficient'],
                        'heldout':heldout,'registered_both_new_families_success':both})
        print('SMC22_WEIGHTED_MEDIAN_RESULT',b,{
          fam:{'eligible':heldout[fam]['statistics']['eligible_goals'],
               'vs_direct':heldout[fam]['statistics'].get('mean_gain_vs_direct'),
               'vs_ridge':heldout[fam]['statistics'].get('mean_gain_vs_original_ridge'),
               'vs_other':heldout[fam]['statistics'].get('mean_gain_vs_other_LAD'),
               'gate':heldout[fam]['statistics']['registered_success']}
          for fam in FAMILIES},flush=True)
    result={
       'schema':'SMC-22/weighted-median-source-residual-fresh-test-v1',
       'pre_registration_commit':PRECOMMIT,
       'original_SMC19_refit_json_sha256':SMC19_SHA,
       'original_SMC21_calibration_json_sha256':SMC21_CALIB_SHA,
       'SMC22_frozen_LAD_json_sha256':sha(ladb),
       'source_original_receipts':locks,
       'original_SMC21_source_calibration':orig_cal,
       'frozen_SMC22_LAD_calibration':lad,
       'no_new_SAE_or_teacher_training':True,
       'fresh_rosters':rosters,'fresh_manifest_sha256':manifest,
       'test_seed_bases':{k:v[0] for k,v in FAMILIES.items()},
       'puzzles_per_family':pilot or 64,
       'full_protocol':not bool(pilot or pair_limit<4),
       'fixed_parameters':{'bootstrap_draws':2000,
         'direct_MAE_ratio_gate':.95,
         'sign_zero_epsilon':1e-8,
         'LAD_clip':[-2.,2.],
         'degenerate_LAD_source_weight_cutoff':1e-12},
       'pairs':results,
       'scope':'Answer-conditioned clean task-gradient on same frozen teacher; independent absolute-loss calibration-only scalar without retraining or changed feature matching.'
    }
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-artifact',required=True,type=Path)
    p.add_argument('--independent-artifact',required=True,type=Path)
    p.add_argument('--conditioned-receipt',required=True,type=Path)
    p.add_argument('--frozen-dir',required=True,type=Path)
    p.add_argument('--smc21-calibration',required=True,type=Path)
    p.add_argument('--lad-calibration',required=True,type=Path)
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,))
    p.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--out',type=Path,default=Path('runs/smc22/full.json'))
    a=p.parse_args()
    run(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
        a.frozen_dir,a.smc21_calibration,a.lad_calibration,a.out,
        a.fetch_source,a.source_checkpoint,a.pilot_puzzles or 0,a.pair_limit)


if __name__=='__main__':main()
