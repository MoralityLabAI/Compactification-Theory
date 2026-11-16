"""SMC-20: anchored clean task-gradient prediction of native SAE refit erasure.

Source-locked, no SAE retraining. Four frozen SMC-19 independent refits and
clue-conditioned assignments are reloaded with their original file SHA
receipts, before constructing two novel 64-puzzle families. Source-native
erasure and the clean task gradient predict a withheld refit-native effect.
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
from smc15_active_necessity import (
    encoded_goals,physical_columns,feature_deltas,other_active_ids,
    dose_matched_other,ablate_margin
)
from smc17_causal_inactive import source_models
from smc19_sae_refit import load_frozen as load_smc19_frozen, fresh_rosters as smc19_rosters
from smc18_task_gradient import task_gradients

PRECOMMIT='eb4b6c419fd741c4915d105235a9979426398710'
FAMILIES={'smc20_novel_30':(202620030000,30),
          'smc20_novel_34':(202620040000,34)}
PAIRS=(
    ('newB_501_SAE_71','newB_501_SAE_131'),
    ('newB_501_SAE_71','newB_501_SAE_167'),
    ('newB_701_SAE_71','newB_701_SAE_131'),
    ('newB_701_SAE_71','newB_701_SAE_167'),
)
GOALS=8
PUZZLES=64
BOOTSTRAPS=2000
ZERO=1e-8
RATIO_GATE=.90
PHYSICAL_TOL=1e-10


def fresh_rosters(old,limit=PUZZLES):
    """Full source-disjoint SMC13-19 exclusion before any new state forward."""
    if limit not in (4,64):
        raise ValueError('SMC20 allows only registered four-puzzle pilot and 64-puzzle full')
    prior,_=smc19_rosters(old,limit=64)
    xs={tuple(r['x']) for rows in list(old.values())+list(prior.values()) for r in rows}
    ys={tuple(r['target']) for rows in list(old.values())+list(prior.values()) for r in rows}
    result={}
    for family,(base,holes) in FAMILIES.items():
        new=[]
        for j in range(limit):
            seed=base+j
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('SMC20 complete solution not novel/noncanonical')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError('SMC20 visible puzzle not exact-unique with registered holes')
            if any(v and v!=y[i] for i,v in enumerate(x)):
                raise ValueError('SMC20 given clue conflicts with final solution')
            tx,ty=tuple(x),tuple(y)
            if tx in xs or ty in ys:
                raise ValueError('SMC20 new Sudoku duplicates prior source/training/heldout')
            xs.add(tx);ys.add(ty)
            new.append({'seed':seed,'x':x,'target':y,'holes':holes,
                        'rectangles':rectangle_count(y)})
        result[family]=new
    return result,{name:fingerprint(rows) for name,rows in result.items()}


def sign_eps(value,cutoff=ZERO):
    if abs(value)<=cutoff:
        return 0
    return 1 if value>0 else -1


def anchored_predict(ref_effect,gradient,reference_delta,refit_delta):
    if gradient.shape!=(48,) or reference_delta.shape!=(48,) or refit_delta.shape!=(48,):
        raise ValueError('SMC20 predictors use physical 48D vectors')
    prediction=float(ref_effect)+float(gradient@(refit_delta-reference_delta))
    if not math.isfinite(prediction):
        raise ArithmeticError('nonfinite anchored causal prediction')
    return prediction


def score_family(rows,seed,draws=BOOTSTRAPS):
    if not rows:
        raise ValueError('SMC20 needs at least one puzzle cluster')
    clusters=[]
    for row in rows:
        if len(row['eligible_mask'])!=GOALS or len(row['metrics'])!=GOALS:
            raise ValueError('each puzzle has eight frozen blank goal slots')
        n=0;gain_a=0.;gain_o=0.;err_a=0.;err_o=0.;err_g=0.
        signs=[0,0,0]
        for flag,item in zip(row['eligible_mask'],row['metrics']):
            if bool(flag)!=(item is not None):
                raise ValueError('ineligible goal cannot contribute an observed loss')
            if not flag:continue
            n+=1
            obs=item['observed_target_drop']
            p_g=item['gradient_anchored_prediction']
            p_a=item['reference_only_prediction']
            p_o=item['other_active_surrogate_prediction']
            if not all(math.isfinite(x) for x in (obs,p_g,p_a,p_o)):
                raise ValueError('nonfinite target native erasure or predictor')
            ag=abs(obs-p_g);aa=abs(obs-p_a);ao=abs(obs-p_o)
            gain_a+=aa-ag;gain_o+=ao-ag
            err_a+=aa;err_o+=ao;err_g+=ag
            signs[0]+=int(sign_eps(obs)==sign_eps(p_a))
            signs[1]+=int(sign_eps(obs)==sign_eps(p_o))
            signs[2]+=int(sign_eps(obs)==sign_eps(p_g))
        clusters.append((n,gain_a,gain_o,err_a,err_o,err_g,*signs))
    n=sum(x[0] for x in clusters)
    puzzles=sum(x[0]>0 for x in clusters)
    if n==0:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_success':False}
    rng=random.Random(seed)
    boot_a=[];boot_o=[]
    for _ in range(draws):
        draw=[rng.randrange(len(clusters)) for row in clusters]
        denom=sum(clusters[i][0] for i in draw)
        if denom:
            boot_a.append(sum(clusters[i][1] for i in draw)/denom)
            boot_o.append(sum(clusters[i][2] for i in draw)/denom)
    boot_a.sort();boot_o.sort()
    valid=len(boot_a)==draws and len(boot_o)==draws
    ci_a=[boot_a[50],boot_a[1949]] if valid and draws==2000 else None
    ci_o=[boot_o[50],boot_o[1949]] if valid and draws==2000 else None
    baseline=sum(x[3] for x in clusters)
    best=sum(x[5] for x in clusters)
    ratio=best/baseline if baseline>ZERO else None
    correct_a=sum(x[6] for x in clusters)
    correct_o=sum(x[7] for x in clusters)
    correct_g=sum(x[8] for x in clusters)
    coverage=n>=24 and puzzles>=20 and valid
    improvement=bool(valid and ci_a[0]>0 and ci_o[0]>0)
    relative=ratio is not None and ratio<=RATIO_GATE
    sign=correct_g>=correct_a
    return {
      'status':'measured' if valid else 'bootstrap_abstention',
      'eligible_goals':n,'eligible_puzzles':puzzles,
      'mean_anchor_minus_gradient_absolute_error_gain':sum(x[1] for x in clusters)/n,
      'mean_other_surrogate_minus_gradient_absolute_error_gain':sum(x[2] for x in clusters)/n,
      'mean_reference_only_absolute_error':baseline/n,
      'mean_other_surrogate_absolute_error':sum(x[4] for x in clusters)/n,
      'mean_gradient_anchored_absolute_error':best/n,
      'relative_gradient_to_reference_absolute_error':ratio,
      'reference_only_sign_accuracy':correct_a/n,
      'other_surrogate_sign_accuracy':correct_o/n,
      'gradient_anchored_sign_accuracy':correct_g/n,
      'reference_only_sign_correct':correct_a,
      'other_surrogate_sign_correct':correct_o,
      'gradient_anchored_sign_correct':correct_g,
      'puzzle_cluster_bootstrap95_gain_over_reference':ci_a,
      'puzzle_cluster_bootstrap95_gain_over_other':ci_o,
      'valid_bootstrap_resamples':len(boot_a),
      'requested_bootstrap_resamples':draws,
      'registered_coverage_gate':coverage,
      'registered_both_bootstrap_improvement_gate':improvement,
      'registered_relative_mae_gate':relative,
      'registered_sign_nondegradation_gate':sign,
      'registered_success':bool(coverage and improvement and relative and sign),
    }


def evaluate_pair_family(ctx,comps,reference,refit,mapping,
                         rosters,family,pair_index):
    old=ctx[comps[reference][0]]['cohorts'][family]
    new=ctx[comps[refit][0]]['cohorts'][family]
    if comps[reference][0]!=comps[refit][0] or not bool((old['inputs']==new['inputs']).all()):
        raise ValueError('SMC20 requires exactly same trained teacher and fresh inputs')
    n=len(rosters)
    stage=old['stage1'];pos=old['positions'];truth=old['truth_at_goals'];clean=old['clean_logits']
    za,top,strict=encoded_goals(stage,pos,*comps[reference][1:])
    zb,_,_=encoded_goals(stage,pos,*comps[refit][1:])
    mapped=torch.tensor(mapping,dtype=torch.long)[top]
    ctrl,avail=other_active_ids(zb,mapped,pair_index,family,[r['seed'] for r in rosters])
    eligible=strict&avail
    a_columns=physical_columns(comps[reference][1],comps[reference][3])
    b_columns=physical_columns(comps[refit][1],comps[refit][3])
    physical_a,z_source=feature_deltas(za,top,a_columns)
    physical_b,z_refit=feature_deltas(zb,mapped,b_columns)
    physical_o=dose_matched_other(physical_b,ctrl,b_columns)
    if not bool(torch.allclose(physical_b.norm(dim=-1),
                               physical_o.norm(dim=-1),atol=1e-12,rtol=1e-12)):
        raise ArithmeticError('original registered SMC19 other-active dose mismatch')
    # All edits, feature selection and pre-outcome source-blind alternative are frozen.
    active_lists=[]
    statuses=[]
    for p in range(n):
        one_active=[];one_status=[]
        for j in range(GOALS):
            active=torch.nonzero(zb[p,j]>0).flatten().tolist()
            one_active.append(active)
            if not bool(strict[p,j]):
                one_status.append('reference_not_strict_top1')
            elif float(z_refit[p,j])<=0:
                one_status.append('refit_matched_inactive')
            elif not bool(avail[p,j]):
                one_status.append('refit_missing_distinct_active_control')
            elif not bool(eligible[p,j]):
                raise AssertionError('source/target preoutcome eligibility mismatch')
            else:
                one_status.append('eligible')
        active_lists.append(one_active);statuses.append(one_status)
    print('SMC20_PREOUTCOME_GEOMETRY_FROZEN',family,pair_index,
          int(eligible.sum()),flush=True)
    model=ctx[comps[reference][0]]['model64']
    for param in model.parameters():param.requires_grad_(False)
    # The reference-only causal effect is permitted as part of the predictor.
    reference_drop,correct,flip_a=ablate_margin(
        model,stage,clean,truth,pos,physical_a)
    gradients=torch.zeros(n,GOALS,48,dtype=torch.float64)
    anchored=torch.zeros(n,GOALS,dtype=torch.float64)
    other_surrogate=torch.zeros(n,GOALS,dtype=torch.float64)
    direct_linear=torch.zeros(n,GOALS,dtype=torch.float64)
    for p in range(n):
        gradients[p]=task_gradients(
            model,stage[p],clean[p],truth[p],pos[p],eligible[p])
        for j in range(GOALS):
            if not bool(eligible[p,j]):continue
            g=gradients[p,j]
            anchored[p,j]=anchored_predict(
                reference_drop[p,j],g,physical_a[p,j],physical_b[p,j])
            other_surrogate[p,j]=anchored_predict(
                reference_drop[p,j],g,physical_a[p,j],physical_o[p,j])
            direct_linear[p,j]=g@physical_b[p,j]
    if not bool(torch.isfinite(anchored).all() and torch.isfinite(other_surrogate).all()):
        raise ArithmeticError('nonfinite frozen refit native erasure predictor')
    print('SMC20_REFERENCE_ANCHOR_AND_CLEAN_GRADIENT_PREDICTIONS_FROZEN',
          family,pair_index,int(eligible.sum()),flush=True)
    # The target's native refit intervention is opened only after all predictors
    # have been specified and computed from the original source/drop/clean g.
    target_drop,correct_t,flip_b=ablate_margin(
        model,stage,clean,truth,pos,physical_b)
    control_drop,correct_c,flip_o=ablate_margin(
        model,stage,clean,truth,pos,physical_o)
    if not bool(((correct==correct_t)&(correct==correct_c)).all()):
        raise AssertionError('single frozen receiving teacher clean target changed')
    raw=[]
    for p,row in enumerate(rosters):
        metrics=[]
        for j in range(GOALS):
            if not bool(eligible[p,j]):
                metrics.append(None)
                continue
            da=physical_a[p,j];db=physical_b[p,j];dc=physical_o[p,j]
            g=gradients[p,j]
            ea=float(reference_drop[p,j]); eb=float(target_drop[p,j]);eo=float(control_drop[p,j])
            pg=float(anchored[p,j]); po=float(other_surrogate[p,j])
            physical_distance=float((db-da).norm())
            physical_norms=[float(v.norm()) for v in (da,db,dc)]
            if min(physical_norms)<=1e-12 or abs(physical_norms[1]-physical_norms[2])>1e-10:
                raise ArithmeticError('nonpositive/mismatched native refit control physical dose')
            decoder_cos=float((da@db)/(da.norm()*db.norm()))
            item={
              'reference_feature':int(top[p,j]),
              'refit_matched_feature':int(mapped[p,j]),
              'refit_other_feature':int(ctrl[p,j]),
              'reference_selected_coefficient':float(z_source[p,j]),
              'refit_selected_coefficient':float(z_refit[p,j]),
              'refit_other_selected_coefficient':float(zb[p,j,ctrl[p,j]]),
              'reference_delta_48':da.tolist(),'target_delta_48':db.tolist(),
              'other_active_delta_48':dc.tolist(),'clean_task_gradient_48':g.tolist(),
              'physical_edit_distance':physical_distance,
              'reference_physical_dose':physical_norms[0],
              'refit_physical_dose':physical_norms[1],
              'other_physical_dose':physical_norms[2],
              'physical_edit_cosine':decoder_cos,
              'observed_reference_drop':ea,
              'observed_target_drop':eb,
              'observed_other_active_drop':eo,
              'gradient_anchored_prediction':pg,
              'reference_only_prediction':ea,
              'other_active_surrogate_prediction':po,
              'direct_clean_linear_refit_prediction':float(direct_linear[p,j]),
              'reference_only_absolute_error':abs(eb-ea),
              'other_surrogate_absolute_error':abs(eb-po),
              'gradient_anchored_absolute_error':abs(eb-pg),
              'absolute_error_gain_over_reference':abs(eb-ea)-abs(eb-pg),
              'absolute_error_gain_over_other':abs(eb-po)-abs(eb-pg),
              'reference_only_predicted_sign':sign_eps(ea),
              'other_surrogate_predicted_sign':sign_eps(po),
              'gradient_anchored_predicted_sign':sign_eps(pg),
              'observed_target_sign':sign_eps(eb),
              'clean_correct':bool(correct[p,j]),
              'source_clean_correct_to_wrong':bool(flip_a[p,j]),
              'refit_clean_correct_to_wrong':bool(flip_b[p,j]),
              'other_clean_correct_to_wrong':bool(flip_o[p,j]),
            }
            metrics.append(item)
        raw.append({
          'puzzle_index':p,'seed':row['seed'],
          'goal_positions':pos[p].tolist(),
          'goal_truth_digits':truth[p].tolist(),
          'source_selected_ids':top[p].tolist(),
          'target_matched_ids':mapped[p].tolist(),
          'refit_other_active_ids':ctrl[p].tolist(),
          'refit_active_feature_ids':active_lists[p],
          'reference_top1_coefficients':z_source[p].tolist(),
          'reference_top2_coefficients':za[p].topk(2,dim=-1).values[:,1].tolist(),
          'refit_matched_coefficients':z_refit[p].tolist(),
          'refit_other_coefficients':zb[p].gather(-1,ctrl[p,:,None]).squeeze(-1).tolist(),
          'eligible_mask':eligible[p].tolist(),
          'preoutcome_reasons':statuses[p],
          'metrics':metrics,
        })
    reasons={}
    for row in raw:
        for key in row['preoutcome_reasons']:
            reasons[key]=reasons.get(key,0)+1
    return raw,{
      'selected_goal_tokens':n*GOALS,
      'eligible_goals':int(eligible.sum()),
      'preoutcome_rejection_counts':reasons,
      'reference_physical_decoder_columns_96x48':a_columns.tolist(),
      'refit_physical_decoder_columns_96x48':b_columns.tolist(),
      'target_clean_goal_accuracy':float(correct.float().mean()),
    }


def run(original_artifact,independent_artifact,conditioned_receipt,frozen_dir,
        allow_fetch=False,source_checkpoint=None,pilot=0,pair_limit=4):
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError('only registered SMC20 four-puzzle pilot and full')
    torch.set_num_threads(2)
    teachers,comps,old,original_assignments,locks=source_models(
        original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch,source_checkpoint)
    frozen,frozen_sha=load_smc19_frozen(frozen_dir,comps,locks,old)
    if (frozen_sha!='612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b'
        or frozen['source_original_seven_mappings']!=original_assignments):
        raise ValueError('SMC19 frozen refit source or seven original assignment SHA differs')
    print('SMC20_SOURCE_AND_FOUR_SMC19_REFITS_REVERIFIED',frozen_sha,flush=True)
    from smc14_causal_transfer import source_group_context
    rosters,manifest=fresh_rosters(old,limit=pilot or PUZZLES)
    ctx,_=source_group_context(teachers,comps,rosters,tuple(FAMILIES))
    pairs=[]
    for index,(ref,refit) in enumerate(PAIRS[:pair_limit]):
        cal=frozen['pair_calibration'][index]
        if (cal['source'],cal['target'])!=(ref,refit):
            raise ValueError('SMC19 original paired calibration refit order changed')
        outcomes={}
        for fam in FAMILIES:
            raw,coverage=evaluate_pair_family(ctx,comps,ref,refit,
                  cal['mapping']['matched'],rosters[fam],fam,index)
            stats=score_family(raw,202620130+100*index+len(fam))
            outcomes[fam]={'per_puzzle':raw,'coverage':coverage,'statistics':stats}
        full=not bool(pilot or pair_limit<4)
        passed=bool(full and all(outcomes[f]['statistics']['registered_success']
                                for f in FAMILIES))
        pairs.append({
          'source':ref,'target':refit,
          'frozen_calibration_mapping':cal['mapping']['matched'],
          'frozen_mapping_hashes':cal['mapping_hashes'],
          'frozen_smc19_calibration_orientations':cal['calibration']['orientation'],
          'heldout':outcomes,'registered_both_family_success':passed})
        print('SMC20_ANCHORED_REFIT_PREDICTION',ref,refit,{
          f:{'eligible':outcomes[f]['statistics']['eligible_goals'],
             'gain_vs_reference':outcomes[f]['statistics'].get(
                  'mean_anchor_minus_gradient_absolute_error_gain'),
             'gain_vs_other':outcomes[f]['statistics'].get(
                  'mean_other_surrogate_minus_gradient_absolute_error_gain'),
             'mae_ratio':outcomes[f]['statistics'].get(
                  'relative_gradient_to_reference_absolute_error'),
             'gate':outcomes[f]['statistics']['registered_success']}
          for f in FAMILIES},flush=True)
    return {
      'schema':'SMC-20/source-frozen-anchored-task-gradient-refit-v1',
      'pre_registration_commit':PRECOMMIT,
      'source_original_receipts':locks,
      'smc19_frozen_json_sha256':frozen_sha,
      'smc19_frozen_source':frozen,
      'old_original_seven_mappings_hash_checked':True,
      'refit_calibrations_locked_before_new_test':True,
      'fresh_rosters':rosters,
      'fresh_manifest_sha256':manifest,
      'fresh_seed_bases':{k:v[0] for k,v in FAMILIES.items()},
      'puzzles_per_family':pilot or PUZZLES,
      'full_protocol':not bool(pilot or pair_limit<4),
      'fixed_parameters':{'bootstrap_draws':BOOTSTRAPS,
         'numerical_sign_cutoff':ZERO,'relative_absolute_error_gate':RATIO_GATE,
         'physical_tolerance':PHYSICAL_TOL,
         'gradient_forward_tolerance':2e-8},
      'pairs':pairs,
      'limitation':'Numerical clean FP64 task-gradient surrogate of one frozen teacher; observation of source native-erasure is an input; independent audit replays saved numerical observations without independent teacher forward pass.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-artifact',type=Path,required=True)
    p.add_argument('--independent-artifact',type=Path,required=True)
    p.add_argument('--conditioned-receipt',type=Path,required=True)
    p.add_argument('--frozen-dir',type=Path,required=True)
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,))
    p.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--out',type=Path,default=Path('runs/smc20/full.json'))
    args=p.parse_args()
    receipt=run(args.original_artifact,args.independent_artifact,
                args.conditioned_receipt,args.frozen_dir,args.fetch_source,
                args.source_checkpoint,args.pilot_puzzles or 0,args.pair_limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'full_protocol':receipt['full_protocol'],
          'pair_gates':[{'target':item['target'],'pass_both_families':item['registered_both_family_success']}
                        for item in receipt['pairs']]},indent=2))


if __name__=='__main__':main()
