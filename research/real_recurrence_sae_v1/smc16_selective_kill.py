"""SMC-16: selective active-feature erasure in frozen Sudoku recurrent teachers.

This program uses the preregistered frozen SMC-13C mapping and new seeded
noncanonical Sudoku cohorts. Eligibility is decided from unsupervised source
and target encodings, never by outcome labels or intervention logits.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
import torch

from smc13_family import (
    new_full_solution,dig_unique,valid_solution,count_solutions,
    rectangle_count,fingerprint,
)
from smc15_active_necessity import (
    fresh_rosters as smc15_rosters,
    encoded_goals,physical_columns,ablate_margin,
)

FAMILIES={'smc16_novel_30':(202616030000,30),
          'smc16_novel_34':(202616040000,34)}
GOALS=8
PUZZLES=64
TAU=0.02
MAX_NATIVE_MULTIPLE=6.0
CODE_TOL=1e-4
ORTH_TOL=1e-8
BOOTSTRAPS=2000
PAIRS=(('pretrainedA_71','newB_501_SAE_71'),
       ('pretrainedA_71','newB_501_SAE_89'),
       ('pretrainedA_71','newB_701_SAE_71'),
       ('pretrainedA_71','newB_701_SAE_89'))


def new_rosters(original,limit=PUZZLES):
    if limit not in (4,64):
        raise ValueError('only four-case pilot or full 64-puzzle cohort')
    old15,_=smc15_rosters(original,limit=64)
    seen_y={tuple(row['target']) for rows in (list(original.values())+list(old15.values())) for row in rows}
    seen_x={tuple(row['x']) for rows in (list(original.values())+list(old15.values())) for row in rows}
    out={}
    for fam,(base,holes) in FAMILIES.items():
        rows=[]
        for idx in range(limit):
            seed=base+idx
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('new solution invalid or canonical')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError('new puzzle underdug, not uniquely solvable')
            if any(v and v!=y[k] for k,v in enumerate(x)):
                raise ValueError('clue contradicts solution')
            yt,xt=tuple(y),tuple(x)
            if yt in seen_y or xt in seen_x:
                raise ValueError('new cohort overlaps SMC13, SMC15 or itself')
            seen_y.add(yt);seen_x.add(xt)
            rows.append({'seed':seed,'x':x,'target':y,
                         'holes':holes,'rectangles':rectangle_count(y)})
        out[fam]=rows
    return out,{name:fingerprint(rows) for name,rows in out.items()}


def physical_covectors(sae,scale):
    with torch.no_grad():
        rows=sae.encoder.weight.detach().double()/scale.detach().double()[None,:]
    if rows.shape!=(96,48) or not bool(torch.isfinite(rows).all()):
        raise ValueError('source fitted encoder weights and scales required')
    return rows


def projected_normal(target_row,other_rows):
    """Exact orthogonal row projection up to FP64, fail on rank defect."""
    if target_row.shape!=(48,) or other_rows.ndim!=2 or other_rows.shape[1]!=48:
        raise ValueError('expected physical 48d covectors')
    with torch.no_grad():
        _,sing,Vh=torch.linalg.svd(other_rows,full_matrices=False)
        if len(sing) and float(sing.min())<=1e-10:
            return None
        proj=target_row-(Vh.T@(Vh@target_row))
        if float(proj.norm())<=1e-8:
            return None
        if float((other_rows@proj).abs().max())>ORTH_TOL:
            return None
        return proj


def encode_one(sae,mean,scale,h):
    with torch.no_grad():
        return sae.encode(((h.float()-mean)/scale)[None,:])[0].detach().double()


def selective_vectors(h,code,feature,sae,mean,scale,
                      covectors,decoders,seed,pair,family,slot):
    """Physical minimal-norm selective kill and exact-dose tangent sham.

    A failed code-cell check removes the goal *before* task labels or output
    are consulted. Returned reasons persist in the raw per-goal receipt.
    """
    blank=torch.zeros(48,dtype=torch.float64)
    info={'status':'', 'matched_clean_coefficient':float(code[feature]),
          'native_decoder_dose':0.,'kill_dose':0.,'sham_dose':0.,
          'native_dose_multiple':None,'killed_feature_reencoded':None,
          'other_active_max_drift':None,'sham_active_max_drift':None,
          'original_active_count':int((code>0).sum())}
    if float(code[feature])<=0:
        info['status']='matched_inactive'
        return blank,blank,info
    active=torch.nonzero(code>0).flatten().tolist()
    if len(active)!=4 or feature not in active:
        info['status']='incomplete_top4_support'
        return blank,blank,info
    others=[i for i in active if i!=feature]
    aj=covectors[feature]
    O=covectors[others]
    normal=projected_normal(aj,O)
    if normal is None:
        info['status']='rank_or_projection_obstruction'
        return blank,blank,info
    z=float(code[feature])
    kill=(z+TAU)*normal/float(normal@normal)
    dose=float(kill.norm())
    native=z*float(decoders[feature].norm())
    info['kill_dose']=dose
    info['native_decoder_dose']=native
    info['native_dose_multiple']=dose/native if native else None
    if native<=1e-12 or dose>MAX_NATIVE_MULTIPLE*native:
        info['status']='dose_cap'
        return blank,blank,info
    if (abs(float(aj@kill)-(z+TAU))>ORTH_TOL or
            float((O@kill).abs().max())>ORTH_TOL):
        info['status']='physical_constraint_failure'
        return blank,blank,info

    post_k=encode_one(sae,mean,scale,h.double()-kill)
    info['killed_feature_reencoded']=float(post_k[feature])
    drift=max(abs(float(post_k[i]-code[i]))/(1+float(code[i])) for i in others)
    info['other_active_max_drift']=drift
    if (float(post_k[feature])>1e-7 or
        any(float(post_k[i])<=0 for i in others) or drift>CODE_TOL):
        info['status']='kill_reencoding_failed'
        return blank,blank,info

    active_cov=covectors[active]
    tangent=None
    for attempt in range(4):
        msg=f'SMC16|{pair}|{family}|{seed}|{slot}|{feature}|{attempt}'
        generator=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8],'big')%(2**63-1))
        candidate=torch.randn(48,generator=generator,dtype=torch.float64)
        vec=projected_normal(candidate,active_cov)
        if vec is None:
            continue
        sham=vec*(dose/float(vec.norm()))
        if abs(float(sham.norm())-dose)>ORTH_TOL:
            continue
        post_s=encode_one(sae,mean,scale,h.double()-sham)
        drift_s=max(abs(float(post_s[i]-code[i]))/(1+float(code[i])) for i in active)
        if all(float(post_s[i])>0 for i in active) and drift_s<=CODE_TOL:
            tangent=sham
            info['sham_dose']=float(sham.norm())
            info['sham_active_max_drift']=drift_s
            break
    if tangent is None:
        info['status']='tangent_sham_reencoding_failed'
        return blank,blank,info
    info['status']='eligible'
    return kill,tangent,info


def score(rows,seed,draws=BOOTSTRAPS):
    sums=[];count=[];flips_k=[];flips_s=[]
    total_clean=0;clean_diffs=[]
    for row in rows:
        if len(row['eligible_mask'])!=GOALS:
            raise ValueError('eight frozen missing goals required')
        diffs=[];nk=0;ns=0
        for j,eligible in enumerate(row['eligible_mask']):
            if not eligible:
                continue
            k,s=row['kill_margin_drop'][j],row['sham_margin_drop'][j]
            if k is None or s is None or not math.isfinite(k-s):
                raise ValueError('eligible nonfinite or missing causal response')
            diffs.append(k-s)
            nk+=int(row['kill_correct_to_wrong'][j])
            ns+=int(row['sham_correct_to_wrong'][j])
            if row['clean_correct'][j]:
                total_clean+=1
                clean_diffs.append(k-s)
        sums.append(sum(diffs));count.append(len(diffs))
        flips_k.append(nk);flips_s.append(ns)
    denom=sum(count)
    n_puzzles=sum(v>0 for v in count)
    if denom==0:
        return {'status':'no_eligible_goals','eligible_goals':0,
                'nonempty_puzzle_clusters':0,'registered_success':False}
    rng=random.Random(seed)
    boots=[]
    for _ in range(draws):
        ids=[rng.randrange(len(rows)) for j in rows]
        total=sum(count[i] for i in ids)
        if total:
            boots.append(sum(sums[i] for i in ids)/total)
    boots.sort()
    valid=len(boots)==draws
    interval=[boots[int(.025*draws)],boots[int(.975*draws)-1]] if valid else None
    fk=sum(flips_k);fs=sum(flips_s)
    good=denom>=24 and n_puzzles>=20 and valid and interval[0]>0 and fk>fs
    return {
        'status':'measured' if valid else 'bootstrap_abstained',
        'eligible_goals':denom,'nonempty_puzzle_clusters':n_puzzles,
        'mean_selective_kill_minus_sham_margin_drop':sum(sums)/denom,
        'unweighted_mean_per_puzzle':sum(sums[i]/count[i] for i in range(len(rows)) if count[i])/n_puzzles,
        'puzzle_cluster_bootstrap95':interval,
        'valid_bootstrap_draws':len(boots),'requested_bootstrap_draws':draws,
        'clean_correct_eligible_goals':total_clean,
        'clean_correct_only_mean_margin_drop_difference':(sum(clean_diffs)/len(clean_diffs) if clean_diffs else None),
        'kill_clean_correct_to_wrong_count':fk,
        'sham_clean_correct_to_wrong_count':fs,
        'kill_flip_rate_eligible':fk/denom,'sham_flip_rate_eligible':fs/denom,
        'registered_success':good,
    }


def evaluate_pair_family(ctx,components,source,target,assignment,roster,family,pair):
    from smc15_active_necessity import encoded_goals
    ta=ctx[components[source][0]]['cohorts'][family]
    tb=ctx[components[target][0]]['cohorts'][family]
    p=ta['positions'];x=ta['inputs']
    if not bool((x==tb['inputs']).all()):
        raise ValueError('cross-teacher original puzzle input differs')
    _,top,strict=encoded_goals(ta['stage1'],p,*components[source][1:])
    t_sae,t_mean,t_scale=components[target][1:]
    code,_,_=encoded_goals(tb['stage1'],p,t_sae,t_mean,t_scale)
    mapped=torch.tensor(assignment,dtype=torch.long)[top]
    cov=physical_covectors(t_sae,t_scale)
    dec=physical_columns(t_sae,t_scale)
    n=len(roster)
    kill=torch.zeros((n,GOALS,48),dtype=torch.float64)
    sham=torch.zeros_like(kill)
    all_info=[]
    status=Counter()
    for i in range(n):
        prow=[]
        for j in range(GOALS):
            h=tb['stage1'][i,int(p[i,j])]
            if not bool(strict[i,j]):
                info={'status':'source_top_not_strict'}
                dk=torch.zeros(48,dtype=torch.float64);ds=dk
            else:
                dk,ds,info=selective_vectors(
                    h,code[i,j],int(mapped[i,j]),t_sae,t_mean,t_scale,
                    cov,dec,roster[i]['seed'],pair,family,j)
            kill[i,j]=dk;sham[i,j]=ds
            prow.append(info);status[info['status']]+=1
        all_info.append(prow)
    eligible=torch.tensor([[d['status']=='eligible' for d in row] for row in all_info],dtype=torch.bool)
    # All eligibility and interventions are frozen before reading any true-digit margins.
    drops_k,correct,flip_k=ablate_margin(ctx[components[target][0]]['model64'],
       tb['stage1'],tb['clean_logits'],tb['truth_at_goals'],p,kill)
    drops_s,correct_sham,flip_s=ablate_margin(ctx[components[target][0]]['model64'],
       tb['stage1'],tb['clean_logits'],tb['truth_at_goals'],p,sham)
    if not bool((correct==correct_sham).all()):
        raise ValueError('same frozen clean logits differ between paired controls')
    raw=[]
    for i in range(n):
        raw.append({
          'puzzle_index':i,'seed':roster[i]['seed'],
          'goal_positions':p[i].tolist(),
          'goal_truth_digits':tb['truth_at_goals'][i].tolist(),
          'source_top_feature_ids':top[i].tolist(),
          'target_matched_feature_ids':mapped[i].tolist(),
          'eligible_mask':eligible[i].tolist(),
          'edit_diagnostics':all_info[i],
          'kill_margin_drop':[float(drops_k[i,j]) if eligible[i,j] else None for j in range(GOALS)],
          'sham_margin_drop':[float(drops_s[i,j]) if eligible[i,j] else None for j in range(GOALS)],
          'clean_correct':correct[i].tolist(),
          'kill_correct_to_wrong':[bool(flip_k[i,j]) if eligible[i,j] else None for j in range(GOALS)],
          'sham_correct_to_wrong':[bool(flip_s[i,j]) if eligible[i,j] else None for j in range(GOALS)]
        })
    doses=[i['kill_dose'] for row in all_info for i in row if i['status']=='eligible']
    return raw, {
      'ineligibility_reason_counts':dict(status),
      'n_fresh_puzzles':n,
      'n_selected_blank_goals':n*GOALS,
      'eligible_goals':int(eligible.sum()),
      'target_clean_goal_accuracy':float(correct.float().mean()),
      'mean_kill_dose_eligible':sum(doses)/len(doses) if doses else None,
    }


def run(original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch=False,source_checkpoint=None,pilot=0,pair_limit=4):
    from smc14_causal_transfer import (
        load_frozen_sources,source_model_from_lock,load_independent_models,
        load_conditioned,source_group_context,exact_frozen_assignments,PAIRS as ALL_PAIRS,
        ORIGINAL_SOURCE_RECEIPT_SHA256,NEW_TEACHER_RECEIPT_SHA256,
        CONDITIONED_JSON_SHA256
    )
    torch.set_num_threads(2)
    if hashlib.sha256((Path(original_artifact)/'joint-full.json').read_bytes()).hexdigest()!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError('source SAE receipt changed')
    if hashlib.sha256((Path(independent_artifact)/'full.json').read_bytes()).hexdigest()!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError('independent teacher receipt changed')
    _,source_saes,_=load_frozen_sources(extracted_dir=original_artifact)
    teacherA,_=source_model_from_lock(allow_fetch,source_checkpoint)
    _,original_rosters,ind=load_independent_models(independent_artifact)
    condition=load_conditioned(conditioned_receipt)
    teachers={'A':teacherA}
    components={'pretrainedA_71':('A',*source_saes[71])}
    for seed,group in ind.items():
        label=f'B{seed}'
        teachers[label]=group['teacher']
        for sae_seed,params in group['saes'].items():
            components[f'newB_{seed}_SAE_{sae_seed}']=(label,*params)
    if tuple(ALL_PAIRS[:4])!=PAIRS or pair_limit not in (1,2,3,4):
        raise ValueError('source-to-new comparison roster changed')
    cal_ctx,cal_features=source_group_context(
        teachers,components,original_rosters,('feature_calibration',))
    frozen={}
    for i,(a,b) in enumerate(ALL_PAIRS):
        receipt=condition['results'][i]
        if (receipt['source'],receipt['target'])!=(a,b):
            raise ValueError('SMC13C pair ordering differs')
        inputs=cal_ctx[components[a][0]]['cohorts']['feature_calibration']['inputs']
        assignments,sha=exact_frozen_assignments(
            cal_features[a]['feature_calibration']['support'],
            cal_features[b]['feature_calibration']['support'],
            inputs,receipt,i)
        frozen[i]=(assignments['matched'],sha)
    print('SMC16_ALL_SEVEN_CONDITIONED_ASSIGNMENTS_FROZEN',flush=True)
    rosters,hashes=new_rosters(original_rosters,limit=pilot or PUZZLES)
    ctx,_=source_group_context(teachers,components,rosters,tuple(FAMILIES))
    comparisons=[]
    for i,(a,b) in enumerate(PAIRS[:pair_limit]):
        families={}
        for name in FAMILIES:
            raw,coverage=evaluate_pair_family(
                ctx,components,a,b,frozen[i][0],rosters[name],name,i)
            stats=score(raw,seed=202616130+i*100+len(name))
            families[name]={'per_puzzle':raw,'coverage':coverage,'statistics':stats}
        success=not bool(pilot or pair_limit<4) and all(
            families[name]['statistics']['registered_success'] for name in FAMILIES)
        comparisons.append({'source':a,'target':b,
                            'source_mapping_sha256':frozen[i][1],
                            'frozen_matched_feature_mapping':frozen[i][0],
                            'heldout':families,
                            'registered_success_both_families':success})
        print('SMC16_SELECTIVE_KILL',b,{
            k:{'effect':v['statistics'].get('mean_selective_kill_minus_sham_margin_drop'),
               'eligible':v['statistics']['eligible_goals'],
               'success':v['statistics']['registered_success']}
            for k,v in families.items()},flush=True)
    return {
       'schema':'SMC-16/selected-top4-encoder-preserving-sham-v1',
       'pre_registration_commit':'777ff633010898103d876b54ac5d84264c12a628',
       'source_receipt_sha256':ORIGINAL_SOURCE_RECEIPT_SHA256,
       'independent_training_receipt_sha256':NEW_TEACHER_RECEIPT_SHA256,
       'conditioned_mapping_receipt_sha256':CONDITIONED_JSON_SHA256,
       'all_seven_original_smc13c_hungarian_assignments_sha_verified':True,
       'all_seven_original_smc13c_mapping_sha256':[frozen[i][1] for i in range(len(ALL_PAIRS))],
       'fixed_parameters':{'overshoot':TAU,'native_dose_multiple_limit':MAX_NATIVE_MULTIPLE,
          'encoder_recode_tolerance':CODE_TOL,'orthogonality_tolerance':ORTH_TOL,
          'bootstrap_resamples':BOOTSTRAPS},
       'test_roster_seed_bases':{k:v[0] for k,v in FAMILIES.items()},
       'fresh_manifest_sha256':hashes,'fresh_rosters':rosters,
       'test_puzzles_per_cohort':pilot or PUZZLES,
       'full_protocol':not bool(pilot or pair_limit<4),
       'comparison_roster':comparisons,
       'limitations':'Finite FP64 causal margins with selective local-affine encoder constraints; perturbations may alter inactive SAE features and teacher hidden-state computations.',
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-artifact',type=Path,required=True)
    p.add_argument('--independent-artifact',type=Path,required=True)
    p.add_argument('--conditioned-receipt',type=Path,required=True)
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,))
    p.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--out',type=Path,default=Path('runs/smc16/full.json'))
    args=p.parse_args()
    result=run(args.original_artifact,args.independent_artifact,
               args.conditioned_receipt,args.fetch_source,args.source_checkpoint,
               args.pilot_puzzles or 0,args.pair_limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(args.out),'full_protocol':result['full_protocol'],
                      'comparisons':[{'target':r['target'],
                         'success':r['registered_success_both_families']}
                          for r in result['comparison_roster']]},indent=2))


if __name__=='__main__':
    main()
