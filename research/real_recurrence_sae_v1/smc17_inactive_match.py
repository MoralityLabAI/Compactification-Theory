"""SMC-17: source-frozen inactive-encoder leakage-matched tangent sham.

One intervention suppresses the matched naturally active SAE atom. The
comparison remains tangent to all four original active encoder rows, but a
fixed-budget outcome-blind optimizer matches the 92 inactive affine shifts.
This is a numerical conditional intervention study, not a global optimizer
certificate or an Arb-verified exact-real recurrent result.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from smc13_family import new_full_solution,dig_unique,valid_solution,rectangle_count,count_solutions,fingerprint
from smc16_selective_kill import (
    new_rosters as smc16_rosters,selective_vectors,physical_covectors,
    score as smc16_score,
)
from smc15_active_necessity import encoded_goals,physical_columns,ablate_margin

FAMILIES={'smc17_novel_30':(202617030000,30),
          'smc17_novel_34':(202617040000,34)}
PAIRS=(('pretrainedA_71','newB_501_SAE_71'),
       ('pretrainedA_71','newB_501_SAE_89'),
       ('pretrainedA_71','newB_701_SAE_71'),
       ('pretrainedA_71','newB_701_SAE_89'))
PRE_REGISTRATION_COMMIT='06598be765bec4742596a044328886334059d3a7'
STEPS=128
OVERSHOOT=.02
LEAK_RATIO_GATE=.85
N_GOALS=8
BOOTSTRAPS=2000
EPS=1e-8


def new_rosters(original,limit=64):
    if limit not in (4,64):
        raise ValueError('only four-puzzle smoke or sixty-four-puzzle full')
    # Exhaust prior test rosters at full 64 samples, including pilots.
    prior16,_=smc16_rosters(original,limit=64)
    from smc15_active_necessity import fresh_rosters as smc15_rosters
    prior15,_=smc15_rosters(original,limit=64)
    seen_x=set()
    seen_y=set()
    for roster in list(original.values())+list(prior15.values())+list(prior16.values()):
        for row in roster:
            x,y=tuple(row['x']),tuple(row['target'])
            if x in seen_x or y in seen_y:
                raise ValueError('prior SMC13-16 roster not disjoint')
            seen_x.add(x);seen_y.add(y)
    out={}
    for family,(base,holes) in FAMILIES.items():
        rows=[]
        for ix in range(limit):
            seed=base+ix
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('SMC17 novel target invalid or canonical')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError('SMC17 target not uniquely determined')
            if any(v and v!=y[i] for i,v in enumerate(x)):
                raise ValueError('puzzle conflicts with solution')
            if tuple(y) in seen_y or tuple(x) in seen_x:
                raise ValueError('SMC17 puzzle/solution overlaps earlier data')
            seen_y.add(tuple(y));seen_x.add(tuple(x))
            rows.append({'seed':seed,'x':x,'target':y,'holes':holes,
                         'rectangles':rectangle_count(y)})
        out[family]=rows
    return out,{k:fingerprint(v) for k,v in out.items()}


def encode_one(sae,mean,scale,state):
    with torch.no_grad():
        return sae.encode(((state.float()-mean)/scale)[None,:])[0].double()


def seeded_generator(pair,family,seed,slot,feature,restart):
    msg=f'SMC17|{pair}|{family}|{seed}|{slot}|{feature}|{restart}'
    val=int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8],'big')%(2**63-1)
    return torch.Generator().manual_seed(val)


def optimal_tangent(h,clean_code,feature,kill,seeded,sae,mean,scale,cov,
                    pair,family,seed,slot):
    """Outcome-blind four-start 128-step sphere search.

    Keep *all* original starts and all iterates. Only encoder-state
    admissibility chooses a candidate; no task target or logits enter.
    """
    blank=torch.zeros(48,dtype=torch.float64)
    info={'status':'','baseline_inactive_mismatch_l2':None,
          'optimized_inactive_mismatch_l2':None,
          'inactive_mismatch_ratio':None,'winner_start':None,'winner_step':None,
          'winner_reencode_max_rel_drift':None,
          'baseline_reencode_max_rel_drift':None}
    active=torch.nonzero(clean_code>0).view(-1).tolist()
    if len(active)!=4 or feature not in active or not float(clean_code[feature])>0:
        info['status']='not_four_active'
        return blank,info
    if kill.dtype!=torch.float64 or seeded.dtype!=torch.float64:
        raise ValueError('physical edits must be FP64')
    r=float(kill.norm())
    if r<=EPS or abs(float(seeded.norm())-r)>1e-8*max(1.,r):
        info['status']='unmatched_initial_dose'
        return blank,info
    active_rows=cov[active]
    inactive=[j for j in range(96) if j not in active]
    I=cov[inactive]
    with torch.no_grad():
        _,s,Vh=torch.linalg.svd(active_rows,full_matrices=True)
        if len(s)!=4 or float(s.min())<=1e-10:
            info['status']='rank_defect'
            return blank,info
        N=Vh[4:,:].T.contiguous()
        if N.shape!=(48,44):
            raise ValueError('expected 44-dimensional nullspace')
        if float((active_rows@N).abs().max())>1e-8:
            info['status']='nullspace_inaccurate'
            return blank,info
        B=I@N
        y=I@kill
        baseline_error=float((I@(seeded-kill)).norm())
        info['baseline_inactive_mismatch_l2']=baseline_error
        if baseline_error<=EPS:
            info['status']='baseline_zero_inactive_difference'
            return seeded.clone(),info
        x0=N.T@seeded
        x1=B.T@y
        starts=[x0,x1]
        for restart in (2,3):
            g=seeded_generator(pair,family,seed,slot,feature,restart)
            starts.append(torch.randn(44,dtype=torch.float64,generator=g))
        xs=[]
        for x in starts:
            if float(x.norm())<=EPS:
                x=x0
            xs.append(r*x/float(x.norm()))
        batch=torch.stack(xs,dim=0)
        eta=1./(4.*float((B*B).sum()))
        if not math.isfinite(eta) or eta<=0:
            info['status']='invalid_inactive_encoder_geometry'
            return blank,info
        candidates=[]
        for step in range(STEPS+1):
            edits=batch@N.T
            errors=((edits-kill)@I.T).square().sum(dim=-1)
            for restart in range(4):
                candidates.append((float(errors[restart]),restart,step,edits[restart].clone()))
            if step==STEPS:break
            grad=2.*((batch@B.T-y)@B)
            tang=grad-((grad*batch).sum(dim=-1)/(r*r))[:,None]*batch
            proposal=batch-eta*tang
            denom=proposal.norm(dim=-1)
            if not bool((denom>EPS).all()):
                info['status']='optimization_numeric_failure'
                return blank,info
            batch=r*proposal/denom[:,None]
        # Geometry selects the best *reencoding-valid* point. The
        # unmodified seeded sham is an explicit fallback candidate.
        candidates.append((baseline_error**2,-1,-1,seeded.clone()))
        candidates.sort(key=lambda row:(row[0],row[1],row[2]))
        for _,restart,step,candidate in candidates:
            if abs(float(candidate.norm())-r)>1e-8*max(1.,r):
                continue
            if float((active_rows@candidate).abs().max())>1e-8:
                continue
            post=encode_one(sae,mean,scale,h.double()-candidate)
            drifts=[abs(float(post[j]-clean_code[j]))/(1+float(clean_code[j])) for j in active]
            if all(float(post[j])>0 for j in active) and max(drifts)<=1e-4:
                result=float((I@(candidate-kill)).norm())
                if result>baseline_error+1e-8:
                    raise ArithmeticError('optimized inactive leakage worse than fallback')
                info.update(status='eligible',optimized_inactive_mismatch_l2=result,
                            inactive_mismatch_ratio=result/baseline_error,
                            winner_start=restart,winner_step=step,
                            winner_reencode_max_rel_drift=max(drifts))
                return candidate.clone(),info
    info['status']='no_reencoding_valid_tangent'
    return blank,info


def bootstrap_scores(raw,seed,draws=BOOTSTRAPS):
    """Exactly the same pre-outcome mask for kill and both tangent shams."""
    per=[];flip=[0,0,0];clean=0
    for row in raw:
        selected=row['eligible_mask']
        if len(selected)!=N_GOALS:
            raise ValueError('eight goal slots required')
        effects=[];prior=[]
        for j,e in enumerate(selected):
            if not e:continue
            k,b,o=[row[n][j] for n in ('kill_margin_drop','seeded_margin_drop','optimized_margin_drop')]
            if any(x is None or not math.isfinite(x) for x in (k,b,o)):
                raise ValueError('missing eligible causal effect')
            effects.append(k-o);prior.append(k-b)
            if row['clean_correct'][j]:clean+=1
            flip[0]+=int(row['kill_flip'][j])
            flip[1]+=int(row['seeded_flip'][j])
            flip[2]+=int(row['optimized_flip'][j])
        per.append((sum(effects),sum(prior),len(effects)))
    n=sum(row[2] for row in per)
    clusters=sum(row[2]>0 for row in per)
    if not n:
        return {'status':'zero_eligible_goals','eligible_goals':0,
                'eligible_puzzles':0,'registered_gate':False}
    rng=random.Random(seed)
    draws_done=[]
    for _ in range(draws):
        idx=[rng.randrange(len(per)) for _ in per]
        count=sum(per[i][2] for i in idx)
        if count:
            draws_done.append(sum(per[i][0] for i in idx)/count)
    draws_done.sort()
    success_boot=len(draws_done)==draws
    interval=[draws_done[int(.025*draws)],draws_done[int(.975*draws)-1]] if success_boot else None
    leak_seed=[r['baseline_inactive_mismatch_l2']**2 for row in raw for r in row['diagnostics'] if r['status']=='eligible']
    leak_opt=[r['optimized_inactive_mismatch_l2']**2 for row in raw for r in row['diagnostics'] if r['status']=='eligible']
    if len(leak_seed)!=n or len(leak_opt)!=n or not all(v>=0 and math.isfinite(v) for v in leak_seed+leak_opt):
        raise ValueError('leakage measurement count inconsistent')
    leak_ratio=(sum(leak_opt)/sum(leak_seed))**.5 if sum(leak_seed)>0 else None
    gate=(n>=24 and clusters>=20 and success_boot and interval[0]>0
          and flip[0]>flip[2] and leak_ratio is not None and leak_ratio<=LEAK_RATIO_GATE)
    return {
       'status':'measured' if success_boot else 'bootstrap_abstention',
       'eligible_goals':n,'eligible_puzzles':clusters,
       'mean_kill_minus_optimized_margin_drop':sum(x[0] for x in per)/n,
       'mean_kill_minus_seeded_margin_drop_same_mask':sum(x[1] for x in per)/n,
       'unweighted_puzzle_mean_kill_minus_optimized':sum(x[0]/x[2] for x in per if x[2])/clusters,
       'bootstrap95_optimized':interval,'bootstrap_valid':len(draws_done),
       'bootstrap_requested':draws,
       'inactive_mismatch_rms_seeded':(sum(leak_seed)/n)**.5,
       'inactive_mismatch_rms_optimized':(sum(leak_opt)/n)**.5,
       'inactive_mismatch_rms_ratio':leak_ratio,
       'kill_clean_correct_to_wrong_count':flip[0],
       'seeded_clean_correct_to_wrong_count':flip[1],
       'optimized_clean_correct_to_wrong_count':flip[2],
       'clean_correct_eligible_goals':clean,
       'registered_gate':gate
    }


def evaluate_family(ctx,components,source,target,mapping,rows,family,pair):
    a=ctx[components[source][0]]['cohorts'][family]
    b=ctx[components[target][0]]['cohorts'][family]
    if not bool((a['inputs']==b['inputs']).all()):
        raise ValueError('paired source/target puzzle roster differs')
    pos=b['positions']
    _,top,strict=encoded_goals(a['stage1'],pos,*components[source][1:])
    sae,mean,scale=components[target][1:]
    z,_,_=encoded_goals(b['stage1'],pos,sae,mean,scale)
    mapped=torch.tensor(mapping,dtype=torch.int64)[top]
    cov=physical_covectors(sae,scale)
    decoder=physical_columns(sae,scale)
    n=len(rows)
    kill=torch.zeros((n,N_GOALS,48),dtype=torch.float64)
    baseline=torch.zeros_like(kill)
    optimized=torch.zeros_like(kill)
    diagnostics=[]
    for i in range(n):
        page=[]
        for j in range(N_GOALS):
            if not bool(strict[i,j]):
                entry={'status':'source_not_strict'}
                page.append(entry)
                continue
            h=b['stage1'][i,int(pos[i,j])]
            chosen=int(mapped[i,j])
            k,seeded,prior=selective_vectors(
                h,z[i,j],chosen,sae,mean,scale,cov,decoder,
                rows[i]['seed'],pair,family,j)
            entry={'status':prior['status'],'original_smc16':prior}
            if prior['status']=='eligible':
                opt,info=optimal_tangent(
                    h,z[i,j],chosen,k,seeded,sae,mean,scale,cov,
                    pair,family,rows[i]['seed'],j)
                entry.update(info)
                if info['status']=='eligible':
                    kill[i,j]=k;baseline[i,j]=seeded;optimized[i,j]=opt
                    entry.update({
                     'clean_active_ids':torch.nonzero(z[i,j]>0).flatten().tolist(),
                     'selected_feature_id':chosen,
                     'original_active_clean_coefficients':[float(z[i,j,q]) for q in torch.nonzero(z[i,j]>0).flatten().tolist()],
                     'kill_physical_delta':k.tolist(),
                     'seeded_physical_delta':seeded.tolist(),
                     'optimized_physical_delta':opt.tolist(),
                     'kill_dose':float(k.norm()),
                     'seeded_dose':float(seeded.norm()),
                     'optimized_dose':float(opt.norm()),
                    })
            page.append(entry)
        diagnostics.append(page)
    eligible=[[diag['status']=='eligible' for diag in page] for page in diagnostics]
    # All three pre-outcome edits are now frozen.
    labels=b['truth_at_goals']
    model=ctx[components[target][0]]['model64']
    dk,correct,fk=ablate_margin(model,b['stage1'],b['clean_logits'],labels,pos,kill)
    ds,correct_s,fs=ablate_margin(model,b['stage1'],b['clean_logits'],labels,pos,baseline)
    do,correct_o,fo=ablate_margin(model,b['stage1'],b['clean_logits'],labels,pos,optimized)
    if not bool(((correct==correct_s)&(correct==correct_o)).all()):
        raise ValueError('same original clean teacher must score all arms')
    raw=[]
    for i,roster in enumerate(rows):
        raw.append({
         'puzzle_index':i,'seed':roster['seed'],
         'selected_goal_positions':pos[i].tolist(),
         'goal_truth_digits':labels[i].tolist(),
         'source_top_feature_ids':top[i].tolist(),
         'target_matched_feature_ids':mapped[i].tolist(),
         'eligible_mask':eligible[i],'diagnostics':diagnostics[i],
         'kill_margin_drop':[float(dk[i,j]) if eligible[i][j] else None for j in range(N_GOALS)],
         'seeded_margin_drop':[float(ds[i,j]) if eligible[i][j] else None for j in range(N_GOALS)],
         'optimized_margin_drop':[float(do[i,j]) if eligible[i][j] else None for j in range(N_GOALS)],
         'clean_correct':correct[i].tolist(),
         'kill_flip':[bool(fk[i,j]) if eligible[i][j] else None for j in range(N_GOALS)],
         'seeded_flip':[bool(fs[i,j]) if eligible[i][j] else None for j in range(N_GOALS)],
         'optimized_flip':[bool(fo[i,j]) if eligible[i][j] else None for j in range(N_GOALS)]
        })
    counts={}
    for page in diagnostics:
        for d in page:
            counts[d['status']]=counts.get(d['status'],0)+1
    return raw,{'eligibility_status_counts':counts,'n_puzzles':n,
                'n_selected_goal_tokens':n*N_GOALS,
                'eligible_goals':sum(sum(x) for x in eligible),
                'physical_covectors_96_by_48':cov.tolist()}


def run(original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch=False,source_checkpoint=None,pilot=0,pair_limit=4):
    from smc14_causal_transfer import (
      load_frozen_sources,source_model_from_lock,load_independent_models,
      load_conditioned,source_group_context,exact_frozen_assignments,
      PAIRS as ORIGINAL_PAIRS,ORIGINAL_SOURCE_RECEIPT_SHA256,
      NEW_TEACHER_RECEIPT_SHA256,CONDITIONED_JSON_SHA256)
    torch.set_num_threads(2)
    if hashlib.sha256((Path(original_artifact)/'joint-full.json').read_bytes()).hexdigest()!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError('original source SAE receipt SHA mismatch')
    if hashlib.sha256((Path(independent_artifact)/'full.json').read_bytes()).hexdigest()!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError('independent teacher receipt SHA mismatch')
    _,osaes,_=load_frozen_sources(extracted_dir=original_artifact)
    pretrained,_=source_model_from_lock(allow_fetch,source_checkpoint)
    _,old,new=load_independent_models(independent_artifact)
    conditioned=load_conditioned(conditioned_receipt)
    teachers={'A':pretrained}
    comps={'pretrainedA_71':('A',*osaes[71])}
    for s,p in new.items():
        tid=f'B{s}';teachers[tid]=p['teacher']
        for seed,params in p['saes'].items():
            comps[f'newB_{s}_SAE_{seed}']=(tid,*params)
    if tuple(ORIGINAL_PAIRS[:4])!=PAIRS or pair_limit not in (1,2,3,4):
        raise ValueError('source-matched primary pair roster differs')
    calibration,features=source_group_context(teachers,comps,old,('feature_calibration',))
    matched=[]
    hashes=[]
    for i,(a,b) in enumerate(ORIGINAL_PAIRS):
        record=conditioned['results'][i]
        if (record['source'],record['target'])!=(a,b):
            raise ValueError('original SMC13C comparison order differs')
        original_inputs=calibration[comps[a][0]]['cohorts']['feature_calibration']['inputs']
        assignments,sha=exact_frozen_assignments(
            features[a]['feature_calibration']['support'],
            features[b]['feature_calibration']['support'],
            original_inputs,record,i)
        matched.append(assignments['matched']);hashes.append(sha)
    print('SMC17_ALL_SEVEN_SMC13C_MATCHINGS_VERIFIED',flush=True)
    # Construct new test states only after original matching hashes are frozen.
    rosters,manifest=new_rosters(old,pilot or 64)
    ctx,_=source_group_context(teachers,comps,rosters,tuple(FAMILIES))
    comparisons=[]
    for index,(source,target) in enumerate(PAIRS[:pair_limit]):
        tests={}
        for family in FAMILIES:
            raw,coverage=evaluate_family(
                ctx,comps,source,target,matched[index],rosters[family],family,index)
            stats=bootstrap_scores(raw,202617130+index*100+len(family))
            tests[family]={'statistics':stats,'coverage':coverage,'per_puzzle':raw}
        success=not bool(pilot or pair_limit!=4) and all(
            tests[f]['statistics'].get('registered_gate',False) for f in FAMILIES)
        comparisons.append({
          'source':source,'target':target,
          'frozen_matched_feature_mapping':matched[index],
          'matched_assignment_hashes':hashes[index],
          'heldout':tests,'registered_success_both_families':success,
        })
        print('SMC17_INACTIVE_LEAKAGE_MATCH',target,{
          f:{k:tests[f]['statistics'].get(k) for k in
             ('eligible_goals','inactive_mismatch_rms_ratio',
              'mean_kill_minus_optimized_margin_drop','registered_gate')}
          for f in FAMILIES},flush=True)
    return {
      'schema':'SMC-17/leakage-matched-active-SAE-kill-v1',
      'pre_registration_commit':PRE_REGISTRATION_COMMIT,
      'source_receipt_sha256':ORIGINAL_SOURCE_RECEIPT_SHA256,
      'independent_teacher_receipt_sha256':NEW_TEACHER_RECEIPT_SHA256,
      'conditioned_mapping_receipt_sha256':CONDITIONED_JSON_SHA256,
      'all_seven_source_assignment_hashes':hashes,
      'seed_bases':{f:v[0] for f,v in FAMILIES.items()},
      'fresh_manifest_sha256':manifest,'fresh_rosters':rosters,
      'fixed_search':{'starts':4,'steps':STEPS,'gradient_step':'1/(4*||B||_F^2)',
                      'inactive_leakage_rms_ratio_gate':LEAK_RATIO_GATE,
                      'goals_per_puzzle':N_GOALS,'bootstrap_draws':BOOTSTRAPS},
      'puzzles_per_cohort':pilot or 64,
      'full_protocol':not bool(pilot or pair_limit!=4),
      'comparisons':comparisons,
      'limitations':'Outcome-blind fixed-budget approximate inactive-affine matching, not exact inactive equivalence or a formal network mechanism proof.'
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
    p.add_argument('--out',type=Path,default=Path('runs/smc17/full.json'))
    args=p.parse_args()
    receipt=run(args.original_artifact,args.independent_artifact,
                args.conditioned_receipt,args.fetch_source,args.source_checkpoint,
                args.pilot_puzzles or 0,args.pair_limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(args.out),'full_protocol':receipt['full_protocol'],
      'pairs':[{ 'target':x['target'],'success':x['registered_success_both_families']}
      for x in receipt['comparisons']]},indent=2))


if __name__=='__main__':main()
