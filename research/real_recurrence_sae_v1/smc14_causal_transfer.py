"""SMC-14: source-locked finite causal-response transfer across trained teachers.

All alignment permutations are rebuilt from the original SMC-13C *calibration*
cohort and MUST match the source 96-feature assignment SHA256s. Global
signed orientation is fitted on calibration causal margins alone. The
heldout puzzle families never influence model/SAE weights, assignments,
intervention signs or stopping decisions.

This is a bounded FP64 causal-intervention experiment, not an Arb proof
of exact-real causal semantics. OLM-14 documents the mathematical limits.
"""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import build_rosters
from smc13_train_teachers import (
    tensors,source_model_from_lock,FAMILIES
)
from smc13_transfer import (
    load_independent_models,hungarian_maximum
)
from smc13_conditioned_transfer import (
    conditional_affinity,conditional_pair_shuffle
)
from smc12_cones import load_frozen_sources
from smc13_transfer import puzzle_bootstrap

CONDITIONED_JSON_SHA256='3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07'
ORIGINAL_SOURCE_RECEIPT_SHA256='7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087'
NEW_TEACHER_RECEIPT_SHA256='1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3'
SOURCE_TEACHER_SHA256='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
CONDITION='position_digit'
RADIUS=0.2
GOALS=8
NULL_SHUFFLES=8
PAIRS=[
    ('pretrainedA_71','newB_501_SAE_71'),
    ('pretrainedA_71','newB_501_SAE_89'),
    ('pretrainedA_71','newB_701_SAE_71'),
    ('pretrainedA_71','newB_701_SAE_89'),
    ('newB_501_SAE_71','newB_501_SAE_89'),
    ('newB_701_SAE_71','newB_701_SAE_89'),
    ('newB_501_SAE_71','newB_701_SAE_71'),
]
SPLITS=('feature_calibration',)+tuple(FAMILIES)
ASSIGNMENTS=('matched','clue_shuffled','random')
TEST_PUZZLES=64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def goal_positions(inputs,n_goals=GOALS):
    """Eight evenly spaced blank positions chosen without target labels."""
    if inputs.ndim!=2 or inputs.shape[1]!=81:
        raise ValueError('expected 81 Sudoku input cells per puzzle')
    chosen=[]
    for row in inputs:
        blank=torch.nonzero(row==0,as_tuple=False).view(-1)
        if len(blank)<n_goals:
            raise ValueError('too few visible blanks')
        idx=(torch.arange(n_goals)*len(blank))//n_goals
        positions=blank[idx].tolist()
        if len(set(positions))!=n_goals:
            raise ValueError('duplicate selected blank positions')
        chosen.append(positions)
    return torch.tensor(chosen,dtype=torch.int64)


def top_features_and_support(sae,mean,scale,stage):
    with torch.no_grad():
        z=sae.encode((stage-mean)/scale)
        chosen=z.argmax(-1)
        values=z.topk(2,dim=-1).values
        strict=(values[...,0]>0)&(values[...,0]>values[...,1])
        support=(z>0).to(torch.float32)
    return chosen,strict,support


def physical_decoder_unit_table(sae,scale):
    """Source SMC-09 physical intervention convention, all 96 atoms."""
    with torch.no_grad():
        product=(sae.decoder.weight.detach()*scale[:,None]).T.to(torch.float64)
        norms=product.norm(dim=-1,keepdim=True)
        if not bool((norms>1e-12).all()):
            raise ValueError('zero physical decoder vector')
        return (product/norms).detach()


def fixed_competitors(clean,targets):
    """Largest non-target CLEAN logit, frozen before both interventions."""
    if clean.ndim!=2 or clean.shape[-1]!=10 or targets.shape!=(len(clean),):
        raise ValueError('bad teacher output or target shape')
    altered=clean.clone()
    altered[torch.arange(len(clean)),targets]=-float('inf')
    return altered.argmax(-1)


def signed_margin_response(teacher64,stage1,clean_logits,targets,positions,
                           feature_ids,unit_directions,radius=RADIUS):
    """Per-puzzle vectorized paired +/- goal-token interventions.

    teacher64: frozen FP64 recurrence; stage1: original FP32 source output.
    One sample per blank goal and sign; all other 80 tokens copied unchanged.
    Score true target logit against the teacher's strongest clean rival.
    """
    if radius<=0 or feature_ids.shape!=positions.shape or targets.shape!=positions.shape:
        raise ValueError('invalid intervention geometry')
    if stage1.ndim!=3 or stage1.shape[1:]!=(81,48) or len(stage1)!=len(positions):
        raise ValueError('frozen stage1 batch must be Nx81x48')
    if unit_directions.shape!=(96,48) or unit_directions.dtype!=torch.float64:
        raise ValueError('physical decoder table must be FP64 96x48')
    k=positions.shape[1]
    if k!=GOALS:
        raise ValueError('wrong predeclared goal count')
    rows=[];correct=[]
    with torch.no_grad():
        for p in range(len(positions)):
            ix=positions[p].long(); truth=targets[p].long()
            unit=unit_directions[feature_ids[p].long()]
            anchor=stage1[p:p+1].double()
            h=anchor.repeat(2*k,1,1)
            jj=torch.arange(k)
            h[jj,ix]+=radius*unit
            h[k+jj,ix]-=radius*unit
            for _ in range(3):
                h=teacher64.block(h)
            output=teacher64.head(h)
            plus=output[jj,ix,:]
            minus=output[k+jj,ix,:]
            base=clean_logits[p,ix,:]
            rival=fixed_competitors(base,truth)
            plus_margin=plus[jj,truth]-plus[jj,rival]
            minus_margin=minus[jj,truth]-minus[jj,rival]
            response=(plus_margin-minus_margin)/(2*radius)
            if not bool(torch.isfinite(response).all()):
                raise ArithmeticError('nonfinite centered causal response')
            rows.append(response.tolist())
            correct.append(base.argmax(-1).eq(truth).tolist())
    return torch.tensor(rows,dtype=torch.float64),torch.tensor(correct,dtype=torch.bool)


def centered_cosine(a,b,valid=None):
    a=torch.as_tensor(a,dtype=torch.float64)
    b=torch.as_tensor(b,dtype=torch.float64)
    if a.shape!=b.shape or a.ndim!=1:
        raise ValueError('paired vector of k goal effects required')
    if valid is not None:
        v=torch.as_tensor(valid,dtype=torch.bool)
        if v.shape!=a.shape:
            raise ValueError('validity shape mismatch')
        a=a[v];b=b[v]
    if len(a)<3:
        return None
    aa=a-a.mean();bb=b-b.mean()
    denom=float(aa.norm()*bb.norm())
    if denom<=1e-14:
        return None
    return float((aa@bb)/denom)


def centered_inner(a,b,valid):
    a=torch.as_tensor(a,dtype=torch.float64)[valid]
    b=torch.as_tensor(b,dtype=torch.float64)[valid]
    if len(a)<3:return None
    return float(((a-a.mean())@(b-b.mean())))


def calibration_orientation(source,target,valid):
    """One global sign from calibration puzzles; heldout data are absent."""
    if source.shape!=target.shape or valid.shape!=source.shape:
        raise ValueError('calibration data geometry mismatch')
    totals=[]
    for i in range(len(source)):
        x=centered_inner(source[i],target[i],valid[i])
        if x is not None:totals.append(x)
    if not totals:
        return {'sign':1,'unidentified':True,'usable_calibration_puzzles':0,'total_centered_inner':0.}
    total=sum(totals)
    return {'sign':1 if total>=0 else -1,
            'unidentified':total==0,
            'usable_calibration_puzzles':len(totals),
            'total_centered_inner':total}


def interval_bootstrap(values,seed,draws=2000):
    """Resample 64 puzzle clusters, preserving within-puzzle eight tokens."""
    if not values:
        return None
    rng=random.Random(seed);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                 for _ in range(draws))
    return {'mean':sum(values)/n,
            'puzzle_bootstrap95':[means[int(.025*draws)],means[int(.975*draws)-1]],
            'n_independent_puzzles':n,'resamples':draws}


def within_puzzle_shuffle_null(a,b,valid,orientation,pair_seed,puzzle_index,
                              draws=NULL_SHUFFLES):
    """Test-time negative control only: shuffle target responses across goals."""
    rng=random.Random(202614000+pair_seed*1009+puzzle_index)
    k=len(b)
    output=[]
    for _ in range(draws):
        inds=list(range(k));rng.shuffle(inds)
        shuffled=b[inds]*orientation
        val=centered_cosine(a,shuffled,valid)
        if val is not None:output.append(val)
    return sum(output)/len(output) if output else None


def source_group_context(teachers,components,rosters,keys):
    """Cache FP32 stage1 and FP64 continuation, only for authorized splits.

    Caller must first invoke this with calibration-only, lock the exact
    feature permutations, then invoke it separately for heldout keys.
    """
    teacher_ctx={}
    supports={}
    for key in keys:
        rows=rosters[key]
        x,y=tensors(rows)
        pos=goal_positions(x)
        tdata={'x':x,'targets':y,'goals':pos}
        for label,teacher in teachers.items():
            with torch.no_grad():
                h=teacher.reason_states(x)[1].detach()
                dbl=teacher_ctx[label]['model64'] if label in teacher_ctx else copy.deepcopy(teacher).double().eval()
                state=h.double()
                # Clean output follows the same exact FP32 stage1 anchor as +/-.
                for _ in range(3):
                    state=dbl.block(state)
                logits=dbl.head(state).detach()
            if label not in teacher_ctx:
                teacher_ctx[label]={'model64':dbl,'cohorts':{}}
            teacher_ctx[label]['cohorts'][key]={
                'stage1':h,'clean_logits':logits,
                'inputs':x,'targets':y,'positions':pos,
                'truth_at_goals':y.gather(1,pos)}
        for component,(teacher_id,sae,mean,scale) in components.items():
            stage=teacher_ctx[teacher_id]['cohorts'][key]['stage1']
            top,strict,positive=top_features_and_support(sae,mean,scale,stage)
            supports.setdefault(component,{})[key]={
                'feature_top':top.gather(1,pos),
                'strict':strict.gather(1,pos),
                'support':positive,
                'goal_support':positive.gather(1,pos[:,:,None].expand(-1,-1,96)),
            }
        tdata['n']=len(rows)
    return teacher_ctx,supports


def exact_frozen_assignments(A,B,calib_inputs,source_record,pair_index):
    """Require same Hungarian assignments as ORIGINAL SMC-13C receipt."""
    score,_=conditional_affinity(A,B,calib_inputs,CONDITION)
    match=hungarian_maximum(score)
    shuffled=conditional_pair_shuffle(B,calib_inputs,711+pair_index+len(CONDITION))
    null_score,_=conditional_affinity(A,shuffled,calib_inputs,CONDITION)
    hard_null=hungarian_maximum(null_score)
    rng=random.Random(20261440+pair_index)
    random_assignment=list(range(96));rng.shuffle(random_assignment)
    if sorted(match)!=list(range(96)) or sorted(hard_null)!=list(range(96)):
        raise ValueError('calibration assignment is not a bijection')
    observed=digest(json.dumps(match).encode())
    null_hash=digest(json.dumps(hard_null).encode())
    expected=source_record['conditional_protocol'][CONDITION]
    if observed!=expected['permutation_sha256']:
        raise ValueError('SMC-13C position-digit conditioned feature assignment SHA mismatch')
    if null_hash!=expected['within_group_shuffled_assignment_sha256']:
        raise ValueError('SMC-13C within-group shuffled feature assignment SHA mismatch')
    return {'matched':match,'clue_shuffled':hard_null,'random':random_assignment},{
        'matched_assignment_sha256':observed,
        'clue_shuffled_assignment_sha256':null_hash,
        'random_assignment_sha256':digest(json.dumps(random_assignment).encode()),
        'matched_to_original_smc13c_receipt':True,
    }


def puzzle_case_scores(source_effect,target_effects,valid,orientations,
                       source_correct,target_correct,target_support,
                       mapped_feature_ids,seed,pair_index,family,index,
                       goal_positions=None,ground_truth_digits=None):
    labels={}
    for control in ASSIGNMENTS:
        response=target_effects[control]
        sign=orientations[control]['sign']
        labels[control]=centered_cosine(source_effect,response*sign,valid)
    within=within_puzzle_shuffle_null(
        source_effect,target_effects['matched'],valid,
        orientations['matched']['sign'],pair_index,index)
    good=int(valid.sum())
    hits={}
    for control in ASSIGNMENTS:
        ids=mapped_feature_ids[control]
        picked=target_support[torch.arange(GOALS),ids]
        hits[control]=float(picked.float().mean())
    return {
        'puzzle_index':index,'n_valid_source_top_features':good,
        'matched_effect_cosine':labels['matched'],
        'within_clue_shuffled_effect_cosine':labels['clue_shuffled'],
        'random_feature_assignment_effect_cosine':labels['random'],
        'within_puzzle_shuffled_goal_response_cosine':within,
        'source_goal_correct_count':int(source_correct.sum()),
        'target_goal_correct_count':int(target_correct.sum()),
        'both_goal_correct_count':int((source_correct&target_correct).sum()),
        'target_matched_atom_top4_rate':hits['matched'],
        'source_causal_response':source_effect.tolist(),
        'target_matched_causal_response':target_effects['matched'].tolist(),
        'target_clue_shuffled_causal_response':target_effects['clue_shuffled'].tolist(),
        'target_random_assignment_causal_response':target_effects['random'].tolist(),
        'source_strict_top1_selection_mask':valid.tolist(),
        'selected_goal_positions':goal_positions.tolist() if goal_positions is not None else None,
        'selected_goal_truth_digits':ground_truth_digits.tolist() if ground_truth_digits is not None else None,
        'n_missing_goal_tokens':GOALS,
        'nonfinite_or_invalid':any(x is None for x in labels.values()),
    }


def summarize_family(rows,seed):
    valid=[x for x in rows if not x['nonfinite_or_invalid']
           and x['within_puzzle_shuffled_goal_response_cosine'] is not None]
    if not valid:
        return {'status':'all_abstained','n_puzzles':len(rows),'valid_puzzles':0}
    keys={
      'matched':'matched_effect_cosine',
      'clue_shuffled':'within_clue_shuffled_effect_cosine',
      'random':'random_feature_assignment_effect_cosine',
      'within_goal_shuffle':'within_puzzle_shuffled_goal_response_cosine',
    }
    means={k:sum(x[v] for x in valid)/len(valid) for k,v in keys.items()}
    differences={}
    for k in ('clue_shuffled','random','within_goal_shuffle'):
        vals=[x[keys['matched']]-x[keys[k]] for x in valid]
        differences[k]=interval_bootstrap(vals,seed+len(k),2000)
    return {
      'status':'measured',
      'n_puzzles':len(rows),'valid_puzzles':len(valid),
      'abstained_puzzles':len(rows)-len(valid),
      'means_oriented_centered_causal_cosine':means,
      'paired_matched_minus_null':differences,
      'primary_both_mapping_nulls_pass':all(
           differences[k]['puzzle_bootstrap95'][0]>0
           for k in ('clue_shuffled','random')),
      'matched_target_atom_top4_rate':sum(x['target_matched_atom_top4_rate'] for x in valid)/len(valid),
      'source_missing_digit_goal_accuracy':sum(x['source_goal_correct_count'] for x in valid)/(len(valid)*GOALS),
      'target_missing_digit_goal_accuracy':sum(x['target_goal_correct_count'] for x in valid)/(len(valid)*GOALS),
      'both_missing_digit_correct_rate':sum(x['both_goal_correct_count'] for x in valid)/(len(valid)*GOALS),
      'per_puzzle':rows,
    }


def load_conditioned(path):
    raw=Path(path).read_bytes()
    if digest(raw)!=CONDITIONED_JSON_SHA256:
        raise ValueError('SMC-13C conditioned calibration JSON SHA256 mismatch')
    value=json.loads(raw)
    if value['schema']!='SMC-13C/observable-clue-conditioned-cross-model-feature-transfer-v1':
        raise ValueError('wrong source-feature calibration schema')
    if len(value['results'])!=len(PAIRS):
        raise ValueError('calibration pair roster incomplete')
    return value


def run(source_checkpoint=None,allow_fetch=False,original_artifact=None,
        independent_artifact=None,conditioned_receipt=None,
        pilot_puzzles=None,only_pairs=None):
    if not original_artifact or not independent_artifact or not conditioned_receipt:
        raise ValueError('three immutable source artifacts required')
    torch.set_num_threads(2)
    original,osaes,oprovenance=load_frozen_sources(extracted_dir=original_artifact)
    if digest(Path(original_artifact,'joint-full.json').read_bytes())!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError('source pretrained SAE receipt mismatch')
    source,_=source_model_from_lock(allow_fetch,source_checkpoint)
    independent,rosters,new=load_independent_models(independent_artifact)
    if digest(Path(independent_artifact,'full.json').read_bytes())!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError('independently trained checkpoint receipt mismatch')
    conditioned=load_conditioned(conditioned_receipt)
    models={'A':source}
    components={'pretrainedA_71':('A',*osaes[71])}
    for seed,group in new.items():
        label=f'B{seed}'
        models[label]=group['teacher']
        for sae_seed,params in group['saes'].items():
            components[f'newB_{seed}_SAE_{sae_seed}']=(label,*params)
    pairs=PAIRS if only_pairs is None else PAIRS[:only_pairs]
    # First evaluate ONLY calibration inputs, then lock and source-hash
    # every feature correspondence. Heldout stages, targets and predictions
    # do not exist in the process until this loop has finished.
    calib_ctx,calib_supports=source_group_context(
        models,components,rosters,('feature_calibration',))
    frozen={}
    for index,(a,b) in enumerate(pairs):
        original_pair=conditioned['results'][index]
        if (original_pair['source'],original_pair['target'])!=(a,b):
            raise ValueError('calibration pair sequence differs')
        observed=calib_ctx[components[a][0]]['cohorts']['feature_calibration']['inputs']
        assigns,sha=exact_frozen_assignments(
            calib_supports[a]['feature_calibration']['support'],
            calib_supports[b]['feature_calibration']['support'],
            observed,original_pair,index)
        frozen[index]=(assigns,sha)
    print('SMC14_CALIBRATION_ASSIGNMENTS_FROZEN',
          {i:frozen[i][1] for i in frozen},flush=True)
    test_ctx,test_supports=source_group_context(
        models,components,rosters,FAMILIES)
    for key in calib_ctx:
        calib_ctx[key]['cohorts'].update(test_ctx[key]['cohorts'])
    for key in calib_supports:
        calib_supports[key].update(test_supports[key])
    ctx,supports=calib_ctx,calib_supports
    reports=[]
    for index,(a,b) in enumerate(pairs):
        assigns,sha=frozen[index]
        component_directions={
            a:physical_decoder_unit_table(components[a][1],components[a][3]),
            b:physical_decoder_unit_table(components[b][1],components[b][3])}
        orientations={}
        source_effects={}
        for family in SPLITS:
            t=ctx[components[a][0]]['cohorts'][family]
            go=t['positions'];truth=t['truth_at_goals']
            num=pilot_puzzles if pilot_puzzles and family!='feature_calibration' else len(go)
            src_feat=supports[a][family]['feature_top'][:num]
            source_effects[family],source_correct=signed_margin_response(
                ctx[components[a][0]]['model64'],t['stage1'][:num],
                t['clean_logits'][:num],truth[:num],go[:num],
                src_feat,component_directions[a])
            if family=='feature_calibration':
                source_calibration_correct=source_correct
            else:
                source_effects[family]=(source_effects[family],source_correct)
        cache={}
        for family in SPLITS:
            t=ctx[components[b][0]]['cohorts'][family]
            go=t['positions'];truth=t['truth_at_goals']
            num=pilot_puzzles if pilot_puzzles and family!='feature_calibration' else len(go)
            selected_source=supports[a][family]['feature_top'][:num]
            target_effects={}
            target_correct={}
            for kind in ASSIGNMENTS:
                assignment=torch.tensor(assigns[kind],dtype=torch.int64)
                selected_B=assignment[selected_source]
                response,correct=signed_margin_response(
                    ctx[components[b][0]]['model64'],t['stage1'][:num],
                    t['clean_logits'][:num],truth[:num],go[:num],
                    selected_B,component_directions[b])
                target_effects[kind]=response
                target_correct[kind]=correct
            if family=='feature_calibration':
                valid=supports[a][family]['strict']
                calibration_causal_raw={
                    'source_response':source_effects[family].tolist(),
                    'source_strict_mask':valid.tolist(),
                    'target_response_by_assignment':{
                        kind:target_effects[kind].tolist()
                        for kind in ASSIGNMENTS},
                    'selected_goal_positions':go.tolist(),
                    'selected_goal_truth_digits':truth.tolist(),
                    'purpose':'global ±1 orientation fitted exclusively on 64 calibration puzzles'
                }
                for kind in ASSIGNMENTS:
                    orientations[kind]=calibration_orientation(
                        source_effects[family],target_effects[kind],valid)
            else:
                A_effect,A_correct=source_effects[family]
                entries=[]
                n=num
                for i in range(n):
                    valid=supports[a][family]['strict'][i]
                    # Pairwise comparison uses exactly same goal positions
                    # and valid subset for treatment and all mapping nulls.
                    target_ids={kind:torch.tensor(assigns[kind],dtype=torch.int64)[
                        supports[a][family]['feature_top'][i]]
                        for kind in ASSIGNMENTS}
                    goal_support=supports[b][family]['goal_support'][i]
                    row=puzzle_case_scores(A_effect[i],
                         {key:values[i] for key,values in target_effects.items()},
                         valid,orientations,
                         A_correct[i],target_correct['matched'][i],
                         goal_support,target_ids,index,index,family,i,
                         go[i],truth[i])
                    entries.append(row)
                cache[family]=summarize_family(entries,seed=20261400+index*100+len(family))
        primary={name:cache[name].get('primary_both_mapping_nulls_pass',False)
                 for name in ('test_novel_30','test_novel_34')}
        reports.append({
          'source_component':a,'target_component':b,
          'comparison_type':'original_pretrained_to_new_teacher' if a=='pretrainedA_71' else
                            'same_teacher_independent_SAE' if a.split('_SAE_')[0]==b.split('_SAE_')[0] else 'independent_new_teachers',
          'calibration':{'source_puzzles':64,'orientation':orientations,
                         'assignment_sha256':sha,'feature_assignments':assigns,
                         'source_causal_orientation_data':calibration_causal_raw,
                         'calibration_only_before_heldout':True},
          'heldout':cache,
          'registered_success_on_both_novel_families':bool(all(primary.values())),
        })
        print('SMC14_CAUSAL_TRANSFER',a,b,{
            fam:{'matched':round(cache[fam]['means_oriented_centered_causal_cosine']['matched'],4),
                 'shuffled':round(cache[fam]['means_oriented_centered_causal_cosine']['clue_shuffled'],4),
                 'random':round(cache[fam]['means_oriented_centered_causal_cosine']['random'],4),
                 'passes':cache[fam]['primary_both_mapping_nulls_pass']}
            for fam in FAMILIES if cache[fam]['status']=='measured'},flush=True)
    return {
      'schema':'SMC-14/calibration-locked-cross-teacher-causal-transfer-v1',
      'source_teacher_sha256':SOURCE_TEACHER_SHA256,
      'original_frozen_sae_receipt_sha256':ORIGINAL_SOURCE_RECEIPT_SHA256,
      'independent_teacher_training_receipt_sha256':NEW_TEACHER_RECEIPT_SHA256,
      'conditioned_alignment_receipt_sha256':CONDITIONED_JSON_SHA256,
      'novel_roster_hashes':{k:v['sha256'] for k,v in independent['novel_family']['cohorts'].items()},
      'feature_calibration_puzzles':64,
      'test_puzzles_per_cohort':pilot_puzzles or TEST_PUZZLES,
      'goal_tokens_per_puzzle':GOALS,
      'intervention_radius_l2':RADIUS,
      'torch_version':torch.__version__,
      'source_precision':'FP32 source stage1 anchor, FP64 frozen three-step continuation',
      'calibration':'SMC13C position+visible-clue centered Hungarian assignments; global sign fit on calibration causal margins, never heldout',
      'outcome':'8 within-puzzle centered finite correct-digit margin responses per teacher, orientation-frozen cosine; bootstrap 64 puzzle IDs',
      'comparisons':reports,
      'limitations':'Numerical FP64 causal associations, not formal interval-certified exact-real transfer; source teacher much more accurate than new teachers; same generator architecture, deterministic seed clusters, original pretrained corpus partly unknown, and multiple pair comparisons.',
      'pilot_only':bool(pilot_puzzles or only_pairs),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--original-artifact',type=Path,required=True)
    p.add_argument('--independent-artifact',type=Path,required=True)
    p.add_argument('--conditioned-receipt',type=Path,required=True)
    p.add_argument('--pilot-puzzles',type=int,choices=(4,8,16))
    p.add_argument('--only-pairs',type=int,choices=(1,2,3,4,5,6,7))
    p.add_argument('--out',type=Path,default=Path('runs/smc14/full.json'))
    a=p.parse_args()
    result=run(a.source_checkpoint,a.fetch_source,a.original_artifact,
               a.independent_artifact,a.conditioned_receipt,
               a.pilot_puzzles,a.only_pairs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
      'out':str(a.out),'pilot_only':result['pilot_only'],
      'comparisons':[
        {'pair':row['source_component']+' -> '+row['target_component'],
         'gates':{name:v['primary_both_mapping_nulls_pass']
                  for name,v in row['heldout'].items()}}
        for row in result['comparisons']
      ]
    },indent=2))


if __name__=='__main__':
    main()
