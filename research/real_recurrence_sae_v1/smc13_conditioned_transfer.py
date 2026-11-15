"""SMC-13C: remove observed-position and given-digit confounds in SAE matching.

The calibration data are independent of teacher training, SAE training and
all three heldout puzzle families. The optimal 96-feature matching is based
on coactivation residuals after centering *within position/input-symbol*
groups on calibration puzzles only. Group labels use visible Sudoku
inputs, NEVER target digits at blank cells or heldout scores.

Paired-null assignments shuffle B activations independently WITHIN every
position+input-symbol group, preserving group-conditioned feature marginals.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import build_rosters
from smc13_train_teachers import tensors,source_model_from_lock,FAMILIES
from smc13_transfer import (
    load_independent_models,hungarian_maximum,
    heldout_overlap,binary_sae_support,
    puzzle_bootstrap,null_permutations,
)
from smc12_cones import load_frozen_sources


def conditional_affinity(A,B,inputs,condition='position_digit'):
    """Exact grouping over public (position, visible symbol), FP64 arithmetic."""
    if A.shape!=B.shape or A.ndim!=3 or A.shape[-1]!=96:
        raise ValueError('paired equally sized positive SAE supports required')
    if inputs.shape!=A.shape[:2]:
        raise ValueError('calibration source inputs must align with paired tokens')
    if condition not in ('position','position_digit'):
        raise ValueError('unsupported observable conditioning variable')
    n,p,d=A.shape
    output=torch.zeros((d,d),dtype=torch.float64)
    used_groups=0;singletons=0;weighted_total=0
    a=A.double();b=B.double()
    for j in range(p):
        unique=(torch.unique(inputs[:,j]).tolist()
                if condition=='position_digit' else [None])
        for label in unique:
            mask=(inputs[:,j]==label if label is not None
                  else torch.ones(n,dtype=torch.bool))
            size=int(mask.sum())
            if size<2:
                singletons+=1
                continue
            left=a[mask,j,:];right=b[mask,j,:]
            left=left-left.mean(dim=0)
            right=right-right.mean(dim=0)
            output+=left.T@right
            used_groups+=1;weighted_total+=size
    return output.tolist(),{
        'condition':condition,'calibration_puzzles':n,'positions':p,
        'usable_centered_groups':used_groups,
        'singleton_groups_ignored':singletons,
        'calibration_tokens_in_usable_groups':weighted_total,
    }


def conditional_pair_shuffle(B,inputs,seed=202613133):
    """Null preserves public group and each feature's within-group count."""
    if B.ndim!=3 or inputs.shape!=B.shape[:2]:
        raise ValueError('paired calibration shape mismatch')
    rng=random.Random(seed)
    out=B.clone()
    for pos in range(B.shape[1]):
        for label in torch.unique(inputs[:,pos]).tolist():
            group=torch.where(inputs[:,pos]==label)[0].tolist()
            perm=group[:]
            rng.shuffle(perm)
            out[group,pos]=B[perm,pos]
    # Check every group/feature sum is EXACTLY preserved by the null.
    for pos in range(B.shape[1]):
        for label in torch.unique(inputs[:,pos]).tolist():
            mask=inputs[:,pos]==label
            if not torch.equal(out[mask,pos,:].sum(0),B[mask,pos,:].sum(0)):
                raise AssertionError('null changed observable-conditioned marginals')
    return out


def score_conditional_match(A,B,inputs,test_tensors,seed=173):
    conditions=('position','position_digit')
    output={}
    for condition in conditions:
        affinity,groups=conditional_affinity(A,B,inputs,condition)
        alignment=hungarian_maximum(affinity)
        baseline=conditional_pair_shuffle(B,inputs,seed+len(condition))
        null_affinity,_=conditional_affinity(A,baseline,inputs,condition)
        null_alignment=hungarian_maximum(null_affinity)
        tests={}
        for name,(X,Y) in test_tensors.items():
            measured=heldout_overlap(X,Y,alignment)
            null=heldout_overlap(X,Y,null_alignment)
            stats=puzzle_bootstrap(measured,draws=2000,seed=seed+len(name))
            tests[name]={
                'heldout_overlap':stats,
                'within_group_shuffle_calibration_null_overlap':sum(null)/len(null),
                'matched_minus_within_group_null':stats['mean']-sum(null)/len(null),
                'random_feature_null':null_permutations(X,Y,32,seed+len(name)+41)['mean'],
            }
        output[condition]={
            'calibration_group_counts':groups,
            'permutation_sha256':hashlib.sha256(json.dumps(alignment).encode()).hexdigest(),
            'within_group_shuffled_assignment_sha256':hashlib.sha256(json.dumps(null_alignment).encode()).hexdigest(),
            'test_families':tests
        }
    return output


def run(source_checkpoint=None,allow_fetch=False,
        original_artifact=None,independent_artifact=None):
    if original_artifact is None or independent_artifact is None:
        raise ValueError('strict original and independent model artifact paths required')
    torch.set_num_threads(2)
    original,original_saes,old_provenance=load_frozen_sources(extracted_dir=original_artifact)
    source,_=source_model_from_lock(allow_fetch,source_checkpoint)
    newreceipt,rosters,new=load_independent_models(independent_artifact)
    comps={'pretrainedA_71':(source,*original_saes[71])}
    for teacher_seed,entry in new.items():
        for sae_seed,params in entry['saes'].items():
            comps[f'newB_{teacher_seed}_SAE_{sae_seed}']=(entry['teacher'],*params)
    pairs=[('pretrainedA_71',name) for name in comps if name.startswith('newB_')]
    for teacher_seed in new:
        pairs.append((f'newB_{teacher_seed}_SAE_71',f'newB_{teacher_seed}_SAE_89'))
    pairs.append(('newB_501_SAE_71','newB_701_SAE_71'))
    keys=['feature_calibration']+list(FAMILIES)
    supports={name:{key:binary_sae_support(*params,rosters[key])
                    for key in keys}
              for name,params in comps.items()}
    observed_inputs,_=tensors(rosters['feature_calibration'])
    results=[]
    for index,(a,b) in enumerate(pairs):
        source_support=supports[a]['feature_calibration']
        target_support=supports[b]['feature_calibration']
        test={key:(supports[a][key],supports[b][key]) for key in FAMILIES}
        stats=score_conditional_match(source_support,target_support,
                                      observed_inputs,test,seed=711+index)
        results.append({'source':a,'target':b,'conditional_protocol':stats})
        print('CONDITIONAL_TRANSFER',a,b,
              {cond:{family:round(data['heldout_overlap']['mean'],5)
                     for family,data in v['test_families'].items()}
               for cond,v in stats.items()},flush=True)
    return {
        'schema':'SMC-13C/observable-clue-conditioned-cross-model-feature-transfer-v1',
        'source_original_teacher_sha256':newreceipt['source_checkpoint_sha256'],
        'source_new_teachers_sha256':{str(k):v['source_sha256'] for k,v in new.items()},
        'new_teacher_training_receipt_sha256':hashlib.sha256((Path(independent_artifact)/'full.json').read_bytes()).hexdigest(),
        'source_original_saes':old_provenance,
        'n_calibration_puzzles':len(rosters['feature_calibration']),
        'n_puzzles_per_test_family':64,
        'conditioning':'token position; or token position PLUS observed digit 0..9, with 0 meaning blank',
        'assignment':'maximum sum of within-public-group centered source/target feature covariance, calibration only',
        'null':'independent pairing of target SAE activation vectors within identical position and visible clue digit; preserves every group marginal and avoids using hidden solution labels',
        'results':results,
        'scientific_scope':'Observational coactivation signal beyond position/observed-symbol effects; no source-model causal semantics, equal checkpoint capability, or independent feature-factor identification guaranteed.'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--original-sae-artifact',required=True,type=Path)
    p.add_argument('--independent-artifact',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('runs/smc13/conditioned.json'))
    a=p.parse_args()
    receipt=run(a.source_checkpoint,a.fetch_source,
                a.original_sae_artifact,a.independent_artifact)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(a.out),'comparisons':len(receipt['results']),
                      'feature_transfer':[
                         {'pair':r['source']+' -> '+r['target'],
                          'novel_30_conditioned':{
                              k:round(v['test_families']['test_novel_30']['heldout_overlap']['mean'],6)
                              for k,v in r['conditional_protocol'].items()}}
                         for r in receipt['results']]},indent=2))


if __name__=='__main__':
    main()
