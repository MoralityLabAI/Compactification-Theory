"""SMC-13B: calibration-only feature matching across independent teachers.

Input artifacts are original source-locked pretrained teacher + SAE and the
two newly trained source-locked teachers with their fresh SAE snapshots.
All frozen model SHA256s and generator manifests are checked.

A centered activation covariance matrix is matched by a deterministic
96x96 Hungarian assignment on a strictly independent 64-puzzle calibration
cohort. The assignment is frozen before three 64-puzzle heldout cohorts
are scored (novel 30-hole, novel 34-hole, canonical 30-hole).

Controls: uniformly random feature permutations, puzzle-shuffled
calibration fit, and same-teacher independent SAE seed alignment.
This tests transfer of feature activation identities, not semantic meaning.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
for p in (HERE,PRE):sys.path.insert(0,str(p))
from smc13_family import build_rosters
from smc13_train_teachers import (
    tensors,source_model_from_lock,source_sae_fingerprint,
    stable_hash_tensors,TEACHER_SEEDS,SAE_SEEDS,FAMILIES
)
from smc12_cones import load_frozen_sources
from smc07_real_sae import TopKSAE
from upstream_arch import TinyRecursiveModel

EXPECTED_NEW_TRAINING_RECEIPT_SHA256='1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3'


def hungarian_maximum(scores):
    """Deterministic O(m³) maximum-weight perfect matching of square matrix."""
    n=len(scores)
    if n==0 or any(len(row)!=n for row in scores):
        raise ValueError('nonempty square affinity matrix required')
    best=max(float(v) for row in scores for v in row)
    # Standard primal-dual shortest augmenting path implementation.
    cost=[[best-float(v) for v in row] for row in scores]
    u=[0.0]*(n+1);v=[0.0]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i
        minval=[float('inf')]*(n+1);used=[False]*(n+1)
        j0=0
        while True:
            used[j0]=True
            i0=p[j0];delta=float('inf');j1=0
            for j in range(1,n+1):
                if used[j]:continue
                cur=cost[i0-1][j-1]-u[i0]-v[j]
                if cur<minval[j]:
                    minval[j]=cur;way[j]=j0
                if minval[j]<delta-1e-12 or (abs(minval[j]-delta)<=1e-12 and j<j1):
                    delta=minval[j];j1=j
            if not delta<float('inf'):
                raise ArithmeticError('invalid assignment matrix')
            for j in range(n+1):
                if used[j]:
                    u[p[j]]+=delta;v[j]-=delta
                else:minval[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    assigned=[-1]*n
    for j in range(1,n+1):
        if p[j]>0:assigned[p[j]-1]=j-1
    if sorted(assigned)!=list(range(n)):
        raise AssertionError('matching is not a permutation')
    return assigned


def binary_sae_support(teacher,sae,mean,scale,rows):
    x,_=tensors(rows)
    teacher.eval();sae.eval()
    with torch.no_grad():
        stage=teacher.reason_states(x)[1]
        encoded=sae.encode((stage-mean)/scale)
        output=(encoded>0).to(torch.float32)
    if output.shape!=(len(rows),81,96):
        raise AssertionError('expected 81x96 feature support per puzzle')
    return output


def center_affinity(X,Y):
    """Covariance of paired feature activity, preserving calibration only."""
    if X.shape!=Y.shape or X.ndim!=3 or X.shape[2]!=96:
        raise ValueError('expected equal 64x81x96 feature tensors')
    a=X.reshape(-1,96).double()
    b=Y.reshape(-1,96).double()
    ac=a-a.mean(0);bc=b-b.mean(0)
    return (ac.T@bc).tolist()


def heldout_overlap(X,Y,assignment):
    """Cluster by puzzle rather than treating 81 tokens as independent."""
    if X.shape!=Y.shape or X.ndim!=3:
        raise ValueError('mismatched heldout feature supports')
    n=len(assignment)
    if sorted(assignment)!=list(range(n)) or X.shape[-1]!=n:
        raise ValueError('invalid source-feature permutation')
    with torch.no_grad():
        intersection=((X>0)&(Y[:,:,assignment]>0)).sum((1,2)).float()
        count=(X>0).sum((1,2)).float()
        if bool((count<=0).any()):
            raise ValueError('source SAE has a puzzle with zero positive features')
        values=(intersection/count).tolist()
    return values


def puzzle_bootstrap(values,draws=2000,seed=20261300):
    if not values:raise ValueError('empty heldout sample')
    rng=random.Random(seed);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return {'mean':sum(values)/n,'puzzle_bootstrap95':[means[int(.025*draws)],means[int(.975*draws)-1]],
            'n_puzzle_clusters':n}


def null_permutations(X,Y,draws=32,seed=119):
    rng=random.Random(seed);n=X.shape[-1]
    means=[]
    for _ in range(draws):
        a=list(range(n));rng.shuffle(a)
        means.append(sum(heldout_overlap(X,Y,a))/len(X))
    return {'draws':draws,'mean':sum(means)/draws,
            'minimum':min(means),'maximum':max(means)}


def load_independent_models(root):
    root=Path(root)
    source_receipt=(root/'full.json').read_bytes()
    if hashlib.sha256(source_receipt).hexdigest()!=EXPECTED_NEW_TRAINING_RECEIPT_SHA256:
        raise ValueError('independent teacher full training receipt SHA256 mismatch')
    report=json.loads(source_receipt)
    if (report['schema']!='SMC-13/independent-trained-teacher-novel-Sudoku-v1'
        or report['source_checkpoint_sha256']!='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
        or tuple(report['teacher_seeds'])!=TEACHER_SEEDS
        or tuple(report['sae_seeds'])!=SAE_SEEDS
        or report['teacher_training_steps']!=600
        or report['sae_training_steps']!=300):
        raise ValueError('unexpected independent model provenance')
    rosters,manifest=build_rosters()
    if manifest!=report['novel_family']:
        raise ValueError('source-locked novel split differs from trained receipt')
    output={}
    for row in report['new_teachers']:
        seed=row['teacher_seed']
        filename=root/'teachers'/f'teacher_seed{seed}.pt'
        data=filename.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['checkpoint']['file_sha256']:
            raise ValueError('different independently trained model checkpoint')
        model_dict=torch.load(filename,weights_only=True,map_location='cpu')
        if model_dict['train_roster_sha256']!=manifest['cohorts']['teacher_train']['sha256']:
            raise ValueError('teacher train roster mismatch')
        teacher=TinyRecursiveModel(48,4).eval()
        teacher.load_state_dict(model_dict['state_dict'],strict=True)
        if stable_hash_tensors(teacher.state_dict())!=row['checkpoint']['model_parameter_sha256']:
            raise ValueError('trained recurrent teacher tensor fingerprint differs')
        saes={}
        for a in row['new_teacher_sae_fits']:
            ss=a['seed']
            saved=root/'saes'/f'teacher{seed}_sae{ss}.pt'
            if hashlib.sha256(saved.read_bytes()).hexdigest()!=a['snapshot']['file_sha256']:
                raise ValueError('frozen SAE saved-file hash mismatch')
            d=torch.load(saved,weights_only=True,map_location='cpu')
            sae=TopKSAE(48,96,4).eval()
            sae.load_state_dict(d['state_dict'],strict=True)
            mean,scale=d['train_mean'],d['train_scale']
            if source_sae_fingerprint(sae,mean,scale)!=a['snapshot']['sae_fingerprint']:
                raise ValueError('trained SAE binary tensor fingerprint mismatch')
            saes[ss]=(sae,mean,scale)
        output[seed]={'teacher':teacher,'saes':saes,'source_sha256':row['checkpoint']['model_parameter_sha256']}
    return report,rosters,output


def evaluate_alignment(A,B,calib,puzzles,seed=13):
    Xa,Xb=calib
    learned=hungarian_maximum(center_affinity(Xa,Xb))
    rng=random.Random(202613000+seed)
    shuffled=list(range(len(Xb)));rng.shuffle(shuffled)
    shuffled_Y=Xb[shuffled]
    shuffled_alignment=hungarian_maximum(center_affinity(Xa,shuffled_Y))
    metrics={}
    for name,(X,Y) in puzzles.items():
        observed=heldout_overlap(X,Y,learned)
        anti=heldout_overlap(X,Y,shuffled_alignment)
        null=null_permutations(X,Y,seed=seed+len(name))
        stats=puzzle_bootstrap(observed,seed=seed+int(len(name)*13))
        metrics[name]={
          'calibrated_overlap':stats,
          'shuffled_calibration_alignment_mean_overlap':sum(anti)/len(anti),
          'random_feature_permutation_control':null,
          'calibrated_minus_random_mean':stats['mean']-null['mean'],
          'calibrated_minus_shuffled_puzzle_calibration':stats['mean']-sum(anti)/len(anti),
        }
    return {'permutation_feature_count':len(learned),
            'assignment_sha256':hashlib.sha256(json.dumps(learned).encode()).hexdigest(),
            'matched_on':'64 noncanonical calibration puzzles, never evaluation cohort',
            'per_family':metrics}


def run(source_checkpoint=None,allow_fetch=False,
        original_sae_artifact=None,new_model_artifact=None):
    torch.set_num_threads(2)
    if original_sae_artifact is None or new_model_artifact is None:
        raise ValueError('both source-pinned model artifact directories required')
    original,fit,metadata=load_frozen_sources(extracted_dir=original_sae_artifact)
    source,lock=source_model_from_lock(allow_fetch,source_checkpoint)
    report,rosters,new=load_independent_models(new_model_artifact)
    components={
      'pretrainedA_71':(source,*fit[71])
    }
    for seed,v in new.items():
        for sae_seed,fit_tuple in v['saes'].items():
            components[f'newB_{seed}_SAE_{sae_seed}']=(v['teacher'],*fit_tuple)
    chosen_pairs=[('pretrainedA_71',label) for label in components if label.startswith('newB_')]
    for seed in new:
        chosen_pairs.append((f'newB_{seed}_SAE_71',f'newB_{seed}_SAE_89'))
    chosen_pairs.append(('newB_501_SAE_71','newB_701_SAE_71'))
    rows=['feature_calibration']+list(FAMILIES)
    supports={}
    for model_name,(teacher,sae,mean,scale) in components.items():
        supports[model_name]={name:binary_sae_support(teacher,sae,mean,scale,rosters[name])
                              for name in rows}
    analyses=[]
    for pair_index,(a,b) in enumerate(chosen_pairs):
        calibration=(supports[a]['feature_calibration'],supports[b]['feature_calibration'])
        cases={name:(supports[a][name],supports[b][name]) for name in FAMILIES}
        stats=evaluate_alignment(a,b,calibration,cases,seed=207+pair_index)
        analyses.append({'source':a,'target':b,'both_models_source_locked':True,**stats})
        print('INDEPENDENT_FEATURE_ALIGNMENT',a,b,
              {k:round(v['calibrated_overlap']['mean'],4)
               for k,v in stats['per_family'].items()},flush=True)
    return {
       'schema':'SMC-13B/calibration-only-cross-recurrent-feature-transfer-v1',
       'source_teacher_sha256':'3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1',
       'new_model_manifest_sha256':hashlib.sha256((Path(new_model_artifact)/'full.json').read_bytes()).hexdigest(),
       'novel_roster_sha256':{key:value['sha256'] for key,value in report['novel_family']['cohorts'].items()},
       'n_independent_calibration_puzzles':64,
       'n_independent_puzzles_per_holdout_family':64,
       'pair_count':len(analyses),
       'source_original_frozen_sae_provenance':metadata,
       'new_teachers_source_sha256':{str(seed):v['source_sha256'] for seed,v in new.items()},
       'protocol':'max weight one-to-one matching on centered positive-feature coactivation, calibration only; test on separate novel-30, novel-34 and canonical-30 puzzle families, compare shuffled-calibration and random-feature permutations',
       'scope':'statistical feature coactivation transfer, not exact semantic feature identification or model performance gain; models independent, source architecture matched',
       'comparisons':analyses,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--original-sae-artifact',type=Path,required=True)
    p.add_argument('--new-model-artifact',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc13/alignment.json'))
    a=p.parse_args()
    output=run(a.source_checkpoint,a.fetch_source,a.original_sae_artifact,a.new_model_artifact)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,sort_keys=True,indent=2)+'
')
    print(json.dumps({'out':str(a.out),'pair_count':output['pair_count'],
                      'comparisons':[{'pair':r['source']+' -> '+r['target'],
                                      'test_mean':{k:round(v['calibrated_overlap']['mean'],5)
                                                   for k,v in r['per_family'].items()}}
                                     for r in output['comparisons']]},indent=2))


if __name__=='__main__':
    main()

