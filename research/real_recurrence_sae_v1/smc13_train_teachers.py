"""SMC-13: independently train two 44K recurrent teachers on noncanonical Sudoku.

Frozen *before* touching new heldout puzzles:
- Source teacher A: the third-party checkpoint pinned in LSV-05.
- Teachers B: same architecture freshly initialized with seeds 501/701,
  trained from scratch on 192 structurally novel 30-hole Sudoku puzzles.
- Teacher selection: validation missing-cell cross-entropy only, from
  checkpoints after 25,50,...,600 optimizer steps.
- SAE fits: two fresh top4 48->96->48 fits per teacher B, seeds 71/89,
  with 300 steps and validation-only checkpoint selection.
- Controls: matching initialized SAE and matched-dimension IID Gaussian SAE.
- Evaluation: disjoint noncanonical 30-hole/34-hole and canonical orbit
  30-hole test cohorts, 64 puzzles each; holdout never selects parameters.

The one-shot run saves independently trained teacher checkpoints with exact
tensor-byte hashes. A separate script performs calibrated cross-model
activation alignment without touching the teacher/SAE fitting phases.
"""
import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
for p in (HERE,PRE):sys.path.insert(0,str(p))
from upstream_arch import TinyRecursiveModel
from run_checkpoint import locked_bytes,load_pretrained
from sudoku_generator import is_valid_solution
from smc07_real_sae import fit_sae,TopKSAE
from smc13_family import build_rosters,fingerprint,rectangle_count
from conic_self_selection_exact import feedback_matrix_self_selection
from smc12_cones import load_frozen_sources,sae_hash

SOURCE_CHECKPOINT='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
SOURCE_ROSTER='eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611'
TEACHER_SEEDS=(501,701)
SAE_SEEDS=(71,89)
TEACHER_STEPS=600
SAE_STEPS=300
MODEL_SHAPE=44001
FAMILIES=('test_novel_30','test_novel_34','test_canonical_30')


def stable_hash_tensors(tensors):
    h=hashlib.sha256()
    for name in sorted(tensors):
        t=tensors[name].detach().cpu().to(torch.float32).contiguous()
        vals=t.reshape(-1).tolist()
        h.update(name.encode('utf-8')+b'\x00')
        h.update(str(tuple(t.shape)).encode('ascii')+b'\x00')
        h.update(struct.pack('<'+'f'*len(vals),*vals))
    return h.hexdigest()


def tensors(rows):
    x=torch.tensor([row['x'] for row in rows],dtype=torch.long)
    y=torch.tensor([row['target'] for row in rows],dtype=torch.long)
    return x,y


def missing_ce(logits,target,inputs):
    masks=inputs==0
    ce=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1),reduction='none').reshape(-1,81)
    return ce[masks].mean()


def teacher_metrics(teacher,rows):
    x,y=tensors(rows)
    teacher.eval()
    with torch.no_grad():
        logits=teacher(x)
        pred=logits.argmax(-1)
        m=x==0
        hole_match=(pred[m]==y[m])
        known_match=(pred[~m]==y[~m])
        full=(pred==y).all(-1)
        invalid=sum(not is_valid_solution(row.tolist()) for row in pred)
    return {
        'n_puzzles':len(rows),'holes_per_case':int(m[0].sum()),
        'missing_cell_correct_count':int(hole_match.sum()),
        'missing_cell_total':int(m.sum()),
        'missing_cell_accuracy':float(hole_match.float().mean()),
        'known_cell_accuracy':float(known_match.float().mean()),
        'full_grid_correct':int(full.sum()),
        'valid_sudoku_predicted_count':len(rows)-invalid,
        'missing_cell_cross_entropy':float(missing_ce(logits,y,x)),
    }


def source_model_from_lock(allow_fetch=False,checkpoint=None):
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,sha=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    if sha!=SOURCE_CHECKPOINT:raise ValueError('source teacher SHA256 differs')
    return load_pretrained(lock,raw).eval(),lock


def train_teacher(train,val,seed,steps=TEACHER_STEPS):
    if not isinstance(steps,int) or steps<25 or steps%25:
        raise ValueError('training length must be a positive multiple of 25')
    torch.manual_seed(seed)
    model=TinyRecursiveModel(48,4)
    if sum(p.numel() for p in model.parameters())!=MODEL_SHAPE:
        raise ValueError('expected 44,001 parameter source-matched architecture')
    init=stable_hash_tensors(model.state_dict())
    x,y=tensors(train)
    xv,yv=tensors(val)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    g=torch.Generator().manual_seed(123456+seed)
    history=[];best=float('inf');best_state=None;best_step=None
    for step in range(1,steps+1):
        model.train()
        ids=torch.randint(len(x),(16,),generator=g)
        logits=model(x[ids])
        targets=y[ids]
        ce=F.cross_entropy(logits.reshape(-1,10),targets.reshape(-1),
                           reduction='none').reshape(-1,81)
        weights=1+2*(x[ids]==0).float()
        loss=(ce*weights).sum()/weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%25==0:
            model.eval()
            with torch.no_grad():
                val_logits=model(xv)
                score=float(missing_ce(val_logits,yv,xv))
            if not math.isfinite(score):
                raise ArithmeticError('invalid teacher validation CE')
            history.append({'step':step,'train_weighted_loss':float(loss),
                            'validation_missing_ce':score})
            if score<best:
                best=score;best_step=step
                best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state,strict=True)
    model.eval()
    output={
        'teacher_seed':seed,'optimizer_steps':steps,'selected_step':best_step,
        'validation_only_selected_missing_ce':best,
        'teacher_initial_fp32_sha256':init,
        'teacher_final_fp32_sha256':stable_hash_tensors(model.state_dict()),
        'training_trajectory':history,
    }
    if output['teacher_initial_fp32_sha256']==output['teacher_final_fp32_sha256']:
        raise AssertionError('new teacher weights did not train')
    return model,output


def save_teacher(teacher,details,roster_sha,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':'SMC-13/independently-trained-teacher-v1',
             'seed':details['teacher_seed'],
             'source_architecture':'TinyRecursiveModel(48,4)',
             'train_roster_sha256':roster_sha,
             'state_dict':{k:v.detach().cpu().clone()
                           for k,v in teacher.state_dict().items()}}
    torch.save(payload,path)
    decoded=torch.load(path,map_location='cpu',weights_only=True)
    if decoded['seed']!=payload['seed'] or decoded['train_roster_sha256']!=roster_sha:
        raise AssertionError('teacher checkpoint roundtrip metadata mismatch')
    clone=TinyRecursiveModel(48,4)
    clone.load_state_dict(decoded['state_dict'],strict=True)
    if stable_hash_tensors(clone.state_dict())!=details['teacher_final_fp32_sha256']:
        raise AssertionError('teacher checkpoint state roundtrip differs')
    return {'checkpoint_path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'model_parameter_sha256':details['teacher_final_fp32_sha256']}


def sae_cone_count(sae):
    W=sae.encoder.weight.detach().cpu().tolist()
    D=sae.decoder.weight.detach().cpu().tolist()
    c=feedback_matrix_self_selection(W,D)
    return {
       'strict_interior':sum(x['strict_interior'] for x in c),
       'n_atoms':len(c),
       'self_reinforcing':sum(x['positive_ray_self_selection'] for x in c),
       'selected_feature_ids':[x['atom'] for x in c if x['strict_interior']],
       'predicate':'exact WD column diagonal dominance, independent of positive source state normalization'
    }


def sae_fit(states_train,states_val,seed):
    sae,mu,std,fit=fit_sae(states_train,states_val,seed=seed,steps=SAE_STEPS)
    return sae,mu,std,fit


def controls_gaussian(shape_train,shape_val):
    rng=torch.Generator().manual_seed(20261313)
    return torch.randn(shape_train,generator=rng),torch.randn(shape_val,generator=rng)


def persist_sae(sae,mu,std,teacher_seed,sae_seed,root):
    from smc11_joint_causal import frozen_sae_fingerprint
    sha=frozen_sae_fingerprint(sae,mu,std)
    path=Path(root)/f'teacher{teacher_seed}_sae{sae_seed}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'schema':'SMC-13/fitted-SAE-v1',
                'teacher_seed':teacher_seed,'sae_seed':sae_seed,
                'state_dict':{k:v.detach().cpu().clone() for k,v in sae.state_dict().items()},
                'train_mean':mu.detach().cpu().clone(),
                'train_scale':std.detach().cpu().clone(),
                'fingerprint':sha},path)
    z=torch.load(path,weights_only=True,map_location='cpu')
    other=TopKSAE(48,96,4);other.load_state_dict(z['state_dict'],strict=True)
    if frozen_sae_fingerprint(other,z['train_mean'],z['train_scale'])!=sha:
        raise AssertionError('fitted SAE checkpoint does not roundtrip')
    return {'path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'sae_fingerprint':sha}


def run(allow_fetch=False,checkpoint=None,original_sae_artifact=None,
        teacher_seeds=TEACHER_SEEDS,sae_seeds=SAE_SEEDS,
        teacher_steps=TEACHER_STEPS,quick=False,output_dir=Path('runs/smc13')):
    torch.set_num_threads(2)
    rosters,manifest=build_rosters(quick=quick)
    source,source_lock=source_model_from_lock(allow_fetch,checkpoint)
    A={}
    for name in FAMILIES:
        A[name]=teacher_metrics(source,rosters[name])
    if not quick and original_sae_artifact is None:
        raise ValueError('full run requires the exact original SMC11C SAE artifact for alignment')
    source_sae=None
    if original_sae_artifact is not None:
        prior,original,prior_provenance=load_frozen_sources(extracted_dir=original_sae_artifact)
        source_sae=original[71]
    run_results=[]
    gaussian_result={}
    for seed in teacher_seeds:
        teacher,fit=train_teacher(rosters['teacher_train'],rosters['teacher_val'],
                                  seed,teacher_steps)
        tmeta=save_teacher(teacher,fit,manifest['cohorts']['teacher_train']['sha256'],
                           Path(output_dir)/'teachers'/f'teacher_seed{seed}.pt')
        accuracy={k:teacher_metrics(teacher,rosters[k]) for k in FAMILIES}
        with torch.no_grad():
            xtrain,_=tensors(rosters['teacher_train'])
            xval,_=tensors(rosters['teacher_val'])
            htrain=teacher.reason_states(xtrain)[1].detach()
            hval=teacher.reason_states(xval)[1].detach()
        sae_data=[]
        for ss in sae_seeds:
            sae,mu,std,info=sae_fit(htrain,hval,ss)
            result={'seed':ss,'best_step':info['best_step'],
                    'validation_normalized_mse':info['best_val_mse'],
                    'cone':sae_cone_count(sae),
                    'snapshot':persist_sae(sae,mu,std,seed,ss,
                                            Path(output_dir)/'saes')}
            sae_data.append(result)
        if not gaussian_result:
            gauss_train,gauss_val=controls_gaussian(htrain.shape,hval.shape)
            for ss in sae_seeds:
                gsae,gmu,gstd,gfit=sae_fit(gauss_train,gauss_val,ss)
                gaussian_result[str(ss)]={
                    'seed':ss,'cone':sae_cone_count(gsae),
                    'best_step':gfit['best_step'],
                    'validation_normalized_mse':gfit['best_val_mse'],
                    'source':'IID standard normal with same 192x81x48 and 64x81x48 tensor dimensions'
                }
        run_results.append({
           'teacher_seed':seed,'teacher_training':fit,'checkpoint':tmeta,
           'source_family_test_accuracy':accuracy,'new_teacher_sae_fits':sae_data
        })
        print('INDEPENDENT_TEACHER',seed,
              {'selected_step':fit['selected_step'],
               'accuracy':{k:round(v['missing_cell_accuracy'],4)
                           for k,v in accuracy.items()},
               'trained_sae_cones':[s['cone']['strict_interior'] for s in sae_data],
               'gaussian_cones':{k:g['cone']['strict_interior']
                                 for k,g in gaussian_result.items()}},flush=True)
    if len({r['checkpoint']['model_parameter_sha256'] for r in run_results})!=len(run_results):
        raise AssertionError('independent teacher seeds produced identical weights')
    report={
      'schema':'SMC-13/independent-trained-teacher-novel-Sudoku-v1',
      'source_checkpoint_sha256':SOURCE_CHECKPOINT,
      'source_teacher_commit':source_lock['revision'],
      'new_teacher_architecture':'44,001-parameter 4-step recurrent Sudoku network, newly initialized independent from original pretrained weights',
      'teacher_training_rule':'weighted all-cell cross entropy with weight3 for missing and weight1 for given; validation missing-cell CE selects among steps25..600 only',
      'teacher_seeds':list(teacher_seeds),'sae_seeds':list(sae_seeds),
      'teacher_training_steps':teacher_steps,'sae_training_steps':SAE_STEPS,
      'novel_family':manifest,
      'original_pretrained_teacher_accuracy':A,
      'original_frozen_sae_provenance':prior_provenance if source_sae else None,
      'new_teachers':run_results,
      'gaussian_held_shape_control':gaussian_result,
      'claim_scope':'Independent new source-family model checkpoints and template-disjoint Sudoku targets; cone counts are internal autoencoder properties. Test labels never select teacher or SAE weights.',
      'next_step':'Calibration-only source-A/new-B SAE feature alignment and heldout cross-checkpoint transfer'
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-source',action='store_true')
    p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--original-sae-artifact',type=Path)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--teacher-seeds',default='501,701')
    p.add_argument('--sae-seeds',default='71,89')
    p.add_argument('--teacher-steps',type=int,default=600)
    p.add_argument('--out',type=Path,default=Path('runs/smc13/full.json'))
    a=p.parse_args()
    result=run(a.fetch_source,a.source_checkpoint,a.original_sae_artifact,
               tuple(int(x) for x in a.teacher_seeds.split(',')),
               tuple(int(x) for x in a.sae_seeds.split(',')),
               a.teacher_steps,a.quick,a.out.parent)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(a.out),
                      'manifest':result['novel_family'],
                      'original_teacher':{k:v['missing_cell_accuracy'] for k,v in result['original_pretrained_teacher_accuracy'].items()},
                      'independent_teachers':[
                       {'seed':r['teacher_seed'],
                        'missing_cell_accuracy':{k:v['missing_cell_accuracy'] for k,v in r['source_family_test_accuracy'].items()},
                        'sae_strict_cones':[x['cone']['strict_interior'] for x in r['new_teacher_sae_fits']]}
                       for r in result['new_teachers']],
                      'gaussian_control':{k:v['cone']['strict_interior'] for k,v in result['gaussian_held_shape_control'].items()}},indent=2))


if __name__=='__main__':
    main()
