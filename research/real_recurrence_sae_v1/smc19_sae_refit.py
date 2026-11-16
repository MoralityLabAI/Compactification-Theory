"""SMC-19: orthogonal gauge transport and independent SAE refit causal stability.

Phase freeze:
  load SHA-pinned original models and seven conditioned assignments;
  refit four top-four SAEs on original SMC13 train/validation only;
  save/verify weights and choose matched/negative-control assignments plus
  global causal sign on the original 64 calibration puzzles; save frozen.json.
Phase evaluate:
  load and SHA-check only that locked artifact; before any new test states,
  rebuild original SMC13 calibration source; create fresh SMC19 rosters;
  evaluate a tautological orthogonal gauge transport and four refit causal
  panels, retaining all failures with 64-puzzle clustered uncertainty.

Frozen checkpoints and data have the same scientific provenance as SMC13-18.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from smc13_family import new_full_solution,dig_unique,valid_solution,rectangle_count,count_solutions,fingerprint
from smc13_train_teachers import tensors,source_sae_fingerprint,sae_fit,persist_sae
from smc13_transfer import hungarian_maximum,heldout_overlap
from smc13_conditioned_transfer import conditional_affinity,conditional_pair_shuffle
from smc15_active_necessity import evaluate_pair_split,orient,score_rows,GOALS
from smc17_causal_inactive import source_models

PRECOMMIT='fd96967fbf251c884a8a705d6d8a68c0c2cddd2e'
FAMILIES={'smc19_novel_30':(202619030000,30),
          'smc19_novel_34':(202619040000,34)}
PAIRS=(
  ('newB_501_SAE_71','newB_501_SAE_131'),
  ('newB_501_SAE_71','newB_501_SAE_167'),
  ('newB_701_SAE_71','newB_701_SAE_131'),
  ('newB_701_SAE_71','newB_701_SAE_167'),
)
SEEDS=(131,167)
N_ORIGINAL=64
N_TEST=64
BOOTS=2000
ROTATION_SEED=20261919
ENGINEERING_TOL=2e-9


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_mapping(mapping):
    return digest(json.dumps(mapping).encode())


def fresh_rosters(old,limit=N_TEST):
    """Recreate previous full roster family and exclude every previous puzzle."""
    if limit not in (4,64):
        raise ValueError('SMC19 only four-puzzle pilot or sixty-four-puzzle full')
    from smc18_task_gradient import fresh_rosters as prev18
    history,_=prev18(old,limit=64)
    seen_y={tuple(r['target']) for rows in list(old.values())+list(history.values()) for r in rows}
    seen_x={tuple(r['x']) for rows in list(old.values())+list(history.values()) for r in rows}
    out={}
    for name,(base,holes) in FAMILIES.items():
        rows=[]
        for ix in range(limit):
            seed=base+ix
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError('noncanonical Sudoku source generator changed')
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError('fresh Sudoku not exact-unique or has wrong holes')
            if any(d and d!=y[j] for j,d in enumerate(x)):
                raise ValueError('visible clue and full solution disagree')
            key_x,key_y=tuple(x),tuple(y)
            if key_x in seen_x or key_y in seen_y:
                raise ValueError('SMC19 cohort duplicates earlier train/calibration/test')
            seen_x.add(key_x);seen_y.add(key_y)
            rows.append({'seed':seed,'x':x,'target':y,
                         'holes':holes,'rectangles':rectangle_count(y)})
        out[name]=rows
    return out,{name:fingerprint(rows) for name,rows in out.items()}


def exact_assignments(a,b,inputs,index):
    """Conditioned calibration-only Hungarian matching and hard nulls."""
    aff,_=conditional_affinity(a,b,inputs,'position_digit')
    match=hungarian_maximum(aff)
    shuffled=conditional_pair_shuffle(b,inputs,202619710+index)
    shuffled_aff,_=conditional_affinity(a,shuffled,inputs,'position_digit')
    clue=hungarian_maximum(shuffled_aff)
    random_ids=list(range(96))
    random.Random(202619810+index).shuffle(random_ids)
    for key,assignment in (('matched',match),('clue_shuffled',clue),
                           ('random',random_ids)):
        if sorted(assignment)!=list(range(96)):
            raise ValueError(key+' calibration assignment not 96-feature bijection')
    return ({'matched':match,'clue_shuffled':clue,'random':random_ids},
            {k+'_sha256':sha_mapping(v) for k,v in
             (('matched',match),('clue_shuffled',clue),('random',random_ids))})


def calibrate_signs(rows):
    a=torch.tensor([x['source_drop'] for x in rows],dtype=torch.float64)
    b=torch.tensor([x['matched_drop'] for x in rows],dtype=torch.float64)
    c=torch.tensor([x['other_active_drop'] for x in rows],dtype=torch.float64)
    mask=torch.tensor([x['eligible_mask'] for x in rows],dtype=torch.bool)
    return orient(a,b,c,mask)


def load_original(original,independent,conditioned,allow_fetch,source_checkpoint):
    torch.set_num_threads(2)
    return source_models(original,independent,conditioned,allow_fetch,source_checkpoint)


def freeze(original_artifact,independent_artifact,conditioned_receipt,
           root,allow_fetch=False,source_checkpoint=None):
    """Train and seal four SAE snapshots before constructing SMC19 test states."""
    from smc14_causal_transfer import source_group_context
    teachers,comps,old,assignments,locks=load_original(
        original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch,source_checkpoint)
    root=Path(root)
    (root/'saes').mkdir(parents=True,exist_ok=True)
    original_hashes={}
    for teacher_seed in (501,701):
        for sae_seed in (71,89):
            label=f'newB_{teacher_seed}_SAE_{sae_seed}'
            _,sae,mean,scale=comps[label]
            original_hashes[label]=source_sae_fingerprint(sae,mean,scale)
    fit_receipts=[]
    # All fit inputs derive solely from SMC13 TRAIN and VALIDATION, not SMC19.
    train_x,_=tensors(old['teacher_train'])
    valid_x,_=tensors(old['teacher_val'])
    for teacher_seed in (501,701):
        teacher=teachers[f'B{teacher_seed}']
        with torch.no_grad():
            train_h=teacher.reason_states(train_x)[1].detach()
            valid_h=teacher.reason_states(valid_x)[1].detach()
        for seed in SEEDS:
            sae,mu,scale,info=sae_fit(train_h,valid_h,seed)
            label=f'newB_{teacher_seed}_SAE_{seed}'
            fingerprint_new=source_sae_fingerprint(sae,mu,scale)
            if fingerprint_new in original_hashes.values() or any(
                    x['sae_fp32_fingerprint']==fingerprint_new for x in fit_receipts):
                raise ValueError('independently refit SAE matches an existing trained fingerprint')
            snapshot=persist_sae(sae,mu,scale,teacher_seed,seed,root/'saes')
            if snapshot['sae_fingerprint']!=fingerprint_new:
                raise ValueError('new refit model failed source SAE fingerprint')
            fits={'teacher_seed':teacher_seed,'sae_seed':seed,'component':label,
                  'best_validation_step':info['best_step'],
                  'validation_normalized_mse':info['best_val_mse'],
                  'optimizer_steps':info['steps'],
                  'initial_decoder_sha256':info['initial_decoder_sha256'],
                  'snapshot_relative_path':f'saes/teacher{teacher_seed}_sae{seed}.pt',
                  'snapshot_file_sha256':snapshot['file_sha256'],
                  'sae_fp32_fingerprint':fingerprint_new}
            fit_receipts.append(fits)
            comps[label]=(f'B{teacher_seed}',sae,mu,scale)
            print('SMC19_REFIT_FROZEN',label,fit_receipts[-1],flush=True)
    # Now construct only ORIGINAL calibration states, matched and sign fit.
    calib_ctx,support=source_group_context(
        teachers,comps,old,('feature_calibration',))
    pcal=old['feature_calibration']
    input_cal=calib_ctx['B501']['cohorts']['feature_calibration']['inputs']
    panels=[]
    for i,(ref,refit) in enumerate(PAIRS):
        A=support[ref]['feature_calibration']['support']
        B=support[refit]['feature_calibration']['support']
        mappings,shas=exact_assignments(A,B,input_cal,i)
        raw,coverage=evaluate_pair_split(calib_ctx,comps,ref,refit,
                  mappings['matched'],'feature_calibration',pcal,i)
        signs=calibrate_signs(raw)
        panels.append({
          'source':ref,'target':refit,
          'mapping':mappings,'mapping_hashes':shas,
          'calibration':{'puzzles':N_ORIGINAL,'raw':raw,'coverage':coverage,
                         'orientation':signs,'cohort':'feature_calibration'},
        })
        print('SMC19_MAPPING_AND_CALIBRATION_SIGN_FROZEN',
              ref,refit,shas,signs,flush=True)
    frozen={
       'schema':'SMC-19/four-source-trained-refit-SAE-calibration-lock-v1',
       'pre_registration_commit':PRECOMMIT,
       'source_original_receipts':locks,
       'source_original_seven_mappings':assignments,
       'source_old_SAE_fingerprints':original_hashes,
       'train_roster_sha256':fingerprint(old['teacher_train']),
       'validation_roster_sha256':fingerprint(old['teacher_val']),
       'calibration_roster_sha256':fingerprint(old['feature_calibration']),
       'refit_roster':fit_receipts,'pair_calibration':panels,
       'torch_version':torch.__version__,
       'training':{'steps':300,'new_seeds':list(SEEDS),'width':96,'k':4,
                   'optimizer':'original SMC07 fit_sae AdamW lr=0.003, weight_decay=1e-5',
                   'training_states':'original SMC13 teacher_train',
                   'selection_states':'original SMC13 teacher_val'},
       'calibration_only_before_new_test':True,
    }
    (root/'frozen.json').write_text(json.dumps(frozen,sort_keys=True,indent=2)+'\n')
    print('SMC19_ALL_FOUR_REFITS_AND_CALIBRATIONS_FROZEN',
          {'frozen_json_sha256':digest((root/'frozen.json').read_bytes()),
           'frozen_dir':str(root),'refits':len(fit_receipts)},flush=True)
    return frozen


def load_frozen(root,comps,original,old):
    """Reload SHA-checked original weights-only fit checkpoints."""
    from smc07_real_sae import TopKSAE
    root=Path(root)
    data=(root/'frozen.json').read_bytes()
    receipt=json.loads(data)
    if (receipt['schema']!='SMC-19/four-source-trained-refit-SAE-calibration-lock-v1'
        or receipt['pre_registration_commit']!=PRECOMMIT or
        not receipt['calibration_only_before_new_test'] or
        tuple(receipt['training']['new_seeds'])!=SEEDS or
        receipt['training']['steps']!=300):
        raise ValueError('SMC19 original frozen refit schema/protocol differs')
    if (receipt['train_roster_sha256']!=fingerprint(old['teacher_train']) or
        receipt['validation_roster_sha256']!=fingerprint(old['teacher_val']) or
        receipt['calibration_roster_sha256']!=fingerprint(old['feature_calibration'])):
        raise ValueError('SMC19 model selection/calibration roster mismatch')
    if receipt['source_original_receipts']!=original:
        raise ValueError('SMC19 previous original artifact SHA locks differ')
    if len(receipt['refit_roster'])!=4 or len(receipt['pair_calibration'])!=4:
        raise ValueError('SMC19 refit/pair roster incomplete')
    for seed in (501,701):
        for old_seed in (71,89):
            label=f'newB_{seed}_SAE_{old_seed}'
            _,s,m,sc=comps[label]
            if source_sae_fingerprint(s,m,sc)!=receipt['source_old_SAE_fingerprints'][label]:
                raise ValueError('previous SAE snapshot changed: '+label)
    for row in receipt['refit_roster']:
        teacher_seed,seed=row['teacher_seed'],row['sae_seed']
        if teacher_seed not in (501,701) or seed not in SEEDS:
            raise ValueError('trained SAE refit seed differs')
        label=f'newB_{teacher_seed}_SAE_{seed}'
        if row['component']!=label or row['optimizer_steps']!=300:
            raise ValueError('refit seed/step audit failed')
        path=root/row['snapshot_relative_path']
        if path!=root/'saes'/f'teacher{teacher_seed}_sae{seed}.pt':
            raise ValueError('unsafe or changed refit weight path')
        if digest(path.read_bytes())!=row['snapshot_file_sha256']:
            raise ValueError('SAE refit weight file hash differs')
        z=torch.load(path,map_location='cpu',weights_only=True)
        if (z['teacher_seed']!=teacher_seed or z['sae_seed']!=seed or
            z['fingerprint']!=row['sae_fp32_fingerprint']):
            raise ValueError('new source SAE snapshot metadata changed')
        sae=TopKSAE(48,96,4).eval()
        sae.load_state_dict(z['state_dict'],strict=True)
        mu,sc=z['train_mean'],z['train_scale']
        if source_sae_fingerprint(sae,mu,sc)!=row['sae_fp32_fingerprint']:
            raise ValueError('SAE refit tensor-byte fingerprint mismatch')
        comps[label]=(f'B{teacher_seed}',sae,mu,sc)
    for i,(ref,refit) in enumerate(PAIRS):
        panel=receipt['pair_calibration'][i]
        if (panel['source'],panel['target'])!=(ref,refit):
            raise ValueError('pair sequence differs')
        for label,assignment in panel['mapping'].items():
            if sorted(assignment)!=list(range(96)) or sha_mapping(assignment)!=panel['mapping_hashes'][label+'_sha256']:
                raise ValueError('refit Hungarian correspondence SHA mismatch')
        if len(panel['calibration']['raw'])!=N_ORIGINAL:
            raise ValueError('original calibration sign source incomplete')
    return receipt,digest(data)


def rotation_matrix():
    gen=torch.Generator().manual_seed(ROTATION_SEED)
    M=torch.randn(48,48,generator=gen,dtype=torch.float64)
    Q,R=torch.linalg.qr(M)
    signs=torch.sign(torch.diagonal(R))
    signs[signs==0]=1.
    return Q*signs[None,:]


def transport_control(teacher,sae,mean,scale,stage,pos):
    """Same model and SAE under a dense orthogonal coordinate wrapper.

    Transported model is F_Q(v)=F(vQ^T); feature decoder vectors use d'=dQ.
    This is tautological gauge equivalence and checks implementation only.
    """
    if stage.ndim!=3 or stage.shape[1:]!=(81,48) or pos.shape[1]!=GOALS:
        raise ValueError('gauge control expects source-locked goal-token states')
    Q=rotation_matrix()
    I=torch.eye(48,dtype=torch.float64)
    ortho=float((Q.T@Q-I).abs().max())
    sae64=copy.deepcopy(sae).double().eval()
    mu=mean.double();sc=scale.double()
    h=stage[:4].double()
    z=sae64.encode((h-mu)/sc)
    z_transport=sae64.encode(((h@Q)@Q.T-mu)/sc)
    feature_error=float((z_transport-z).abs().max())
    def continue_logits(states):
        for _ in range(3):
            states=teacher.block(states)
        return teacher.head(states)
    with torch.no_grad():
        orig=continue_logits(h)
        transformed=continue_logits((h@Q)@Q.T)
        clean_error=float((orig-transformed).abs().max())
        decoder=(sae64.decoder.weight.detach()*sc[:,None]).T
        max_edited=0.
        for j in range(min(4,len(h))):
            for g in pos[j].tolist():
                atom=int(z[j,g].argmax())
                delta=z[j,g,atom]*decoder[atom]
                x=h[j:j+1].clone();x[0,g]-=delta
                xr=(h[j:j+1]@Q).clone();xr[0,g]-=delta@Q
                one=continue_logits(x)
                two=continue_logits(xr@Q.T)
                max_edited=max(max_edited,float((one-two).abs().max()))
    metrics={'rotation_seed':ROTATION_SEED,'orthogonality_max_error':ortho,
             'feature_code_max_abs_error':feature_error,
             'clean_logits_max_abs_error':clean_error,
             'native_erase_logits_max_abs_error':max_edited,
             'tested_puzzles':min(4,len(h)),'tested_goal_tokens':min(4,len(h))*GOALS,
             'gauge_type':'exact joint orthogonal row-hidden transport, not an independently retrained model'}
    if any(v>=ENGINEERING_TOL for v in (ortho,feature_error,clean_error,max_edited)):
        raise ArithmeticError('orthogonal gauge transport failed the preregistered numerical bound')
    return metrics


def evaluate(original_artifact,independent_artifact,conditioned_receipt,
             frozen_dir,allow_fetch=False,source_checkpoint=None,
             pilot=0,pair_limit=4):
    from smc14_causal_transfer import source_group_context
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError('only predeclared SMC19 pilot/full sizes')
    teachers,comps,old,original,locks=load_original(
        original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch,source_checkpoint)
    frozen,frozen_sha=load_frozen(frozen_dir,comps,locks,old)
    if frozen['source_original_seven_mappings']!=original:
        raise ValueError('original SMC13C assignments differed from frozen refit stage')
    print('SMC19_ALL_REFITS_AND_CALIBRATIONS_RELOADED_AND_HASHED',
          {'frozen_json_sha256':frozen_sha,
           'pair_mapping_sha256':[x['mapping_hashes'] for x in frozen['pair_calibration']]},flush=True)
    # Heldout states are constructed for the FIRST time only now.
    test,manifests=fresh_rosters(old,limit=pilot or N_TEST)
    ctx,support=source_group_context(teachers,comps,test,tuple(FAMILIES))
    base=ctx['B501']['cohorts']['smc19_novel_30']
    gauge=transport_control(ctx['B501']['model64'],comps['newB_501_SAE_71'][1],
                             comps['newB_501_SAE_71'][2],
                             comps['newB_501_SAE_71'][3],
                             base['stage1'],base['positions'])
    panels=[]
    for i,(ref,refit) in enumerate(PAIRS[:pair_limit]):
        cal=frozen['pair_calibration'][i]
        heldout={}
        for family in FAMILIES:
            roster=test[family]
            raw,coverage=evaluate_pair_split(ctx,comps,ref,refit,
                   cal['mapping']['matched'],family,roster,i)
            stats=score_rows(raw,cal['calibration']['orientation'],
                             seed=202619130+i*100+len(family),draws=BOOTS)
            old_support=support[ref][family]['support']
            new_support=support[refit][family]['support']
            overlaps={}
            for arm,assignment in cal['mapping'].items():
                values=heldout_overlap(old_support,new_support,assignment)
                overlaps[arm]={'mean':sum(values)/len(values),
                               'per_puzzle':values}
            heldout[family]={'per_puzzle':raw,'coverage':coverage,
                             'statistics':stats,'activation_overlap':overlaps}
        success=bool(not pilot and pair_limit==4 and all(
                     heldout[f]['statistics']['registered_success'] for f in FAMILIES))
        panels.append({'source':ref,'target':refit,'frozen_mapping_hashes':cal['mapping_hashes'],
                       'calibration_orientation':cal['calibration']['orientation'],
                       'heldout':heldout,'registered_success_both_families':success})
        print('SMC19_SAE_REFIT_CAUSAL',ref,refit,{
            f:{'eligible':heldout[f]['statistics']['eligible_goals'],
               'puzzles':heldout[f]['statistics']['nonempty_puzzle_clusters'],
               'effect':heldout[f]['statistics'].get('matched_minus_other_active'),
               'gate':heldout[f]['statistics']['registered_success']}
            for f in FAMILIES},flush=True)
    return {
       'schema':'SMC-19/orthogonal-gauge-and-independent-SAE-refit-causal-v1',
       'pre_registration_commit':PRECOMMIT,
       'source_original_receipts':locks,
       'frozen_refit_json_sha256':frozen_sha,
       'frozen_calibration_source':frozen,
       'source_original_seven_assignments_hash_verified':True,
       'trained_models_and_SAEs_unchanged':True,
       'fresh_rosters':test,'fresh_manifest_sha256':manifests,
       'test_seed_bases':{f:v[0] for f,v in FAMILIES.items()},
       'puzzles_per_family':pilot or N_TEST,
       'full_protocol':not bool(pilot or pair_limit<4),
       'calibration_frozen_before_new_test':True,
       'orthogonal_gauge_transport_control':gauge,
       'pairs':panels,
       'scientific_limits':'Exact gauge transport is a tautological software control; separately fitted SAEs may differ in task-functional feature identities even with identical teacher weights. Puzzle clustered bootstrap is descriptive.'
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=('freeze','evaluate'),required=True)
    ap.add_argument('--original-artifact',type=Path,required=True)
    ap.add_argument('--independent-artifact',type=Path,required=True)
    ap.add_argument('--conditioned-receipt',type=Path,required=True)
    ap.add_argument('--fetch-source',action='store_true')
    ap.add_argument('--source-checkpoint',type=Path)
    ap.add_argument('--frozen-dir',type=Path,default=Path('runs/smc19/frozen'))
    ap.add_argument('--pilot-puzzles',type=int,choices=(4,))
    ap.add_argument('--pair-limit',type=int,choices=(1,2,3,4),default=4)
    ap.add_argument('--out',type=Path,default=Path('runs/smc19/full.json'))
    a=ap.parse_args()
    torch.set_num_threads(2)
    if a.phase=='freeze':
        result=freeze(a.original_artifact,a.independent_artifact,
                      a.conditioned_receipt,a.frozen_dir,
                      a.fetch_source,a.source_checkpoint)
        print(json.dumps({'frozen_dir':str(a.frozen_dir),
          'new_SAE_fingerprints':[f['sae_fp32_fingerprint'] for f in result['refit_roster']]},indent=2))
    else:
        result=evaluate(a.original_artifact,a.independent_artifact,
                        a.conditioned_receipt,a.frozen_dir,
                        a.fetch_source,a.source_checkpoint,
                        a.pilot_puzzles or 0,a.pair_limit)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        print(json.dumps({'out':str(a.out),'full_protocol':result['full_protocol'],
          'gauge':result['orthogonal_gauge_transport_control'],
          'pairs':[{'target':x['target'],'registered_gate':x['registered_success_both_families']}
                   for x in result['pairs']]},indent=2))


if __name__=='__main__':
    main()
