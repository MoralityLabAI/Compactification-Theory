"""SMC-12: EXACT angular/decoder self-selection cones of frozen SMC-11C SAEs.

Reads a previously validated, source-locked SMC-11C artifact containing all
three weights-only SAE fits and 48 exact baseline goal-token selection cases.
It does not retrain a source SAE, mutate the teacher, or access new test data.

The 96 affine inequality normals and source decoder direction are converted
to exact Fractions. The exact squared cone margin, dyadic lower root bound,
unit-chord and relative raw decoder guards, and a SHARP rational adversarial
hyperplane witness are checked for every selected top neuron.

The physical ray follows the precise SMC-09 convention: elementwise FP32
decoder*train_scale, then FP64 normalization, then exact-real Arb normalization
(the last positive scalar changes neither the ray nor the cone margin).
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile
from fractions import Fraction as Q

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc07_real_sae import TopKSAE
from smc11_selection import frozen_active_direction_dyadics
from conic_self_selection_exact import (
    physical_encoder_normals, cone_certificate, relative_raw_guard,
    unit_chord_guard, joint_normal_and_decoder_guard,
    dyadic_sqrt_bracket, sharp_boundary_adversary
)

EXPECTED_ZIP='1b62914ba7fecd91614eab36c1f799e2fc0d6d4e3279d0e1ac03a0ee5ee2a2fb'
EXPECTED_RECEIPT='7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087'
EXPECTED_TEACHER='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
EXPECTED_ROSTER='eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611'
EXPECTED_FITS={
 71:('00542304e1f15f606a20519d59c7617d574a4f0266c9a33b2220cf146015c3da',
     'b9992439de96c12f1d721a0c265ecc8031640ea7f72a4cd869a8c8c2bc9cdf3a'),
 89:('ba0bfd9d82d68bd3398bb37e8b7539d93dff6d4e21f153d44bfa9f1ccae2ebcb',
     '2f91f686b1dc9b82c46d2ef3f6e1a183570b07730ccfac07f4552b0b4066ac42'),
113:('087787bcfdd4d4350b56595049c07a7fd62bfd7db69f52258905a872e1f5b5af',
     '96914494b8670a369bf54fbdea0607c03e688cb8b13c108e40e6851c95e2a6b1')
}
RAW_BUDGETS=('0.1','0.15','0.2','0.21','0.215','0.22','0.25','0.3','0.4','0.5')
CHORD_BUDGETS=('0.1','0.2','0.21','0.218','0.22','0.25','0.3','0.4','0.5','0.55','0.59')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_frozen_sources(archive=None,extracted_dir=None):
    """Pin the exact original CI archive and independently hash all members."""
    if bool(archive)==bool(extracted_dir):
        raise ValueError('supply precisely one archive ZIP or extracted directory')
    if archive is not None:
        raw=Path(archive).read_bytes()
        if sha256(raw)!=EXPECTED_ZIP:
            raise ValueError('source ZIP SHA256 mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            blob={name:z.read(name) for name in z.namelist()}
        file_provenance={'archive_zip_sha256':sha256(raw)}
    else:
        root=Path(extracted_dir)
        blob={}
        for path in ('joint-full.json',)+tuple(
            f'joint-frozen-weights/sae_seed{s}.pt' for s in EXPECTED_FITS):
            blob[path]=(root/path).read_bytes()
        file_provenance={'archive_zip_sha256':EXPECTED_ZIP,
                         'downloaded_original_archive_sha_verified_elsewhere':True}
    receipt=blob['joint-full.json']
    if sha256(receipt)!=EXPECTED_RECEIPT:
        raise ValueError('original 480-case causal receipt SHA256 mismatch')
    data=json.loads(receipt)
    if (data['schema']!='SMC-11C/same-fit-SAE-identity-and-Arb-local-causal-v1'
        or data['checkpoint_sha256']!=EXPECTED_TEACHER
        or data['roster_sha256']!=EXPECTED_ROSTER
        or data['seeds']!=[71,89,113]
        or data['sae_steps']!=300
        or len(data['exact_goal_selection_cases'])!=48
        or len(data['cases'])!=480):
        raise ValueError('unexpected source-trained model or roster')
    snapshots={}
    seen=set()
    for entry in data['frozen_sae_snapshots']:
        seed=entry['seed']
        if seed in seen or seed not in EXPECTED_FITS:
            raise ValueError('repeated/unexpected SAE seed')
        seen.add(seed)
        filename=f'joint-frozen-weights/sae_seed{seed}.pt'
        raw=blob[filename]
        file_sha,fitted_sha=EXPECTED_FITS[seed]
        if sha256(raw)!=file_sha or entry['snapshot_file_sha256']!=file_sha:
            raise ValueError('frozen SAE model file SHA256 mismatch')
        if entry['trained_sae_sha256']!=fitted_sha:
            raise ValueError('trained SAE tensor provenance mismatch')
        params=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
        if params['seed']!=seed or params['teacher_sha256']!=EXPECTED_TEACHER:
            raise ValueError('serialized SAE source identity differs')
        model=TopKSAE(48,96,4).eval()
        model.load_state_dict(params['state_dict'],strict=True)
        if (model.encoder.weight.dtype!=torch.float32 or
            params['train_scale'].dtype!=torch.float32 or
            torch.any(params['train_scale']<=0)):
            raise ValueError('unexpected source FP32/normalization tensor')
        if sae_hash(model,params['train_mean'],params['train_scale'])!=fitted_sha:
            raise ValueError('original fitted SAE fingerprint differs')
        snapshots[seed]=(model,params['train_mean'],params['train_scale'])
    if set(snapshots)!=set(EXPECTED_FITS):
        raise ValueError('missing source-trained SAE checkpoint')
    return data,snapshots,{
       **file_provenance,'joint_json_sha256':EXPECTED_RECEIPT,
       'per_seed_weights_sha256':{str(k):EXPECTED_FITS[k][0] for k in sorted(snapshots)}
    }


def sae_hash(sae,mean,scale):
    """Original SMC-11C canonical tensor-byte fingerprint, independently."""
    vals={**{'sae.'+k:v for k,v in sae.state_dict().items()},
          'train_mean':mean,'train_scale':scale}
    h=hashlib.sha256()
    for name in sorted(vals):
        t=vals[name]
        if t.dtype!=torch.float32:
            raise ValueError('fingerprint expects exactly stored FP32 tensors')
        h.update(name.encode('utf-8')+b'\x00')
        h.update(str(tuple(t.shape)).encode('ascii')+b'\x00')
        raw=t.detach().cpu().contiguous().reshape(-1).tolist()
        h.update(struct.pack('<'+'f'*len(raw),*raw))
    return h.hexdigest()


def analyze_frozen_cones(report,snapshots,provenance):
    registered={(r['seed'],r['puzzle_id']):r
                for r in report['exact_goal_selection_cases']}
    expected={(s,p) for s in (71,89,113) for p in range(3,64,4)}
    if set(registered)!=expected:
        raise ValueError('incomplete 16-puzzle/3-seed source roster')
    outputs=[]
    for seed in (71,89,113):
        sae,_,scale=snapshots[seed]
        normals=physical_encoder_normals(
            sae.encoder.weight.detach().cpu().tolist(),scale.tolist())
        for pid in range(3,64,4):
            source=registered[(seed,pid)]
            if source['status']!='certified' or not source['torch_same_top1_and_top4']:
                raise ValueError('original source goal feature selection not strict')
            top=source['top1_feature']
            if top not in source['top4_features']:
                raise ValueError('recorded active set contradicts top neuron')
            ray=frozen_active_direction_dyadics(sae,scale,top)
            cert=cone_certificate(normals,(top,),ray)
            group=cone_certificate(normals,source['top4_features'],ray)
            if (cert['status']!='interior' or
                not source['rays']['positive']['top1']['infinite_forward_radius']):
                raise ValueError('original positive all-radius top1 not certified')
            if (group['all_positive_ray_amplitudes_preserve_selected_set'] !=
                source['rays']['positive']['topk']['infinite_forward_radius']):
                raise ValueError('top4 cone conflicts with original directional source')
            if cert['nontrivial_constraints']!=96:
                raise ValueError('unexpected source model cone constraints')
            q2=cert['minimal_normalized_slope_squared']
            if q2 is None or not q2>0:
                raise ArithmeticError('source cone lacks positive angular interior')
            low,hi=dyadic_sqrt_bracket(q2,64)
            witness=sharp_boundary_adversary(normals,(top,),ray,cert)
            if (witness['relative_raw_boundary_distance_squared']!=q2
                or witness['boundary_gap_slope']!=0
                or not witness['outside_gap_slope']<0):
                raise ArithmeticError('sharp adversarial raw decoder cone witness failed')
            q_float=float(q2)
            angle=math.asin(math.sqrt(q_float))
            chord=math.sqrt(2*q_float/(1+math.sqrt(max(0,1-q_float))))
            outputs.append({
               'seed':seed,'puzzle_id':pid,'goal_token':source['goal_token'],
               'selected_top_feature':top,'selected_top4_features':source['top4_features'],
               'ray_cone_status':cert['status'],
               'strict_positive_cone_slopes':cert['nontrivial_constraints'],
               'min_normalized_slope_squared_exact':str(q2),
               'limiting_constraint':cert['limiting_constraint'],
               'raw_relative_distance_dyadic_lower':str(low),
               'raw_relative_distance_dyadic_upper':str(hi),
               'raw_relative_distance_approx':math.sqrt(q_float),
               'unit_direction_chord_distance_approx':chord,
               'angle_rad_approx':angle,
               'angle_deg_approx':angle*180/math.pi,
               'raw_decoder_strict_guards':{
                  b:relative_raw_guard(cert,Q(b)) for b in RAW_BUDGETS},
               'unit_chord_strict_guards':{
                  b:unit_chord_guard(cert,Q(b)) for b in CHORD_BUDGETS},
               'joint_decoder_10pct_and_constraint_normal_10pct':
                   joint_normal_and_decoder_guard(cert,Q(1,10),Q(1,10)),
               'full_top4_support_infinite_positive_ray':
                   group['all_positive_ray_amplitudes_preserve_selected_set'],
               'sharp_boundary_witness':{
                   'relative_squared_margin_matches_exact':True,
                   'at_boundary_score_slope_exact':str(witness['boundary_gap_slope']),
                   'overshoot_score_slope_strictly_negative':witness['outside_gap_slope']<0,
                   'overshoot_fraction':str(Q(1,100)),
               }
            })
    if len(outputs)!=48:
        raise ValueError('wrong source goal count')
    radii=[x['raw_relative_distance_approx'] for x in outputs]
    angles=[x['angle_deg_approx'] for x in outputs]
    chord=[x['unit_direction_chord_distance_approx'] for x in outputs]
    min_by_q=min(outputs,key=lambda x:Q(x['min_normalized_slope_squared_exact']))
    summary={
      'total_source_goal_cases':48,
      'exact_certified_strict_interior_cones':48,
      'all_96_score_constraint_slopes_strictly_positive_each':True,
      'minimum_squared_normalized_slope_exact':min_by_q['min_normalized_slope_squared_exact'],
      'minimizing_case':{'seed':min_by_q['seed'],
                         'puzzle_id':min_by_q['puzzle_id'],
                         'limiting_constraint':min_by_q['limiting_constraint']},
      'relative_raw_decoder_margin':{
        'minimum_approx':min(radii),'median_approx':statistics.median(radii),
        'maximum_approx':max(radii)},
      'unit_direction_chord_margin':{
        'minimum_approx':min(chord),'median_approx':statistics.median(chord),
        'maximum_approx':max(chord)},
      'angular_margin_degrees':{
        'minimum_approx':min(angles),'median_approx':statistics.median(angles),
        'maximum_approx':max(angles)},
      'raw_relative_budget_certified_counts':{
         b:sum(x['raw_decoder_strict_guards'][b] for x in outputs)
         for b in RAW_BUDGETS},
      'unit_chord_budget_certified_counts':{
         b:sum(x['unit_chord_strict_guards'][b] for x in outputs)
         for b in CHORD_BUDGETS},
      'joint_10pct_decoder_10pct_constraint_normal_cone_count':
           sum(x['joint_decoder_10pct_and_constraint_normal_10pct'] for x in outputs),
      'entire_top4_support_infinite_positive_ray_count':
           sum(x['full_top4_support_infinite_positive_ray'] for x in outputs),
      'exact_boundary_adversary_verified_count':len(outputs),
    }
    return {
       'schema':'SMC-12/verified-frozen-sae-self-selection-angular-cone-v1',
       'source_commit':report['source_commit'],
       'checkpoint_sha256':report['checkpoint_sha256'],
       'roster_sha256':report['roster_sha256'],
       'original_same_fit_snapshot_provenance':provenance,
       'sae_seeds':sorted(snapshots),
       'source_train_updates':300,
       'source_test_puzzle_ids':list(range(3,64,4)),
       'source_model':'externally pretrained 44,001-parameter TRM-inspired Sudoku recurrent model',
       'math_note':'MathCan OLM-12, sharp rational angular/raw decoder cone theorem',
       'interpretation':'Every certified interior cone ray keeps its original highest positive SAE neuron for arbitrary nonnegative intervention length, for fixed encoder. Downstream causal output ranking remains at its previous source-locked finite radii/budgets.',
       'scopes_excluded':'No change to model weights, re-fitted encoder, arbitrary puzzle population, remote tokens, or machine-execution roundoff is certified.',
       'summary':summary,'per_case':outputs,
    }


def run(archive=None,extracted_dir=None):
    report,snapshots,provenance=load_frozen_sources(archive,extracted_dir)
    return analyze_frozen_cones(report,snapshots,provenance)


def main():
    p=argparse.ArgumentParser()
    inp=p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--zip',type=Path)
    inp.add_argument('--artifact-dir',type=Path)
    p.add_argument('--out',type=Path,default=Path('runs/smc12/cones.json'))
    args=p.parse_args()
    out=run(args.zip,args.artifact_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'out':str(args.out),
        'checkpoint_sha256':out['checkpoint_sha256'],
        'source_zip_sha256':out['original_same_fit_snapshot_provenance']['archive_zip_sha256'],
        'summary':out['summary'],
    },indent=2,sort_keys=True))


if __name__=='__main__':
    main()
