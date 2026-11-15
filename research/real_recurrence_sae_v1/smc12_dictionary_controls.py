"""SMC-12B: exact dictionary-wide trained vs initialization cone controls.

For each of the three source-frozen SAE fit seeds, recompute every 96-atom
self-selection cone under the original training-only physical scaling.
Use the independently re-created initial TopKSAE whose decoder SHA256 is
pinned to the original SMC-07 training receipt. Check EVERY cone score
normal with exact Fractions (no floating threshold classifications).

A dictionary cone result is CONDITIONAL: if neuron t is strictly highest
positive at some starting hidden state, then its decoder ray will keep
that t identity for all positive amplitudes iff the cone slopes satisfy
the criterion. Merely owning an interior cone does not prove a neuron
ever activates, nor that its feature is semantically interpretable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from smc07_real_sae import TopKSAE,tensor_hash
from smc11_selection import frozen_active_direction_dyadics
from smc12_cones import load_frozen_sources
from conic_self_selection_exact import physical_encoder_normals,cone_certificate,feedback_matrix_self_selection

EXPECTED_INITIAL_DECODER_SHA={
 71:'cf8cbb36f0a430ef4c1023f2de2100c9269a263040137628b57d5195a62dac92',
 89:'37adb786b738bf5682868f3b223520249796b555c09854b65927a64e977c33f8',
113:'f3d7977da5a632207409ee0fe2c10da94798c9c6db44516fa72de3c54edf5be8'
}


def exact_dictionary_cones(sae,scale):
    """Reconstruct all 96 normalized decoder rays using frozen source rules."""
    if sae.encoder.weight.shape!=(96,48) or sae.decoder.weight.shape!=(48,96):
        raise ValueError('wrong original dictionary dimensions')
    normals=physical_encoder_normals(
       sae.encoder.weight.detach().cpu().tolist(),
       scale.detach().cpu().tolist())
    counts={'interior':0,'boundary':0,'outside':0,'unconstrained':0}
    allowed=[]
    min_strict_squared=[]
    for i in range(96):
        direction=frozen_active_direction_dyadics(sae,scale,i)
        cert=cone_certificate(normals,(i,),direction)
        counts[cert['status']]+=1
        if cert['all_positive_ray_amplitudes_preserve_selected_set']:
            allowed.append(i)
        if cert['status']=='interior':
            min_strict_squared.append(float(cert['minimal_normalized_slope_squared']))
    if sum(counts.values())!=96:
        raise AssertionError('not every atom classified')
    feedback=feedback_matrix_self_selection(
        sae.encoder.weight.detach().cpu().tolist(),
        sae.decoder.weight.detach().cpu().tolist())
    feedback_interior=[row['atom'] for row in feedback if row['strict_interior']]
    actual_interior=sorted(allowed)
    mismatch=sorted(set(feedback_interior)^set(actual_interior))
    return {'counts':counts,'infinite_self_selection_atom_ids':allowed,
            'strict_cone_interior_count':counts['interior'],
            'all_source_atom_count':96,
            'minimum_interior_q2_approx':min(min_strict_squared,default=None),
            'ideal_WD_strict_diagonal_max_count':len(feedback_interior),
            'rounded_causal_ray_vs_ideal_WD_mismatch_atom_ids':mismatch}


def initialization_control(seed,scale):
    if seed not in EXPECTED_INITIAL_DECODER_SHA:
        raise ValueError('unregistered original training seed')
    torch.manual_seed(seed)
    initial=TopKSAE(48,96,4).eval()
    digest=tensor_hash(initial.decoder.weight)
    if digest!=EXPECTED_INITIAL_DECODER_SHA[seed]:
        raise ValueError('initial decoder does not match original SMC-07 training receipt')
    return initial,digest


def run(archive=None,extracted_dir=None):
    report,snapshots,provenance=load_frozen_sources(archive,extracted_dir)
    selected={}
    for row in report['exact_goal_selection_cases']:
        selected.setdefault(row['seed'],set()).add(row['top1_feature'])
    results=[]
    for seed in (71,89,113):
        sae,mean,scale=snapshots[seed]
        original,sha=initialization_control(seed,scale)
        trained=exact_dictionary_cones(sae,scale)
        random=exact_dictionary_cones(original,scale)
        if not selected[seed].issubset(set(trained['infinite_self_selection_atom_ids'])):
            raise AssertionError('frozen selected goal feature lacks self-selection cone')
        if trained['rounded_causal_ray_vs_ideal_WD_mismatch_atom_ids'] or random['rounded_causal_ray_vs_ideal_WD_mismatch_atom_ids']:
            raise AssertionError('ideal W D cross-talk and frozen rounded source rays disagree')
        results.append({
           'seed':seed,
           'original_initial_decoder_sha256':sha,
           'source_frozen_selected_unique_top_features':len(selected[seed]),
           'trained_exact':trained,
           'original_random_initialization_exact':random,
           'same_96_atom_names_paired':True,
           'same_frozen_source_train_scale_for_both':True,
        })
        print('SMC12_DICTIONARY_CONTROL',seed,
              {'trained':trained['counts'],
               'initialization':random['counts'],
               'selected_distinct':len(selected[seed])},flush=True)
    trained=sum(x['trained_exact']['strict_cone_interior_count'] for x in results)
    initialization=sum(x['original_random_initialization_exact']['strict_cone_interior_count']
                       for x in results)
    return {
       'schema':'SMC-12B/exact-288-dictionary-atom-self-selection-controls-v1',
       'source_commit':report['source_commit'],
       'checkpoint_sha256':report['checkpoint_sha256'],
       'roster_sha256':report['roster_sha256'],
       'provenance':provenance,
       'seeds':[71,89,113],
       'trained_96_rail_self_selection_cones':trained,
       'random_initial_96_rail_self_selection_cones':initialization,
       'total_atoms_each_arm':288,
       'initialization_control':'same original fit seeds, TopKSAE 48->96->48 PyTorch 2.7.1, original SMC-07 SHA256 init decoder weights, exact same frozen training scales',
       'limitations':'Finite matched trained/initialization dictionary comparison, not a hypothesis test across independently trained teachers, proof of SAE semantic factors, or causal interpretation. The cone property is conditional on strict source identity when a feature activates.',
       'per_seed':results,
    }


def main():
    p=argparse.ArgumentParser()
    i=p.add_mutually_exclusive_group(required=True)
    i.add_argument('--zip',type=Path)
    i.add_argument('--artifact-dir',type=Path)
    p.add_argument('--out',type=Path,default=Path('runs/smc12/dictionary-control.json'))
    a=p.parse_args()
    out=run(a.zip,a.artifact_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
       'out':str(a.out),'trained_self_selection':out['trained_96_rail_self_selection_cones'],
       'random_initial_self_selection':out['random_initial_96_rail_self_selection_cones'],
       'total_per_arm':out['total_atoms_each_arm'],
    },indent=2))


if __name__=='__main__':
    main()
