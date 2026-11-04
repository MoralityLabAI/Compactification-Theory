"""LSV-07: pre-damage finite-horizon interval certificate, pinned third-party TRM.

The interval operator is a conditional FP64 implementation of mathematically
valid interval formulas. Source checksum and roster are fixed by LSV-05.
No damaged trajectory is used to form any predictive certificate.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import torch

from predictive_interval import (Interval, invariant_box_check,
                                 predictive_goal_region)
from run_checkpoint import locked_bytes, load_pretrained, prepare_roster

HERE=Path(__file__).resolve().parent
AMPS=('0','0.00000001','0.000001','0.0001','0.02','0.10','0.50')
MODES=('all_coordinates','first_missing_cell')


def single_cell_mask(inputs:torch.Tensor)->torch.Tensor:
    if inputs.ndim!=2 or inputs.shape[1]!=81:
        raise ValueError('expected Sudoku puzzle [N,81]')
    missing=inputs.eq(0)
    if not bool(missing.any(-1).all()):
        raise ValueError('every puzzle must have a missing cell')
    first=missing.to(torch.int64).argmax(-1)
    mask=torch.zeros((len(inputs),81,48),dtype=torch.bool)
    mask[torch.arange(len(inputs)),first,:]=True
    return mask


def verify_predicate(classes:torch.Tensor,reference:torch.Tensor,labels:torch.Tensor)->dict:
    certain=classes>=0
    matches=(classes==reference)
    if not bool(matches[certain].all()):
        raise AssertionError('interval predictor conflicts with FP64 undamaged execution')
    correct=(classes==labels)
    full_certified=certain.all(-1)
    full_correct=correct.all(-1)
    return {
      'n_cells':classes.numel(),
      'certified_stable_cells':int(certain.sum()),
      'certified_stable_fraction':round(certain.double().mean().item(),8),
      'certified_correct_cells':int((certain & correct).sum()),
      'certified_stable_but_wrong_cells':int((certain & ~correct).sum()),
      'fully_certified_puzzles':int(full_certified.sum()),
      'fully_certified_correct_puzzles':int((full_certified & full_correct).sum())}


def observed_attacks(model, z1:torch.Tensor, clean_labels:torch.Tensor,
                     classes:torch.Tensor, eps:float, mask:torch.Tensor|None):
    tests=[]
    # Distinct noise patterns; none participates in computing the boxes.
    for seed in (34,89):
        rng=torch.Generator().manual_seed(seed)
        noise=(torch.rand(z1.shape,dtype=torch.float64,generator=rng)*2-1)*eps
        if mask is not None:noise=noise*mask.to(torch.float64)
        perturbed=z1+noise
        with torch.no_grad():
            for _ in range(3):perturbed=model.block(perturbed)
            pred=model.head(perturbed).argmax(-1)
        cert=classes>=0
        violations=(cert & (classes!=pred))
        if bool(violations.any()):
            raise AssertionError('conditional interval certificate contradicted by FP64 sampled execution')
        tests.append({'seed':seed,'changed_cells':int((pred!=clean_labels).sum()),
                      'certified_violations':int(violations.sum()),
                      'max_initial_damage':round(float(noise.abs().amax()),9)})
    return tests


def run(lock_path:Path,checkpoint_path:Path|None=None,allow_fetch=False)->dict:
    lock=json.loads(lock_path.read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':
        raise ValueError('source lock schema mismatch')
    data,checkpoint_sha=locked_bytes(lock,checkpoint_path,allow_fetch)
    model=load_pretrained(lock,data)
    (inputs,answers,holes),roster=prepare_roster(lock)
    torch.set_num_threads(1)
    with torch.no_grad():
        z1=model.reason_states(inputs)[1].detach().double()
        exact=copy.deepcopy(model).double().eval()
        clean=z1
        for _ in range(3):clean=exact.block(clean)
        clean_labels=exact.head(clean).argmax(-1)
        # This test checks finite-step stability from stage 1, not correctness.
        baseline_full_correct=int((clean_labels==answers).all(-1).sum())
        # Invariance requires a self-mapping region, separately checked.
        inverse_box=Interval.around(clean,0.02)
        invariant=invariant_box_check(inverse_box,exact.block,exact.head,clean_labels)
        mask=single_cell_mask(inputs)
        rows=[]
        t0=time.perf_counter()
        for mode in MODES:
            this_mask=None if mode=='all_coordinates' else mask
            for a in AMPS:
                eps=float(a)
                c,meta=predictive_goal_region(exact,z1,eps,mask=this_mask)
                result={**verify_predicate(c,clean_labels,answers),
                        'mode':mode,'amplitude':a,
                        'max_final_hidden_box_width':meta['final_interval_width'],
                        'per_block_box_width':meta['per_step_width_upper']}
                if eps in (0.0,0.02,0.10,0.50):
                    result['independent_forward_checks']=observed_attacks(exact,z1,clean_labels,c,eps,this_mask)
                rows.append(result)
        elapsed=round(time.perf_counter()-t0,3)
    return {'schema':'LSV-07/predictive-interval-checkpoint-v1',
            'classification':'conditional_float64_interval_on_independently_pretrained_TRM_inspired_checkpoint',
            'checkpoint_sha256':checkpoint_sha,
            'roster_sha256':roster,'source_revision':lock['revision'],
            'n_puzzles':len(inputs),'n_missing_per_puzzle':int(holes[0].sum()),
            'undamaged_fp64_full_correct_count':baseline_full_correct,
            'test_uses_ground_truth_to_construct_intervals':False,
            'test_uses_damage_trajectories_to_construct_intervals':False,
            'target':'unchanged predicted output under ANY stage-one perturbation in the declared box',
            'precision_scope':'ideal real arithmetic with stored FP32 tensors as exact coefficients; numerical implementation uses FP64 with explicit assumed erf/BLAS/sqrt errors and unverified backend',
            'fully_verified_numerical_certificates':False,
            'invariance_at_depth_four_clean_box_radius_0_02':invariant,
            'seconds_excluding_checkpoint_loading_and_clean_forward':elapsed,
            'torch_version':torch.__version__,
            'rows':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    receipt=run(args.lock,args.checkpoint,args.fetch)
    out=json.dumps(receipt,indent=2,sort_keys=True)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(out)
    # CI log summary; complete receipt is archived separately.
    print('LSV-07 CHECKPOINT',receipt['checkpoint_sha256'])
    print('ROSTER',receipt['roster_sha256'])
    print('INVARIANT',json.dumps(receipt['invariance_at_depth_four_clean_box_radius_0_02']))
    for row in receipt['rows']:
        print('ROW',row['mode'],row['amplitude'],row['certified_stable_cells'],
              row['certified_correct_cells'],row['fully_certified_correct_puzzles'],
              'maxwidth',row['max_final_hidden_box_width'])
    print('RECEIPT',str(args.out),'seconds',receipt['seconds_excluding_checkpoint_loading_and_clean_forward'])


if __name__=='__main__':main()
