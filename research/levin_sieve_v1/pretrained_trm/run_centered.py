"""LSV-08: post-LSV-07 exploratory centered sensitivity on SAME frozen TRM.

Never claim that this exploration was preregistered, and never use the
fixed held-out Sudoku solutions to build a certificate. Same stage-one
checkpoint, same 64 test puzzles, same radii and modes as LSV-07.
"""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import torch

from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from run_predictive import (single_cell_mask,verify_predicate,observed_attacks,
                            AMPS,MODES)
from centered_reachability import certify_centered

HERE=Path(__file__).resolve().parent


def run(lock_path:Path,checkpoint_path:Path|None=None,allow_fetch=False):
    lock=json.loads(lock_path.read_text())
    if lock['schema']!='LSV-05/pretrained-trm-source-lock-v1':
        raise ValueError('source-lock mismatch')
    data,sha=locked_bytes(lock,checkpoint_path,allow_fetch)
    model_fp32=load_pretrained(lock,data)
    (inputs,targets,holes),roster_sha=prepare_roster(lock)
    torch.set_num_threads(1)
    with torch.no_grad():
        # Identical stage-one stored FP32 values to LSV-07 and LSV-06.
        z1=model_fp32.reason_states(inputs)[1].detach().double()
    model=copy.deepcopy(model_fp32).double().eval()
    with torch.no_grad():
        clean=z1.clone()
        for _ in range(3):clean=model.block(clean)
        clean_pred=model.head(clean).argmax(-1)
        mask=single_cell_mask(inputs)
        rows=[]
        t0=time.perf_counter()
        for mode in MODES:
            selected_mask=mask if mode=='first_missing_cell' else None
            for a in AMPS:
                eps=float(a)
                classes,meta,center,radius=certify_centered(model,z1,eps,selected_mask)
                row={**verify_predicate(classes,clean_pred,targets),
                     'mode':mode,'amplitude':a,
                     'per_block_radius_upper':meta['per_block_radius_upper'],
                     'max_final_radius':meta['max_final_radius'],
                     'mean_final_radius':meta['mean_final_radius']}
                if eps in (0.,0.02,0.10,0.50):
                    row['independent_forward_checks']=observed_attacks(model,z1,clean_pred,classes,eps,selected_mask)
                rows.append(row)
    return {'schema':'LSV-08/centered-reachability-v1',
        'status':'EXPLORATORY_HYPOTHESIS_GENERATION_AFTER_LSV07_FAIL',
        'checkpoint_sha256':sha,'roster_sha256':roster_sha,
        'source_revision':lock['revision'],
        'n_cases':len(inputs),
        'clean_fp64_full_correct':int((clean_pred==targets).all(-1).sum()),
        'conditional_on':'ideal exact nominal real-valued centers and exact gain arithmetic, numerical backend currently Torch FP64',
        'formally_verified_numeric_end_to_end':False,
        'finite_horizon_only':True,
        'elapsed_seconds_excluding_loading':round(time.perf_counter()-t0,3),
        'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,default=HERE/'source_lock.json')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    receipt=run(a.lock,a.checkpoint,a.fetch)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('LSV-08 SHA256',receipt['checkpoint_sha256'])
    print('ROSTER',receipt['roster_sha256'])
    for row in receipt['rows']:
        print('ROW',row['mode'],row['amplitude'],row['certified_stable_cells'],
              row['certified_correct_cells'],row['fully_certified_correct_puzzles'],
              'radius',row['max_final_radius'])
    print('elapsed_seconds',receipt['elapsed_seconds_excluding_loading'])

if __name__=='__main__':main()
