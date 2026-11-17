"""SCM-21 exact-design CPU geometry control, separate from trained RMP evidence.

A rank-four PCA direction learned on calibration donor differences and a rank-four
orthogonal write fiber receive equal-norm patches on fresh held-out pairs.
The frozen readout is a rank-four linear map in R^128. This is an illustrative
constructive geometry witness, not a neural-mechanism experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _pairs(rng: np.random.Generator, n: int, s: np.ndarray, nuisance: np.ndarray):
    rows = []
    for _ in range(n):
        r = int(rng.integers(0,4))
        d = (r + int(rng.integers(1,4))) % 4
        zr,zd = np.zeros(4),np.zeros(4)
        zr[r]=2.0;zd[d]=2.0
        zr += rng.normal(0,0.05,4)
        zd += rng.normal(0,0.05,4)
        shared=rng.normal(0,0.2,4)
        rows.append((s@zr+nuisance@shared,s@zd+nuisance@shared,r,d))
    return rows


def run(seed: int = 20261117, calibration: int = 80, heldout: int = 256) -> dict:
    rng=np.random.default_rng(seed)
    directions=rng.standard_normal((128,12))
    q,_=np.linalg.qr(directions,mode='reduced')
    semantic=q[:,:4];native=q[:,4:8];nuisance=q[:,8:12]
    cal=_pairs(rng,calibration,semantic,nuisance)
    train_d=np.stack([b-a for a,b,_,_ in cal])
    _,singular,vt=np.linalg.svd(train_d,full_matrices=False)
    if singular[3] < 1e-7:raise ValueError('calibration differences fail rank-four test')
    learned=vt[:4].T
    tests=_pairs(rng,heldout,semantic,nuisance)
    successes={'full_difference':0,'difference_pca4':0,'native_fiber4':0}
    errors=[];max_ratio_error=0.;max_orthogonal=0.
    for a,b,recipient,donor in tests:
        delta=b-a
        full=delta
        alt=learned@(learned.T@delta)
        raw=native@(native.T@delta)
        length=np.linalg.norm(raw)
        native_matched=raw/length*np.linalg.norm(delta) if length>1e-10 else native[:,0]*np.linalg.norm(delta)
        if np.argmax(semantic.T@a)!=recipient or np.argmax(semantic.T@b)!=donor:
            raise AssertionError('constructed clean label incorrect')
        if np.linalg.norm(alt)<1e-8:raise AssertionError('zero calibration patch')
        alt=alt/np.linalg.norm(alt)*np.linalg.norm(delta)
        patches={'full_difference':full,'difference_pca4':alt,'native_fiber4':native_matched}
        for label,patch in patches.items():
            successes[label]+=int(np.argmax(semantic.T@(a+patch))==donor)
        max_ratio_error=max(max_ratio_error,abs(np.linalg.norm(alt)-np.linalg.norm(native_matched)))
        max_orthogonal=max(max_orthogonal,float(np.linalg.norm(semantic.T@native_matched)))
        errors.append(float(np.linalg.norm(alt-delta)))
    return {
        'schema':'scm21.synthetic_geometry_control.v1',
        'seed':seed,'calibration_pairs':calibration,'heldout_pairs':heldout,
        'hidden_dimensions':128,'readout_rank':4,'both_patch_ranks':4,
        'same_per_case_patch_norm':bool(max_ratio_error<1e-9),
        'max_norm_matching_error':max_ratio_error,
        'max_native_readout_effect':max_orthogonal,
        'max_diff_pca_reconstruction_error':max(errors),
        'successes':successes,
        'scientific_scope':'constructed linear exact-design CPU control, not trained network behavior'
    }


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'scm21_synthetic_control.json')
    args=p.parse_args()
    result=run()
    data=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=data:raise SystemExit('synthetic receipt changed')
        print('SCM-21 synthetic geometry receipt integrity PASS')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(data)
        print('SCM-21 synthetic geometry receipt written',args.out)
    print(result['successes'])

if __name__=='__main__':main()
