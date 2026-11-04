"""CPU analysis for a deliberately normalized MONO-HODGE/HF export.

Usage:
  python hodge01_analyze.py --pairs normalized_pairs.jsonl --output results.json
  python hodge01_analyze.py --pairs normalized_pairs.jsonl --reference nominal_states.npy --topk 3
  python hodge01_analyze.py --voi voi_normalized.npz --output voi_results.json
  python hodge01_analyze.py --toy

The original Hugging Face archive schemas must be inspected and mapped explicitly
before supplying normalized inputs. Files using object pickle are rejected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import hodge01 as H


def analyze_pairs(pairs_path, reference_path=None, topk=3):
    _, report=H.read_normalized_pairs(pairs_path)
    if reference_path:
        if 'state_differences' not in report:
            raise ValueError('paired activation vectors are required for spectral analysis')
        X=np.load(reference_path,allow_pickle=False)
        vals, V=H.antisymmetric_modes(report.pop('state_differences'),X,topk=topk)
        report['antisymmetric_generalized_eigenvalues']=[float(v) for v in vals]
        report['antisymmetric_eigenvectors_shape']=list(V.shape)
        report['spectral_interpretation']='diagnostic order-sensitive directions, no causal repair'
    else:
        report.pop('state_differences',None)
    report['input_source']=str(Path(pairs_path).resolve())
    return report


def analyze_voi(npz_path,folds=5):
    with np.load(npz_path,allow_pickle=False) as z:
        required={'public_state','latent','continuation_advantage','family_id'}
        if not required.issubset(z.files):
            raise ValueError(f'missing keys: {required-set(z.files)}')
        return H.incremental_probe_value(z['public_state'],z['latent'],z['continuation_advantage'],z['family_id'],folds=folds)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    grp=ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pairs')
    grp.add_argument('--voi')
    grp.add_argument('--toy',action='store_true')
    ap.add_argument('--reference',help='unmodified nominal activations N x d, .npy')
    ap.add_argument('--topk',type=int,default=3)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--output')
    a=ap.parse_args()
    if a.toy: result=H.toy_demo()
    elif a.pairs: result=analyze_pairs(a.pairs,a.reference,a.topk)
    else: result=analyze_voi(a.voi,a.folds)
    out=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.output:Path(a.output).write_text(out,encoding='utf-8')
    print(out)

if __name__=='__main__':main()