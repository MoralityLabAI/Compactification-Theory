"""SCM-11 fixed synthetic 64x128x4-headx8-visit attention-response reference.

Run --out to CREATE an immutable receipt, --check to replay it. Actual neural
weights, deployed float32/CUDA inference, and trained RMP artifacts are not
part of this reference. Conventional CPU IEEE-binary64/BLAS arithmetic is an
explicit conditional verifier premise.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import torch
from scm09_reference import DeterministicRMP
from scm10_reference import _hash_model
from ball_centered import verified_ball_tubes
from exact_input import rmp_input_enclosure
from validated_forward import verified_centered_pair_bounds,verified_injection_amplitude
from refined_response import refined_disk_bounds
from fast_centered import ieee_preflight
from rmp_capture import sha256_file

SCHEMA='scm11_fullshape_source_attention_diameter_v1'
SOURCE_FILES=('refined_response.py','gauge_null.py',
              'fast_centered.py','ball_centered.py','validated_forward.py')


def one_case():
    model=DeterministicRMP(d=128,seq=64,visits=8,tied=True,
                           reinjection=False,beta=.1)
    frequencies=torch.tensor([2.**(-k) for k in range(64)],dtype=torch.float32)
    source=rmp_input_enclosure(model,model.tokens,
                                positional_frequency_table=frequencies)
    tubes=verified_ball_tubes(model,source)
    chart=torch.eye(128,dtype=torch.float32)[:,:2]
    amplitude=verified_injection_amplitude(tubes,injected_token=4,
                         after_visit=0,radius_fraction=Fraction(1,1_000_000))
    refined=refined_disk_bounds(model,tubes,chart=chart,axes=chart,
          amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4)
    legacy=verified_centered_pair_bounds(model,tubes,chart=chart,axes=chart,
          amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4)
    maxima_new=[max(Fraction(v) for v in row.values()) for row in refined['visit_pair_bounds']]
    maxima_old=[max(Fraction(v) for v in row.values()) for row in legacy['per_visit_pair_bounds']]
    # Refined should dominate the older global-value bound on this frozen case.
    if not all(a<=b for a,b in zip(maxima_new,maxima_old)):
        raise AssertionError('SCM-11 regression: a refined bound exceeds the legacy bound')
    if not all(x['null_braid_ideal_real'] for x in refined['null_braid_ideal_real']):
        raise AssertionError('frozen tied source null-braid gate failed')
    return {
        'case':'synthetic_tied_beta_010',
        'model_sha256':_hash_model(model),
        'frequency_sha256':source.frequency_sha256,
        'shape':{'tokens':64,'hidden':128,'heads':4,'visits':8},
        'injection':{'token':4,'after_visit':0,'radius_relative_to_token_rms':'1/1000000'},
        'observed_tokens':[16,32,48],
        'source_amplitude_upper':str(amplitude),
        'legacy_max_pair_radius':[float(x) for x in maxima_old],
        'refined_max_pair_radius':[float(x) for x in maxima_new],
        'refined_over_legacy_at_visit8':float(maxima_new[-1]/maxima_old[-1]),
        'source_null_braid_visits':[x['visit'] for x in refined['null_braid_ideal_real'] if x['null_braid_ideal_real']],
        'all_source_refined_bounds_at_most_legacy':True,
        'CPU_IEEE_BLAS_assumed':True,
        'trained_RMP_checkpoints_tested':False,
        'physical_float_execution_certified':False,
        'observed_neural_link_certified':False,
    }


def receipt():
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parent
    return {
       'schema':SCHEMA,'date':'2026-11-14',
       'evidence':'frozen_synthetic_source_only_conditional_CPU_IEEE_BLAS',
       'backend_sha256':{name:sha256_file(base/name) for name in SOURCE_FILES},
       'numeric_contract':ieee_preflight(),
       'cases':[one_case()]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    current=receipt()
    if args.check:
        frozen=json.loads(args.out.read_text(encoding='utf-8'))
        if frozen['schema']!=current['schema'] or frozen['backend_sha256']!=current['backend_sha256']:
            raise SystemExit('SCM11 source hashes changed; reference requires review')
        for a,b in zip(current['cases'],frozen['cases']):
            for key in ('case','model_sha256','frequency_sha256','source_null_braid_visits',
                        'all_source_refined_bounds_at_most_legacy'):
                if a[key]!=b[key]:
                    raise SystemExit(f'SCM11 reference mismatch: {key}')
            for key in ('legacy_max_pair_radius','refined_max_pair_radius'):
                for x,y in zip(a[key],b[key]):
                    if not math.isclose(x,y,rel_tol=.05,abs_tol=1e-15):
                        raise SystemExit(f'SCM11 reference mismatch in {key}')
        print('SCM11 SOURCE-LOCKED SYNTHETIC REFERENCE MATCH',flush=True)
    else:
        if args.out.exists():
            raise FileExistsError('immutable receipt exists, choose another path')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        s=json.dumps(current,sort_keys=True,indent=2,allow_nan=False)+'\n'
        args.out.write_text(s,encoding='utf-8')
        print('SCM11 REFERENCE SHA256',hashlib.sha256(s.encode()).hexdigest(),flush=True)
        print('SCM11 VISIT8',current['cases'][0]['refined_max_pair_radius'][-1],flush=True)

if __name__=='__main__':main()
