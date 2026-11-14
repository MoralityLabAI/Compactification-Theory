"""Frozen SCM-14 synthetic source Hessian run and its external assumptions.

Run with --out NEW_PATH; `--check --out EXISTING` requires replayed source
hashes and registered envelope thresholds, and retains negative results.
No trained RMP checkpoint, D-shared artifact, or GPU trace is used.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import torch

from scm09_reference import DeterministicRMP
from scm10_reference import _hash_model
from exact_input import rmp_input_enclosure
from ball_centered import verified_ball_tubes
from validated_forward import verified_injection_amplitude
from scm14_hessian import hessian_disk_bounds

SCHEMA='scm14/source_hessian_reference/v1'
FILES=['scm14_hessian.py','rmp_lipschitz.py','ball_centered.py',
       'fast_centered.py','exact_input.py']


def _source_model(*,d=128,seq=64,visits=8,tied=True,reinjection=False,beta=.1):
    model=DeterministicRMP(d=d,seq=seq,visits=visits,tied=tied,
                           reinjection=reinjection,beta=beta)
    frequency_table=torch.tensor([2.**(-k) for k in range((d+1)//2)],dtype=torch.float32)
    source=rmp_input_enclosure(model,model.tokens,positional_frequency_table=frequency_table)
    return model,source


def case(*,d=128,seq=64,visits=8,tied=True,reinjection=False,beta=.1,
         source_local=True):
    model,source=_source_model(d=d,seq=seq,visits=visits,
                               tied=tied,reinjection=reinjection,beta=beta)
    tubes=verified_ball_tubes(model,source,fixed_input_reinjection=reinjection,
                              max_seq=64,max_dim=128,max_visits=8)
    chart=torch.eye(d,dtype=torch.float32)[:,:2]
    injection=4 if seq>=48 else 1
    obs=(16,32,48) if seq>=48 else (2,4,6)
    amplitude=verified_injection_amplitude(tubes,injected_token=injection,
                        after_visit=0,radius_fraction=Fraction(1,10**6))
    source_local=hessian_disk_bounds(model,tubes,axes=chart,chart=chart,
          amplitude=amplitude,injected_token=injection,observed_tokens=obs,
          source_local_values=source_local)
    # The global-QKV variant supplies an independent upper-bound comparator.
    global_qkv=hessian_disk_bounds(model,tubes,axes=chart,chart=chart,
          amplitude=amplitude,injected_token=injection,observed_tokens=obs,
          source_local_values=False)
    def maxima(x):
        return [max(Fraction(v['taylor_remainder_on_unit_disk_upper'])
                    for v in row['observed'].values()) for row in x['per_visit']]
    local,global_=maxima(source_local),maxima(global_qkv)
    if not all(a<=b for a,b in zip(local,global_)):
        raise AssertionError('Source QKV localization made the Hessian bound worse')
    return {'case':('tied' if tied else 'untied')+('_reinjected' if reinjection else ''),
            'model_sha256':_hash_model(model),
            'frequency_sha256':source.frequency_sha256,
            'shape':{'tokens':seq,'hidden':d,'visits':visits,
                     'heads':model.blocks[0].num_heads},
            'amplitude_upper':str(amplitude),
            'injection_token':injection,'observed_tokens':list(obs),
            'local_curvature_error_by_visit':[str(x) for x in local],
            'global_curvature_error_by_visit':[str(x) for x in global_],
            'local_at_most_global':True,
            'source_hessian_certified_under_cpu_contract':True,
            'verified_affine_base_and_jacobian':False,
            'four_dimensional_seifert_invariant':False,
            'physical_float32_execution_certified':False,
            'trained_rmp_evidence':False}


def receipt():
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parent
    return {'schema':SCHEMA,'date':'2026-11-14',
      'proof_scope':'synthetic_exact_real_source_with_conditional_cpu_ieee_BLAS',
      'source_sha256':{name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in FILES},
      'cases':[case(d=128,seq=64,visits=8,tied=True,beta=.1),
               case(d=8,seq=8,visits=3,tied=False,reinjection=True,beta=.1)]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    data=receipt()
    if args.check:
        old=json.loads(args.out.read_text(encoding='utf-8'))
        if old['schema']!=data['schema'] or old['source_sha256']!=data['source_sha256']:
            raise SystemExit('SCM14 source identity drifted')
        for original,new in zip(old['cases'],data['cases']):
            for name in ('model_sha256','frequency_sha256','shape','amplitude_upper',
                         'local_at_most_global'):
                if original[name]!=new[name]:
                    raise SystemExit('SCM14 replay mismatch: '+name)
            # The interval proof is independently recomputed on every host.
            # Center estimates may differ slightly across permitted BLAS
            # implementations. Compare declared synthetic envelopes with a
            # registered tolerance, and enforce a separate absolute cap.
            for key in ('local_curvature_error_by_visit','global_curvature_error_by_visit'):
                if len(original[key])!=len(new[key]):
                    raise SystemExit('SCM14 replay visit count changed: '+key)
                for old,new_value in zip(original[key],new[key]):
                    a,b=float(Fraction(old)),float(Fraction(new_value))
                    if not math.isclose(a,b,rel_tol=.05,abs_tol=1e-13):
                        raise SystemExit('SCM14 synthetic source envelope drift: '+key)
            if original['shape']['hidden']==128:
                if Fraction(new['local_curvature_error_by_visit'][-1])>=Fraction(1,20):
                    raise SystemExit('SCM14 visit-eight source-local cap exceeded')
        print('SCM14 SOURCE-LOCKED SYNTHETIC REFERENCE MATCH')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
            f.write('\n')
        print('SCM14 reference SHA256',hashlib.sha256(args.out.read_bytes()).hexdigest())
        print('full_size_last_curvature_bound',data['cases'][0]['local_curvature_error_by_visit'][-1])

if __name__=='__main__':main()
