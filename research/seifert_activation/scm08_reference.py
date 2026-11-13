"""SCM-08 deterministic CPU mock demonstrating independent clean-state tubes.

The fixture is an untrained four-dimensional RMP-like decoder. All positive
ideal-real claims are established from exact stored weights and rational
input enclosures, with explicit source scope. Float comparisons are diagnostics.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_clean_tubes import ToyRMP
from clean_tubes import clean_state_tubes,certified_injection_amplitude,certified_zero_disk,verified_local_pair_bounds
from exact_input import rmp_input_enclosure
from local_jacobian import visit_token_bounds


def run_reference() -> dict:
    torch.set_num_threads(1)
    cases={}
    tokens=torch.tensor([1,2,3,4,5])
    chart=torch.eye(4)[:,:2]
    for tied,reinjection,beta in ((True,False,.0),(True,False,.01),(True,False,.1),
                                  (False,True,.01)):
        torch.manual_seed(617)
        model=ToyRMP(tied=tied,beta=beta,visits=3).eval()
        initial=rmp_input_enclosure(model,tokens)
        tubes=clean_state_tubes(model,initial,fixed_input_reinjection=reinjection)
        amplitude=certified_injection_amplitude(initial,tubes,after_visit=0,
                                 injected_token=0,radius_fraction='1/10000')
        global_bound=visit_token_bounds(model,sequence=5,chart=chart,axes=chart,
              amplitude=amplitude,observed_tokens=(1,2,3),injected_token=0)
        local=verified_local_pair_bounds(model,initial,chart=chart,axes=chart,
              amplitude=amplitude,observed_tokens=(1,2,3),injected_token=0,
              fixed_input_reinjection=reinjection,clean=tubes)
        disk=certified_zero_disk(initial,tubes,chart,local,(1,2,3))
        key=f'{"LOOP-T" if tied else "FF-U"}_beta_{beta}'
        cases[key]={
          'source_global_pair_upper':[str(max(map(Fraction,row.values()))) for row in global_bound['per_visit_pair_bounds']],
          'derived_clean_local_pair_upper':[str(max(map(Fraction,row.values()))) for row in local['per_visit_pair_bounds']],
          'input_error_max':str(max(initial.radii)),
          'clean_post_tube_max':[str(max(row)) for row in tubes.post_radii],
          'injection_amplitude_upper':str(amplitude),
          'source_null_disk_by_visit':[entry['null_braid_for_ideal_real_model'] for entry in disk],
          'independently_derived_clean_centers':local['center_enclosures_verified_by_this_program'],
        }
    return {
      'schema':'scm08_cpu_reference_v1',
      'date':'2026-11-13',
      'seed':617,
      'torch':str(torch.__version__),
      'source_scope':'untrained_RMP_like_mock_exact_dyadic_weights_and_real_functions',
      'frequency_contract':'frozen_host_generated_float32_frequency_table_interpreted_as_exact_dyadic',
      'floating_execution_certified':False,
      'trained_activation_evidence':False,
      'cases':cases,
      'source_sha256':{path:hashlib.sha256((Path(__file__).parent/path).read_bytes()).hexdigest()
          for path in ('exact_input.py','clean_tubes.py','scm08_audit.py')},
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('reference receipt is immutable')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as out:
        json.dump(run_reference(),out,indent=2,sort_keys=True,allow_nan=False)
        out.write('\n')
    print(args.out)

if __name__=='__main__':
    main()
