"""Deterministic synthetic CPU fixture for SCM-07 tokenwise local bounds.

The mock is a *random untrained* RMP-equivalent explicit attention block.
State centers are PyTorch float observations with ZERO *assumed* center error,
so local results and null-disk comparisons are CONDITIONAL demonstrations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent/'tests'))
from test_rmp_lipschitz import MiniExplicitBlock
from local_jacobian import capture_float_clean_centers,visit_token_bounds
from rmp_lipschitz import visit_bounds,topology_zero_disk_certificate


class Toy8(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([MiniExplicitBlock(d=128,h=4,beta=.25)])
        self.module_indices=(0,)*8


def run():
    torch.set_num_threads(1)
    torch.manual_seed(317)
    m=Toy8().eval()
    clean_input=torch.randn(1,64,128)
    axes=torch.eye(128,dtype=torch.float32)[:,:2]
    chart=axes
    baseline=capture_float_clean_centers(m,clean_input)
    projected=[(state[[16,32,48]] @ chart).tolist() for state in baseline['post']]
    # Zero error is assumed solely for a controlled synthetic illustration.
    # This script supplies no proof that floating clean states bound exact ones.
    zero=[[0]*64 for _ in range(8)]
    results={}
    for amplitude in ('1/100', '1/10000', '1/1000000'):
        legacy=visit_bounds(m,sequence=64,chart=chart,axes=axes,
             amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4)
        causal=visit_token_bounds(m,sequence=64,chart=chart,axes=axes,
             amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4)
        local=visit_token_bounds(m,sequence=64,chart=chart,axes=axes,
             amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4,
             pre_centers=baseline['pre'],mid_centers=baseline['mid'],
             pre_center_errors=zero,mid_center_errors=zero)
        def maxima(upper):
            return [float(max(map(Fraction, row.values())))
                    for row in upper['per_visit_pair_bounds']]
        disk=topology_zero_disk_certificate(projected,pair_bounds_result=local)
        results[amplitude]={
            'previous_whole_sequence_pair_max':maxima(legacy),
            'new_causal_source_global_pair_max':maxima(causal),
            'new_interval_local_pair_max_conditional':maxima(local),
            'disk_inequality_conditionally_passes':[
                bool(x['conditional_null_if_clean_baseline_exact']) for x in disk],
        }
    negative=visit_token_bounds(m,sequence=64,chart=chart,axes=axes,
             amplitude='1',observed_tokens=(16,32,48),injected_token=63)
    assert all(all(z=='0' for z in row.values()) for row in negative['per_visit_pair_bounds'])
    result={
        'schema':'scm07_cpu_reference_v1',
        'model':'untrained_128d_four_head_T64_tied_eight_visit_cpu_mock',
        'seed':317, 'torch':torch.__version__,
        'center_scope':'assumed_exact_ideal_center_zero_error_SYNTHETIC_ONLY',
        'trained_model_evidence':False,
        'real_activation_certified':False,
        'parameters':{'sequence':64,'hidden':128,'heads':4,'visits':8,'beta':'1/4',
                      'injected_token':4,'observed_tokens':[16,32,48]},
        'radii':results,
        'strict_causal_prefix_injected_at_63':True,
        'math_files_sha256':{f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest()
            for f in ('local_jacobian.py','rmp_lipschitz.py')},
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results/scm07_cpu_reference.json')
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as writer:
        json.dump(run(),writer,sort_keys=True,indent=2,allow_nan=False)
        writer.write('\n')
    print('Written:',args.out)

if __name__=='__main__':
    main()
