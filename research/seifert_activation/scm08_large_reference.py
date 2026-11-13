"""SCM-08: source-derived clean tubes for the eight-visit RMP-shaped CPU mock.

This larger fixture tests scalability and null-braid refusal without trained
checkpoints. It uses the exact-real dyadic-weight mathematical graph and a
separately frozen positional-frequency table, not certified torch outputs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import hashlib
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_rmp_lipschitz import MiniExplicitBlock
from clean_tubes import (clean_state_tubes,certified_injection_amplitude,
                         certified_zero_disk,verified_local_pair_bounds)
from exact_input import rmp_input_enclosure
from local_jacobian import visit_token_bounds


class LargeMock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size=128
        self.embedding=nn.Embedding(128,128)
        self.input_norm=nn.LayerNorm(128)
        self.use_sinusoidal_positions=True
        self.carry_policy='reset'
        self.blocks=nn.ModuleList([MiniExplicitBlock(d=128,h=4,beta=.25)])
        self.module_indices=(0,)*8


def run_reference() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(317)
    model=LargeMock().eval()
    source=rmp_input_enclosure(model,torch.arange(64))
    clean=clean_state_tubes(model,source)
    chart=torch.eye(128)[:,:2]
    amplitude=certified_injection_amplitude(source,clean,injected_token=4,
                   after_visit=0,radius_fraction='1/1000000')
    baseline=visit_token_bounds(model,sequence=64,chart=chart,axes=chart,
           amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4)
    derived=verified_local_pair_bounds(model,source,chart=chart,axes=chart,
           amplitude=amplitude,observed_tokens=(16,32,48),injected_token=4,clean=clean)
    null=certified_zero_disk(source,clean,chart,derived,(16,32,48))
    def max_pair(report):
        return [str(max(Fraction(x) for x in row.values())) for row in report['per_visit_pair_bounds']]
    return {
       'schema':'scm08_large_cpu_reference_v1',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'source':'untrained_128d_four_head_64_token_tied_eight_visit_mock',
       'seed':317,'torch':str(torch.__version__),
       'exact_real_graph_only':True,
       'floating_output_certified':False,
       'deployed_frequency_table_certified':False,
       'trained_model_evidence':False,
       'source_input_error_max':str(max(source.radii)),
       'derived_clean_post_radius_max':[str(max(row)) for row in clean.post_radii],
       'source_derived_injection_amplitude_upper':str(amplitude),
       'source_global_pair_upper':max_pair(baseline),
       'source_derived_local_pair_upper':max_pair(derived),
       'source_only_null_braid_by_visit':[row['null_braid_for_ideal_real_model'] for row in null],
       'source_only_null_certificate_count':sum(row['null_braid_for_ideal_real_model'] for row in null),
       'requires_tighter_verified_trajectory_enclosure':True,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('refusing to overwrite immutable source receipt')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as file:
        json.dump(run_reference(),file,indent=2,sort_keys=True)
        file.write('\n')
    print(args.out)

if __name__=='__main__':
    main()
