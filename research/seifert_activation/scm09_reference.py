"""SCM-09 deterministic CPU reference for verified centered transformer bounds.

Synthetic explicit-attention blocks exactly follow RMP's pinned operator graph,
with publicly generated fixed dyadic weight values. These are NOT trained RMP
weights. All recorded proof radii concern exact-real functional semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import time

import torch
from torch import nn

from exact_input import rmp_input_enclosure
from clean_tubes import clean_state_tubes,certified_injection_amplitude
from validated_forward import (verified_centered_tubes,
    verified_centered_pair_bounds,centered_null_disk,verified_injection_amplitude)


class DeterministicExplicitBlock(nn.Module):
    def __init__(self,d:int,h:int,beta:float):
        super().__init__()
        self.hidden_size=d
        self.num_heads=h
        self.head_size=d//h
        self.alpha=1.0
        self.beta=beta
        self.norm1=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False)
        self.attention_out=nn.Linear(d,d,bias=False)
        self.norm2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,4*d,bias=False),nn.GELU(),nn.Linear(4*d,d,bias=False))


class DeterministicRMP(nn.Module):
    def __init__(self,*,d=8,seq=8,visits=8,tied=True,reinjection=False,beta=.1):
        super().__init__()
        self.hidden_size=d
        self.embedding=nn.Embedding(24,d)
        self.input_norm=nn.LayerNorm(d)
        self.blocks=nn.ModuleList([DeterministicExplicitBlock(d,2 if d==8 else 4,beta)
                                   for _ in range(1 if tied else visits)])
        self.module_indices=tuple(0 if tied else v for v in range(visits))
        self.use_sinusoidal_positions=True
        self.carry_policy='reset'
        self.reinjection=reinjection
        self.tokens=torch.tensor([1+i%16 for i in range(seq)],dtype=torch.long)
        self._fill_fixed_weights()
        self.eval()

    @torch.no_grad()
    def _fill_fixed_weights(self):
        # Exact dyadic values; no RNG, no torch version-dependent initializer.
        for j,(name,param) in enumerate(self.named_parameters()):
            raw=torch.arange(param.numel(),dtype=torch.int64)
            if name.endswith('weight') and 'norm' in name:
                values=1+((raw*13+7*j)%17-8).to(torch.float64)/256
            elif name.endswith('bias'):
                values=((raw*7+5*j)%11-5).to(torch.float64)/512
            elif name=='embedding.weight':
                values=((raw*19+3)%131-65).to(torch.float64)/96
            else:
                values=((raw*23+7*j)%47-23).to(torch.float64)/512
            param.copy_(values.reshape(param.shape).to(dtype=param.dtype))


def make_case(*,dimension=8,sequence=8,tied=True,reinjection=False,visits=8)->dict:
    model=DeterministicRMP(d=dimension,seq=sequence,tied=tied,
                           reinjection=reinjection,visits=visits)
    # Frozen exact-dyadic frequency table makes the synthetic receipt
    # independent of CPU/libm exp implementations. Production audit uses
    # the source model's separately verified real frequency table.
    frequencies=torch.tensor([2.0**(-k) for k in range((dimension+1)//2)],dtype=torch.float32)
    initial=rmp_input_enclosure(model,model.tokens,positional_frequency_table=frequencies)
    anchor=clean_state_tubes(model,initial,fixed_input_reinjection=reinjection)
    validated=verified_centered_tubes(model,initial,
        fixed_input_reinjection=reinjection,max_dim=dimension,max_seq=sequence)
    chart=torch.eye(dimension,dtype=torch.float32)[:,:2]
    source_radius_fraction=Fraction(1,10**6)
    source=verified_injection_amplitude(validated,injected_token=0,
          after_visit=0,radius_fraction=source_radius_fraction)
    bounds=verified_centered_pair_bounds(model,validated,chart=chart,axes=chart,
          amplitude=source,observed_tokens=(1,2,4),injected_token=0)
    null=centered_null_disk(validated,chart,(1,2,4),bounds['per_visit_pair_bounds'])
    oldfinal=max(anchor.post_radii[-1]); newfinal=max(validated.post_center_errors[-1])
    return {
        'dimension':dimension,'sequence':sequence,'visits':visits,
        'tied':tied,'reinjection':reinjection,
        'status':'synthetic_source_only_exact_real',
        'initial_radius_max':str(max(initial.radii)),
        'anchored_clean_post_max':str(oldfinal),
        'validated_centered_post_max':str(newfinal),
        'error_radius_improvement_lower':str(oldfinal/newfinal),
        'anchored_per_visit':list(map(lambda r:str(max(r)),anchor.post_radii)),
        'centered_per_visit':list(map(lambda r:str(max(r)),validated.post_center_errors)),
        'injection_amplitude_upper':str(source),
        'null_braid_ideal_real_visits':[n['visit'] for n in null if n['null_braid_ideal_real']],
        'frequency_table_sha256':initial.frequency_sha256,
        'float_execution_certified':False,
        'observed_link_certified':False,
    }


def receipt():
    return {
      'schema':'scm09_exact_real_centered_interval_reference_v1',
      'experiment_date':'2026-11-13',
      'source':'deterministic_synthetic_dyadic_RMP_graph',
      'cases':[make_case(dimension=8,sequence=8,tied=True),
               make_case(dimension=8,sequence=8,tied=False,reinjection=True),
               make_case(dimension=16,sequence=12,tied=True)],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--check',action='store_true',help='compare with source-bound reference; no writes')
    args=parser.parse_args(argv)
    data=json.dumps(receipt(),sort_keys=True,separators=(',',':'))+'\n'
    if args.check:
        if args.out.read_text(encoding='utf-8') != data:
            raise SystemExit('SCM-09 reference mismatch')
        print('SCM09 REFERENCE MATCH')
    else:
        if args.out.exists():
            raise FileExistsError('reference file already exists')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data,encoding='utf-8')
        print('SCM09 source reference',args.out,
              'sha256',hashlib.sha256(data.encode()).hexdigest())


if __name__=='__main__':
    main()
