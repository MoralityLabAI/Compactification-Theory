"""SCM-06 deterministic CPU proof-instrument reference.

Purely local synthetic network: test code reproduces RMP's pre-norm explicit
attention architecture and strict causal mask. No actual trained checkpoint
results are claimed.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import torch

from rmp_lipschitz import block_bound, spectral_upper, visit_bounds, topology_zero_disk_certificate

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'tests'))
from test_rmp_lipschitz import MiniModel, MiniExplicitBlock


def reference() -> dict:
    torch.manual_seed(20261113)
    chart=torch.eye(4)[:,:2]
    axes=chart.clone()
    m=MiniModel(tied=True).eval()
    b=visit_bounds(m,sequence=5,chart=chart,axes=axes,
                   amplitude=Fraction(1,100000),observed_tokens=(1,2,3),injected_token=0)
    prefix=visit_bounds(m,sequence=5,chart=chart,axes=axes,
                        amplitude=Fraction(1,10),observed_tokens=(0,1,2),injected_token=4)
    if any(value!='0' for row in prefix['per_visit_pair_bounds'] for value in row.values()):
        raise AssertionError('Causality negative control failed')
    clean=[[[0.,0.],[8.,0.],[0.,8.]] for _ in range(3)]
    zero_disk=topology_zero_disk_certificate(clean,pair_bounds_result=b)
    fullsize=MiniExplicitBlock(d=128,h=4).eval()
    full=block_bound(fullsize,sequence=64)
    diagonal=spectral_upper(torch.diag(torch.tensor([3.,4.])))
    assert diagonal==4
    return {
        'schema':'scm06_cpu_reference_v1',
        'source':'deterministic_random_untrained_synthetic_explicit_attention_only',
        'real_arithmetic_certificate_scope':'real-valued network using stored floating weights exactly',
        'machine_level_continuity_proven':False,
        'checkpoint_backed_RMP_capture':'unavailable',
        'scalar_norm_fixture':{'diag_3_4_exact_upper':str(diagonal)},
        'tiny_model':{'sequence':5,'hidden':4,'visits':3,'heads':2,
          'pair_bounds_by_visit':[row['0,1'] for row in b['per_visit_pair_bounds']],
          'prefix_control_all_zero':True,
          'separated_baseline_disk_proves_null':[r['conditional_null_if_clean_baseline_exact'] for r in zero_disk]},
        'realistic_shape_random_untrained_block':{'sequence':64,'hidden':128,'heads':4,
          'L_attention_upper':str(full['attention']),
          'L_mlp_upper':str(full['mlp']),
          'L_block_upper':str(full['full'])},
        'trained_model_result':'not_run_missing_external_RMP_artifacts',
    }


def main():
    payload=reference()
    path=HERE/'results'/'scm06_cpu_reference.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({'path':str(path),'sha256':digest,'summary':payload},indent=2))

if __name__=='__main__':
    main()
