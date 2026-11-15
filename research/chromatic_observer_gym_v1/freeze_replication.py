"""Regenerate COE-05 independent synthetic model seeds from COE-04 architecture.

Requires torch 2.10 CPU; reuses the exact native equations of run_frozen_loop.
This intentionally runs no model downloads, uses no cloud compute, and never
fits readouts. The frozen activation and weight artifacts are committed too.
"""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
import torch
try:
    from .run_frozen_loop import freeze, canonical
except ImportError:
    from run_frozen_loop import freeze, canonical

MODEL_SEEDS = (2411, 2412, 2413)


def regenerate(out: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[]
    weights={}
    for seed in MODEL_SEEDS:
        row, w=freeze(seed)
        rows.append(row)
        weights[str(seed)]=w
        print('frozen',seed,row['model_checkpoint_sha256'],row['train_loss'])
    bundle={
        'schema':'COE-04-native-tied-loop-frozen-v1',
        'task':'copy first token, with second token as a distracting query position',
        'model_fit_pairs':'all 8 x 8 symbolic cases before probe fit',
        'probe_fit_contexts':[0,1,2,3],
        'probe_heldout_contexts':[4,5,6,7],
        'evaluator_graph':'K8 on source-token labels 0..7',
        'model_training_framework':{'torch':str(torch.__version__)},
        'items':rows,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:f.write(canonical(bundle))
    with gzip.open(out.with_name('replication_weights.json.gz'),'wt') as f:
        f.write(canonical(weights))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('replication_activations.json.gz'))
    args=parser.parse_args()
    regenerate(args.out)
