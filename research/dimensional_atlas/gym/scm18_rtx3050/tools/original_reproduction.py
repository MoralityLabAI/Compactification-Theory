"""Instrument the unmodified full SCM-15 pipeline on CPU or CUDA.

Device shims only move public feature tensors and model parameters. The
original training, discovery, replay and final-scoring functions are reused.
Launch only after SCM-18 freeze because the original runner opens EvalMath's
official public roster at its own post-training boundary.
"""
import argparse
import json
from pathlib import Path
from research.dimensional_atlas.gym import run as original
from research.dimensional_atlas.gym import tasks
from research.dimensional_atlas.gym.scm18_rtx3050.core import (
    HERE,Meter,device_setup,digest,load_checkpoint,sha,write_new)
from research.dimensional_atlas.gym.scm18_rtx3050.campaign import OracleAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=('cpu','cuda'))
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    args=p.parse_args()
    assert (HERE/'results/freeze.json').is_file(),'main campaign must already be frozen'
    c=json.loads((HERE/'config.json').read_text());device=device_setup(args.device,c)
    meter=Meter(device,c)
    oracle=tasks.load_evalmath(args.oracle);audit=OracleAudit(oracle,allow_final=True)
    original_class=original.LoopedMathTransformer
    native_tensorize=original.tensorize;native_encode=tasks.encode;native_steps=original.optimizer_steps
    captured={}
    def make_model(**kwargs):
        model=original_class(**kwargs)
        captured['initial_sha256']=digest(model)
        captured['model']=model
        model.to(device)
        model.register_forward_pre_hook(lambda m,a:meter.forward(len(a[0]),m.visits))
        return model
    def steps(*args,**kwargs):
        meter.counts[kwargs['stage']+'_optimizer_steps']+=kwargs['steps']
        return native_steps(*args,**kwargs)
    original.LoopedMathTransformer=make_model
    original.tensorize=lambda ts: native_tensorize(ts,device=device)
    tasks.encode=lambda task:native_encode(task).to(device)
    original.encode=tasks.encode
    original.optimizer_steps=steps
    original.load_evalmath=lambda path:oracle
    private_receipt=args.private/f'original_full_{args.device}.json'
    original.main(['--backend','evalmath','--evalmath-source',str(args.oracle),
        '--steps','3','--sae-steps','2','--vpd-steps','3','--replay-steps','2',
        '--train-count','24','--discovery-count','12','--readout-count','12',
        '--hidden','16','--visits','4','--sae-width','20','--sae-k','3',
        '--vpd-rank','2','--batch','4','--output',str(private_receipt)])
    receipt=json.loads(private_receipt.read_text())
    write_new(HERE/f'results/original_evalmath_{args.device}.json',{
        'source':'unmodified SCM15 run.py/model.py with device-only shims and accounting hooks',
        'shim_source_sha256':sha(__file__),'initial_sha256':captured['initial_sha256'],
        'frozen_model_sha256':digest(captured['model']),'original_receipt':receipt,
        'resources':meter.report(),'oracle_source_calls':dict(audit.counts),
        'private_gold_sha256':sha(private_receipt.with_suffix('.gold.private.json')),
        'generated_public_sha256':sha(private_receipt.with_suffix('.generated.public.json'))})


if __name__=='__main__':main()
