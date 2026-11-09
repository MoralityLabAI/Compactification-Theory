"""RRC-05 CPU-only: learn shared gains for a supplied divisor-graph residual circuit.

The model receives exact b_n/n calculated by the public log-derivative
preprocessor; the recurrence already contains the divisor graph. This is a
learned numerical-correction experiment, not raw-coefficient neural discovery.
Tests and held-out sources are never materialized in this file.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from .rrc05 import (PREFIX, SEED, roster, roster_digest, proper_divisors,
                        log_derivative_coefficients, learned_iteration,
                        feedforward_estimate)
    from .rrc import euler_exponents
except ImportError:
    from rrc05 import (PREFIX, SEED, roster, roster_digest, proper_divisors,
                       log_derivative_coefficients, learned_iteration,
                       feedforward_estimate)
    from rrc import euler_exponents

HERE=Path(__file__).resolve().parent
STEPS=300
BATCH=96
LR=.06
DEPTH=7


def source_sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def train(*,steps:int=STEPS,batch_size:int=BATCH,out_dir:Path|None=None):
    import torch
    import torch.nn.functional as F
    if steps<1 or batch_size<1:
        raise ValueError('steps and batch must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out_dir = out_dir or HERE/'results'
    splits=roster(include_test=False)
    data=splits['train']+splits['dev']
    train_len=len(splits['train'])
    bs=[log_derivative_coefficients(c.prefix) for c in data]
    tgt=[euler_exponents(c.prefix) for c in data]
    c = torch.tensor([[0.]+[b[n]/n for n in range(1,PREFIX+1)] for b in bs],dtype=torch.float32)
    y = torch.tensor(tgt,dtype=torch.float32)
    P=torch.zeros(PREFIX+1,PREFIX+1,dtype=torch.float32)
    for n in range(1,PREFIX+1):
        for d in proper_divisors(PREFIX)[n]:
            P[n,d]=d/n
    A=torch.eye(PREFIX+1)+P
    def recurrent(c_batch,raw):
        alpha=2*torch.sigmoid(raw[0]); beta=2*torch.sigmoid(raw[1])
        x=torch.zeros_like(c_batch)
        for _ in range(DEPTH):
            x=x+alpha*(c_batch-x-beta*(x@P.T))
        return x,alpha,beta
    def feedforward(c_batch,raw):
        gain=2*torch.sigmoid(raw[0]); bend=torch.tanh(raw[1])
        return gain*c_batch+bend*torch.tanh(c_batch),gain,bend
    fingerprints={'rrc05.py':source_sha(HERE/'rrc05.py'),
                  'train05.py':source_sha(HERE/'train05.py')}
    trained={}
    for idx,kind in enumerate(('recurrent','feedforward','shuffled_recurrent')):
        torch.manual_seed(SEED+100+idx)
        # Exactly two trained FP32 scalar parameters in every arm.
        raw = torch.nn.Parameter(torch.tensor((-.85,-.55),dtype=torch.float32))
        optim=torch.optim.Adam([raw],lr=LR)
        rng=random.Random(SEED+200+idx)
        if kind=='shuffled_recurrent':
            perm=list(range(train_len))
            rng.shuffle(perm)
            bad_labels=y[perm].clone()
        else:
            bad_labels=None
        trace=[]
        for step in range(steps):
            indices=torch.tensor([rng.randrange(train_len) for _ in range(batch_size)])
            features=c[indices]
            expected=bad_labels[indices] if bad_labels is not None else y[indices]
            if kind=='feedforward':
                predicted,alpha,beta=feedforward(features,raw)
            else:
                predicted,alpha,beta=recurrent(features,raw)
            mse=F.mse_loss(predicted[:,1:],expected[:,1:])
            residual=F.mse_loss((predicted@A.T)[:,1:],features[:,1:])
            # Shuffled control intentionally has no correct residual teacher.
            loss=mse if bad_labels is not None else mse+.20*residual
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if (step+1)%50==0 or step==steps-1:
                with torch.no_grad():
                    dev_c,dev_y=c[train_len:],y[train_len:]
                    res,alpha,beta=(feedforward(dev_c,raw) if kind=='feedforward'
                                     else recurrent(dev_c,raw))
                    dev_mse=F.mse_loss(res[:,1:],dev_y[:,1:]).item()
                    dev_res=F.mse_loss((res@A.T)[:,1:],dev_c[:,1:]).item()
                    trace.append({'step':step+1,'dev_mse':round(dev_mse,9),
                                  'dev_residual_mse':round(dev_res,9)})
        with torch.no_grad():
            output,alpha,beta=(feedforward(c,raw) if kind=='feedforward'
                               else recurrent(c,raw))
            # Portable stdlib parity for actual learned scalar parameters.
            a=float(alpha); b=float(beta)
            for pos in (0, 17, train_len, len(data)-1):
                got=feedforward_estimate(bs[pos],a,b) if kind=='feedforward' else \
                    learned_iteration(bs[pos],DEPTH,a,b)
                diff=max(abs(float(output[pos,n])-got[n]) for n in range(PREFIX+1))
                if diff>2e-5:
                    raise AssertionError(f'portable vs torch mismatch {kind}: {diff}')
            checkpoint={
                'schema':'rrc05.two_scalar.v1',
                'kind':kind,'parameter_count':2,'weight_bits_fp32':64,
                'live_state_values':PREFIX+1 if kind!='feedforward' else 0,
                'depth_trained':DEPTH,'alpha':a,'beta':b,
                'raw_parameters':[float(v) for v in raw],
                'optimizer':{'name':'Adam','steps':steps,'batch_size':batch_size,
                             'lr':LR,'seed':SEED+100+idx,
                             'sampling_seed':SEED+200+idx},
                'train_moduli':[2,3,4,6,8,9,10,12],
                'train_roster_sha256':roster_digest(splits['train']),
                'dev_roster_sha256':roster_digest(splits['dev']),
                'dev_trace':trace,
                'source_sha256':fingerprints,
                'torch_version':torch.__version__,
                'test_seen_during_training':False,
                'permuted_target':kind=='shuffled_recurrent',
                'feature_contract':'public exact b_n/n, computed from f_0..f_64; divisor graph prewired',
            }
            path=out_dir/f'rrc05_{kind}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(checkpoint,sort_keys=True,indent=2)+'\n')
            trained[kind]={'alpha':a,'beta':b,'last_dev':trace[-1],
                           'sha256':source_sha(path)}
    return trained


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--batch-size',type=int,default=BATCH)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=p.parse_args()
    print(json.dumps(train(steps=args.steps,batch_size=args.batch_size,
                           out_dir=args.out_dir),sort_keys=True,indent=2))

if __name__=='__main__':main()
