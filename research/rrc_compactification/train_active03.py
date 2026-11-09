"""RRC-03 optional CPU PyTorch imitation run, with teacher-only train labels.

All held-out model selection sizes (13,25,49) are absent from training. A random
mixture of midpoint and nonmidpoint actions supplies off-policy histories.
Run at repository root: python research/rrc_compactification/train_active03.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

try:
    from .active03 import HERE, midpoint_policy, posterior_interval
    from .student03 import (QueryStudent, HIDDEN, MAX_STEPS, PARAMETERS,
                            rnn_inputs, packed_inputs)
except ImportError:
    from active03 import HERE, midpoint_policy, posterior_interval
    from student03 import (QueryStudent, HIDDEN, MAX_STEPS, PARAMETERS,
                           rnn_inputs, packed_inputs)

TRAIN_SIZES=(9,17,33,65)
HELDOUT_SIZES=(13,25,49)
SEED=20261109


def generate_histories(*,seed:int=SEED,per_size:int=1000):
    """Freshly sampled hidden cases and arbitrary informative query histories.

    Targets come exclusively from the public posterior interval implied by the
    history. No test hypothesis size appears in this generator.
    """
    rng=random.Random(seed)
    samples=[]
    for size in TRAIN_SIZES:
        for _ in range(per_size):
            actual=rng.randrange(size)
            history=()
            for step in range(MAX_STEPS+1):
                lo,hi=posterior_interval(history,size)
                if lo==hi:break
                teacher=midpoint_policy(history,size,MAX_STEPS)
                samples.append((history,size,teacher/(size-2)))
                if step==MAX_STEPS:break
                threshold=(teacher if rng.random()<.55 else rng.randrange(lo,hi))
                response=int(actual<=threshold)
                history+=((threshold,response),)
    return samples


def train(*,steps:int=1500, batch_size:int=256, per_size:int=1000,
          seed:int=SEED, out_dir:Path=HERE/'results') -> dict:
    import torch
    from torch import nn
    import torch.nn.functional as F

    if min(steps,batch_size,per_size)<1:raise ValueError('positive run budget required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    samples=generate_histories(seed=seed,per_size=per_size)
    maxlen=MAX_STEPS
    sequence=[]; flat=[]; target=[]; lengths=[]
    for h,size,y in samples:
        v=list(rnn_inputs(h,size))
        sequence.append(v + [(0.0,)*4]*(maxlen-len(v)))
        flat.append(packed_inputs(h,size))
        target.append(y)
        lengths.append(len(h))
    x=torch.tensor(sequence,dtype=torch.float32)
    z=torch.tensor(flat,dtype=torch.float32)
    y=torch.tensor(target,dtype=torch.float32)
    lengths=torch.tensor(lengths,dtype=torch.long)
    roster_hash=hashlib.sha256(json.dumps([(h,s,v) for h,s,v in samples],separators=(',',':')).encode()).hexdigest()

    class Rec(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_layer=nn.Linear(4,HIDDEN)
            self.state=nn.Linear(HIDDEN,HIDDEN,bias=False)
            self.out=nn.Linear(HIDDEN,1)
        def forward(self,x,z,lengths):
            hidden=torch.zeros((x.shape[0],HIDDEN),dtype=x.dtype)
            for i in range(MAX_STEPS):
                next_hidden=(self.in_layer(x[:,i])+self.state(hidden)).tanh()
                hidden=torch.where((lengths>i)[:,None],next_hidden,hidden)
            return self.out(hidden).sigmoid().squeeze(-1)

    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc=nn.Linear(20,HIDDEN)
            self.out=nn.Linear(HIDDEN,1)
        def forward(self,x,z,lengths):
            return self.out(self.fc(z).tanh()).sigmoid().squeeze(-1)

    out_dir.mkdir(parents=True,exist_ok=True)
    summary={}
    for idx,(kind,ctor) in enumerate((('recurrent',Rec),('feedforward',FF))):
        state_seed=seed+idx+77
        torch.manual_seed(state_seed)
        model=ctor()
        count=sum(p.numel() for p in model.parameters())
        if count!=PARAMETERS:raise AssertionError('model parameter mismatch')
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        rng=random.Random(state_seed+19)
        history=[]
        for i in range(steps):
            ids=torch.tensor([rng.randrange(len(samples)) for _ in range(batch_size)],dtype=torch.long)
            pred=model(x[ids],z[ids],lengths[ids])
            loss=F.mse_loss(pred,y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (i+1)%250==0 or i+1==steps:
                history.append([i+1,round(float(loss.detach()),7)])
        if kind=='recurrent':
            layers=(('in',model.in_layer),('state',model.state),('out',model.out))
        else:
            layers=(('fc',model.fc),('out',model.out))
        weights={}
        for name,layer in layers:
            weights[name+'_w']=layer.weight.detach().tolist()
            weights[name+'_b']=None if layer.bias is None else layer.bias.detach().tolist()
        checkpoint={
            'schema':'rrc03.student.v1', 'variant':kind,
            'parameter_count':count, 'weights':weights,
            'training':{'seed':state_seed,'train_sizes':list(TRAIN_SIZES),
                        'heldout_sizes':list(HELDOUT_SIZES),'steps':steps,
                        'batch_size':batch_size,'per_size':per_size,
                        'examples':len(samples),'roster_sha256':roster_hash,
                        'optimizer':'Adam','learning_rate':.01,
                        'loss_history':history,'torch':str(torch.__version__)}}
        portable=QueryStudent(checkpoint)
        for sample_id in (0,17,79,len(samples)//2,len(samples)-1):
            h,size,_=samples[sample_id]
            with torch.no_grad():
                pred=float(model(x[sample_id:sample_id+1],z[sample_id:sample_id+1],lengths[sample_id:sample_id+1])[0])
            if abs(pred-portable.predict_fraction(h,size))>5e-6:
                raise AssertionError(f'torch/portable parity failed {kind}:{sample_id}')
        path=out_dir/f'rrc03_{kind}.json'
        path.write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')
        summary[kind]={'path':str(path),'parameters':count,'final_batch_loss':history[-1][1],
                       'roster_sha256':roster_hash,'examples':len(samples)}
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--per-size',type=int,default=1000)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    a=p.parse_args()
    print(json.dumps(train(steps=a.steps,batch_size=a.batch_size,per_size=a.per_size,out_dir=a.out_dir),indent=2))

if __name__=='__main__':main()
