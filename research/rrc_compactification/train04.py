"""RRC-04 predeclared CPU training from raw coefficient prefixes.

Train and development contain only products with caps {unlimited,1,2} and
moduli {2,3,4,6,8,9,10,12}. Hidden-modulus products, gap constraints,
nonperiodic sets and future-defect pairs are never constructed during training.
Frozen canonical targets are computed from train-prefix Euler inversion, not
source-program labels. Use `--steps 650 --batch-size 128` for registered pilot.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from .program04 import (HERE, PREFIX, SEED, TRAIN_PERIODS, TEST_PERIODS,
                            roster, synthesize, roster_digest, simple_features)
    from .student04 import (ProgramStudent, RECUR_PARAMS, FF_PARAMS, RNN_H, FF_H)
except ImportError:
    from program04 import (HERE, PREFIX, SEED, TRAIN_PERIODS, TEST_PERIODS,
                           roster, synthesize, roster_digest, simple_features)
    from student04 import (ProgramStudent, RECUR_PARAMS, FF_PARAMS, RNN_H, FF_H)

TRAIN_COUNT=1500
DEV_COUNT=180
STEPS=650
BATCH=128
DEPTH=2
LR=0.008


def train(*, steps:int=STEPS, batch_size:int=BATCH, train_size:int=TRAIN_COUNT,
          dev_size:int=DEV_COUNT, out_dir:Path=HERE/'results',seed:int=SEED)->dict:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if min(steps,batch_size,train_size,dev_size)<1: raise ValueError('positive budgets required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits=roster(train_size=train_size,dev_size=dev_size,include_test=False)
    train_cases=splits['train']
    dev_cases=splits['dev']
    samples=train_cases+dev_cases
    labels=[synthesize(c.prefix) for c in samples]
    if any(p is None for p in labels): raise AssertionError('synthetic training grammmar not recoverable')
    x=torch.tensor([simple_features(c.prefix) for c in samples],dtype=torch.float32)
    flat=x[:,:,:2].reshape(len(samples),-1)
    cap=torch.tensor([p.cap for p in labels],dtype=torch.long)
    period=torch.tensor([p.period-1 for p in labels],dtype=torch.long)
    mask=torch.tensor([[float(r in p.residues) for r in range(12)] for p in labels],dtype=torch.float32)
    active=torch.tensor([[float(r<p.period) for r in range(12)] for p in labels],dtype=torch.float32)
    num_train=len(train_cases)
    hash_train=sha256(json.dumps(
        [(c.name,list(c.prefix),labels[i].as_dict()) for i,c in enumerate(train_cases)],
        separators=(',',':'),sort_keys=True).encode()).hexdigest()
    hash_dev=sha256(json.dumps(
        [(c.name,list(c.prefix),labels[num_train+i].as_dict()) for i,c in enumerate(dev_cases)],
        separators=(',',':'),sort_keys=True).encode()).hexdigest()

    class Rec(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru=nn.GRU(3,RNN_H,batch_first=True)
            self.refine=nn.Linear(RNN_H,RNN_H)
            self.head=nn.Linear(RNN_H,27)
        def forward(self,x,flat,depth=DEPTH):
            h=self.gru(x)[1][0]
            for _ in range(depth): h=h+.20*self.refine(h).tanh()
            return self.head(h)

    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Linear(130,FF_H)
            self.head=nn.Linear(FF_H,27)
        def forward(self,x,flat,depth=DEPTH):
            return self.head(self.encoder(flat).tanh())

    def obj(logits, indices, cap_labels=cap, period_labels=period, mask_labels=mask,
            active_labels=active):
        positive=F.binary_cross_entropy_with_logits(logits[:,15:],mask_labels[indices],reduction='none')
        mask_loss=(positive*active_labels[indices]).sum()/active_labels[indices].sum()
        return F.cross_entropy(logits[:,:3],cap_labels[indices])+F.cross_entropy(
            logits[:,3:15],period_labels[indices])+mask_loss

    out_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for index,kind in enumerate(('recurrent','feedforward','shuffled_recurrent')):
        initial_seed=seed+10*(index+1)
        torch.manual_seed(initial_seed)
        network=FF() if kind=='feedforward' else Rec()
        parameters=sum(p.numel() for p in network.parameters())
        if parameters!=(FF_PARAMS if kind=='feedforward' else RECUR_PARAMS):
            raise AssertionError('unexpected architecture parameter budget')
        optimizer=torch.optim.Adam(network.parameters(),lr=LR)
        rng=random.Random(initial_seed+404)
        # Fixed random re-assignment destroys the relation between raw coefficients
        # and the exact program label while retaining the same label marginals.
        if kind=='shuffled_recurrent':
            order=list(range(num_train)); rng.shuffle(order)
            sc=cap[order].clone(); sp=period[order].clone()
            sm=mask[order].clone(); sa=active[order].clone()
            cap_labels=torch.cat((sc,cap[num_train:]))
            period_labels=torch.cat((sp,period[num_train:]))
            mask_labels=torch.cat((sm,mask[num_train:]))
            active_labels=torch.cat((sa,active[num_train:]))
        else:
            cap_labels,period_labels,mask_labels,active_labels=cap,period,mask,active
        curve=[]
        for step in range(steps):
            ids=torch.tensor([rng.randrange(num_train) for _ in range(batch_size)],dtype=torch.long)
            logits=network(x[ids],flat[ids])
            loss=obj(logits,ids,cap_labels,period_labels,mask_labels,active_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step+1)%100==0 or step==steps-1:
                with torch.no_grad():
                    dev_ids=torch.arange(num_train,len(samples))
                    dev_logits=network(x[dev_ids],flat[dev_ids])
                    dev_loss=obj(dev_logits,dev_ids)
                curve.append([step+1,round(float(loss.detach()),6),round(float(dev_loss),6)])
        state=network.state_dict()
        if kind=='feedforward':
            tensor_keys={
                'encoder_w':'encoder.weight','encoder_b':'encoder.bias',
                'head_w':'head.weight','head_b':'head.bias'}
        else:
            tensor_keys={
                'gru_ih_w':'gru.weight_ih_l0','gru_hh_w':'gru.weight_hh_l0',
                'gru_ih_b':'gru.bias_ih_l0','gru_hh_b':'gru.bias_hh_l0',
                'refine_w':'refine.weight','refine_b':'refine.bias',
                'head_w':'head.weight','head_b':'head.bias'}
        weights={k:state[v].detach().tolist() for k,v in tensor_keys.items()}
        checkpoint={
            'schema':'rrc04.student.v1','variant':kind,'parameters':parameters,
            'weights':weights,
            'training':{'seed':initial_seed,'roster_seed':seed,'train_size':num_train,
                        'dev_size':len(dev_cases),'excluded_test_generators':
                        ['local_gap','nonperiodic','late_defect'],
                        'held_out_moduli':list(TEST_PERIODS),
                        'train_moduli':list(TRAIN_PERIODS),
                        'train_prefix_sha256':hash_train,'dev_prefix_sha256':hash_dev,
                        'steps':steps,'batch_size':batch_size,'lr':LR,'depth':DEPTH,
                        'loss_curve_train_dev':curve,
                        'optimizer':'Adam','torch_version':torch.__version__,
                        'source_labels_shuffled':kind=='shuffled_recurrent'}}
        portable=ProgramStudent(checkpoint)
        for sample_id in (0,18,130,num_train,num_train+len(dev_cases)-1):
            with torch.no_grad(): expected=network(x[sample_id:sample_id+1],flat[sample_id:sample_id+1])[0].tolist()
            got=portable.logits(samples[sample_id].prefix)
            err=max(abs(a-b) for a,b in zip(expected,got))
            if err>5e-5:
                raise AssertionError(f'Portable {kind} parity failed, sample {sample_id}, err={err}')
        path=out_dir/f'rrc04_{kind}.json'
        path.write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')
        results[kind]={'checkpoint':str(path),'parameters':parameters,'last_losses':curve[-1],
                       'model_sha256':sha256(path.read_bytes()).hexdigest()}
    results['train_prefix_sha256']=hash_train
    results['dev_prefix_sha256']=hash_dev
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--batch-size',type=int,default=BATCH)
    p.add_argument('--train-size',type=int,default=TRAIN_COUNT)
    p.add_argument('--dev-size',type=int,default=DEV_COUNT)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=p.parse_args()
    print(json.dumps(train(steps=args.steps,batch_size=args.batch_size,
                           train_size=args.train_size,dev_size=args.dev_size,
                           out_dir=args.out_dir),indent=2))

if __name__=='__main__': main()
