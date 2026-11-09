"""RRC-06 deterministic CPU training on only in-distribution periodic sources.

Neural forward passes use raw_features only; exact Euler/log-derivative
inversion creates supervised training labels. No evaluation split is created.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time

try:
    from .rrc06 import (PREFIX,SEED,roster,roster_digest,raw_features,teacher)
    from .raw_student06 import (RECURRENT_HIDDEN,FEEDFORWARD_HIDDEN,
                                DEFAULT_LOOPS,parameter_count,portable_forward)
except ImportError:
    from rrc06 import (PREFIX,SEED,roster,roster_digest,raw_features,teacher)
    from raw_student06 import (RECURRENT_HIDDEN,FEEDFORWARD_HIDDEN,
                               DEFAULT_LOOPS,parameter_count,portable_forward)

HERE=Path(__file__).resolve().parent
TRAIN_STEPS=500
BATCH_SIZE=64
LR=.003


def sha_file(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def prepare(cases):
    import torch
    features=[];classes=[];bscaled=[]
    for case in cases:
        features.append(raw_features(case.prefix))
        a,b=teacher(case.prefix)
        if any(x not in (-1,0,1) for x in a[1:]):
            raise AssertionError('training periodic-products have exactly ternary Euler exponents')
        classes.append([min(2,max(0,v+1)) for v in a])
        bscaled.append([0.]+[math.asinh(b[n]/n)/3 for n in range(1,PREFIX+1)])
    return (torch.tensor(features,dtype=torch.float32),
            torch.tensor(classes,dtype=torch.long),
            torch.tensor(bscaled,dtype=torch.float32))


def build_torch(kind):
    import torch
    from torch import nn
    if kind in ('recurrent','shuffled_recurrent'):
        class R(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru=nn.GRU(3,RECURRENT_HIDDEN,batch_first=True)
                self.head=nn.Linear(RECURRENT_HIDDEN,4)
            def forward(self,x,loops=DEFAULT_LOOPS):
                h=None
                for _ in range(loops):
                    activ,h=self.gru(x,h)
                return self.head(activ)
        return R()
    if kind=='feedforward':
        class F(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1=nn.Linear((PREFIX+1)*3,FEEDFORWARD_HIDDEN)
                self.l2=nn.Linear(FEEDFORWARD_HIDDEN,(PREFIX+1)*4)
            def forward(self,x,loops=None):
                out=self.l2(torch.tanh(self.l1(x.flatten(1))))
                return out.reshape(x.shape[0],PREFIX+1,4)
        return F()
    raise ValueError(kind)


def loss_fn(out,y_a,y_b):
    import torch
    import torch.nn.functional as F
    class_weights=torch.tensor((1.4,.7,1.0),dtype=torch.float32)
    ce=F.cross_entropy(out[:,1:,:3].reshape(-1,3),y_a[:,1:].reshape(-1),
                       weight=class_weights)
    mse=F.mse_loss(out[:,1:,3],y_b[:,1:])
    return ce+.2*mse,ce,mse


def train(*,steps:int=TRAIN_STEPS,batch_size:int=BATCH_SIZE,out_dir:Path|None=None,
          kinds:tuple[str,...]=("recurrent","feedforward","shuffled_recurrent")):
    import torch
    if steps<1 or batch_size<1:raise ValueError('invalid training budget')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False  # stable portable CPU GRU replay
    out_dir=out_dir or HERE/'results'
    out_dir.mkdir(parents=True,exist_ok=True)
    splits=roster(include_test=False)
    train_data=prepare(splits['train'])
    dev_data=prepare(splits['dev'])
    ntrain=len(splits['train'])
    fingerprints={p:sha_file(HERE/p) for p in ('rrc06.py','raw_student06.py','train06.py')}
    output={}
    for idx,kind in enumerate(('recurrent','feedforward','shuffled_recurrent')):
        if kind not in kinds:continue
        torch.manual_seed(SEED+400+idx)
        net=build_torch(kind)
        actual_params=sum(p.numel() for p in net.parameters())
        if actual_params!=parameter_count(kind):raise AssertionError((kind,actual_params))
        optim=torch.optim.Adam(net.parameters(),lr=LR)
        sampler=torch.Generator().manual_seed(SEED+500+idx)
        y_a,y_b=train_data[1:]
        if kind=='shuffled_recurrent':
            perm=torch.randperm(ntrain,generator=torch.Generator().manual_seed(SEED+600))
            y_a=y_a[perm].clone();y_b=y_b[perm].clone()
        traces=[]
        start=time.monotonic()
        for step in range(steps):
            indices=torch.randint(ntrain,(batch_size,),generator=sampler)
            logits=net(train_data[0][indices])
            loss,_,_=loss_fn(logits,y_a[indices],y_b[indices])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if (step+1)%100==0 or step==steps-1:
                with torch.no_grad():
                    dev_logits=net(dev_data[0])
                    dev_loss,ce,mse=loss_fn(dev_logits,*dev_data[1:])
                    dev_pred=dev_logits[:,1:,:3].argmax(dim=-1)
                    acc=(dev_pred==dev_data[1][:,1:]).float().mean().item()
                    traces.append({'step':step+1,'dev_loss':round(float(dev_loss),7),
                                   'dev_a_accuracy':round(acc,7),
                                   'dev_b_mse':round(float(mse),7)})
        elapsed=round(time.monotonic()-start,3)
        state={k:v.detach().cpu().tolist() for k,v in net.state_dict().items()}
        ckpt={
            'schema':'rrc06.raw_coefficient_student.v1',
            'kind':kind,'parameter_count':actual_params,
            'weight_bits_fp32':actual_params*32,
            'live_state_fp32':RECURRENT_HIDDEN if kind!='feedforward' else 0,
            'training_depth':DEFAULT_LOOPS if kind!='feedforward' else 1,
            'weights':state,'feature_contract':'log1p(raw f_n), neighboring log difference, public degree',
            'input_algebraic_preprocessing':'none',
            'source_sha256':fingerprints,
            'train_roster_sha256':roster_digest(splits['train']),
            'dev_roster_sha256':roster_digest(splits['dev']),
            'trained_test_data_access':False,
            'optimization':{'seed':SEED+400+idx,'sampler_seed':SEED+500+idx,
                            'steps':steps,'batch_size':batch_size,'lr':LR,'optimizer':'Adam',
                            'algebraic_teacher_targets':'Euler a_n and asinh((b_n/n))/3',
                            'target_shuffle':kind=='shuffled_recurrent'},
            'development_trace':traces,'torch_version':torch.__version__,
        }
        # PyTorch/portable logits parity independently tested before serialization.
        with torch.no_grad():
            for j in (0,len(splits['dev'])//2,len(splits['dev'])-1):
                torch_logits=net(dev_data[0][j:j+1])[0].tolist()
                portable=portable_forward(raw_features(splits['dev'][j].prefix),ckpt)
                diff=max(abs(a-b) for x,y in zip(torch_logits,portable) for a,b in zip(x,y))
                if diff>5e-5:raise AssertionError(f'portable parity failure {kind}: {diff}')
        fn=out_dir/f'rrc06_{kind}.json'
        fn.write_text(json.dumps(ckpt,indent=2,sort_keys=True)+'\n')
        output[kind]={'parameters':actual_params,'dev_trace':traces[-1],
                      'checkpoint_sha256':sha_file(fn),'wall_seconds_local':elapsed}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=TRAIN_STEPS)
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--kinds',nargs='+',choices=('recurrent','feedforward','shuffled_recurrent'),
                   default=['recurrent','feedforward','shuffled_recurrent'])
    args=p.parse_args()
    print(json.dumps(train(steps=args.steps,batch_size=args.batch_size,
                           out_dir=args.out_dir,kinds=tuple(args.kinds)),indent=2,sort_keys=True))

if __name__=='__main__':main()
