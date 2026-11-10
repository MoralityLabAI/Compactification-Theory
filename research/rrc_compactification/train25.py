"""RRC-25 CPU-only fit; teacher supplies true base header on training cases.

At inference the full public four-base header scan constructs candidate headers.
Training never calls held-out generators or sees future targets as features.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
try:
    from . import compound25 as b, model25 as m
except ImportError:
    import compound25 as b
    import model25 as m

HERE=Path(__file__).resolve().parent
STEPS=400
BATCH=96
LR=.005


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in
        ('compound25.py','model25.py','train25.py','delay24.py','graph11.py')}


def modules():
    from torch import nn
    class GRU(nn.Module):
        def __init__(self):
            super().__init__();self.rnn=nn.GRU(8,m.HIDDEN,batch_first=True)
            self.head=nn.Linear(m.HIDDEN,16)
        def forward(self,x):return self.head(self.rnn(x)[0][:,-1])
    class FF(nn.Module):
        def __init__(self):
            super().__init__();self.fc=nn.Linear(72,m.FF_HIDDEN)
            self.head=nn.Linear(m.FF_HIDDEN,16)
        def forward(self,x):
            import torch
            return self.head(torch.tanh(self.fc(x.flatten(1))))
    return {'recurrent':GRU,'feedforward':FF,'shuffled':GRU}


def training_arrays(items):
    import torch
    features=[];targets=[]
    for c in items:
        op,k,delta,amp,_,_=c.program
        h=b.Header(op,k,delta,amp)
        features.append(b.features(c,h));targets.append(b.labels(c.program))
    return torch.tensor(features,dtype=torch.float32),torch.tensor(targets,dtype=torch.long)


def fit(*,steps=STEPS,train_per=b.TRAIN_PER,dev_per=b.DEV_PER,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=b.roster(include_test=False,train_per=train_per,dev_per=dev_per)
    train=data['train'];dev=data['dev']
    tx,ty=training_arrays(train);dx,dy=training_arrays(dev)
    rng=random.Random(b.SEED+1212)
    batches=[tuple(rng.randrange(len(train)) for _ in range(BATCH)) for _ in range(steps)]
    shuffled=ty.tolist();random.Random(b.SEED+3939).shuffle(shuffled)
    sy=torch.tensor(shuffled,dtype=torch.long)
    models={}
    for i,(kind,klass) in enumerate(modules().items()):
        torch.manual_seed(b.SEED+700+i)
        net=klass();opt=torch.optim.Adam(net.parameters(),lr=LR)
        for ids in batches:
            j=torch.tensor(ids,dtype=torch.long)
            loss=F.cross_entropy(net(tx[j]),sy[j] if kind=='shuffled' else ty[j])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            preds=net(dx)
            accuracy=float((preds.argmax(-1)==dy).float().mean())
            top4=float((preds.topk(4,dim=-1).indices==dy[:,None]).any(dim=1).float().mean())
        n=sum(z.numel() for z in net.parameters())
        assert n==m.parameters(kind),(kind,n,m.parameters(kind))
        models[kind]={'schema':'rrc25.weights.v1','kind':kind,'parameters':n,
            'steps':steps,'batch':BATCH,'lr':LR,'seed':b.SEED,'optimizer':'Adam.cpu.fp32',
            'train_examples':len(train),'dev_examples':len(dev),
            'train_sha256':b.fingerprint(train),'dev_sha256':b.fingerprint(dev),
            'source_sha256':source_hashes(),'torch_version':torch.__version__,
            'teacher_header_during_training':True,'test_generators_during_training':False,
            'future_access_during_inference':False,'dev_top1':accuracy,'dev_top4':top4,
            'state_dict':{k:v.detach().tolist() for k,v in net.state_dict().items()}}
        if verbose:print(kind,n,'dev top1',round(accuracy,4),'top4',round(top4,4),flush=True)
    return models


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for kind,ck in fit(steps=a.steps).items():
        path=a.out_dir/f'rrc25_{kind}.json'
        path.write_text(json.dumps(ck,indent=2,sort_keys=True)+'\n')
        print('wrote',path)
if __name__=='__main__':main()
