"""CPU-only supervised base-operator rankers, trained before held-out generation."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
try:
    from . import delay24 as d, model24 as m
except ImportError:
    import delay24 as d
    import model24 as m

HERE=Path(__file__).resolve().parent
STEPS=340
BATCH=96
LR=.006


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest()
            for name in ('delay24.py','model24.py','train24.py','graph11.py')}


def train(*,steps=STEPS,train_per=d.TRAIN_PER,dev_per=d.DEV_PER,verbose=True):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=d.dataset(include_test=False,train_per=train_per,dev_per=dev_per)
    train=data['train'];dev=data['dev']
    features=lambda rows:torch.tensor([d.learn_features(c) for c in rows],dtype=torch.float32)
    tx=features(train);dx=features(dev)
    y=torch.tensor([d.OPS.index(c.program[0]) for c in train],dtype=torch.long)
    dy=torch.tensor([d.OPS.index(c.program[0]) for c in dev],dtype=torch.long)
    rng=random.Random(d.SEED+12345)
    inds=[[rng.randrange(len(train)) for _ in range(BATCH)] for _ in range(steps)]
    perm=y.tolist()
    random.Random(d.SEED+98765).shuffle(perm)
    y_shuffled=torch.tensor(perm,dtype=torch.long)
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__();self.rnn=nn.GRU(4,m.HIDDEN,batch_first=True);self.head=nn.Linear(m.HIDDEN,4)
        def forward(self,x):return self.head(self.rnn(x)[0][:,-1])
    class FF(nn.Module):
        def __init__(self):
            super().__init__();self.fc=nn.Linear(68,m.FF_HIDDEN);self.head=nn.Linear(m.FF_HIDDEN,4)
        def forward(self,x):return self.head(torch.tanh(self.fc(x.flatten(1))))
    out={}
    for index,(kind,model_type) in enumerate((('recurrent',Recurrent),('feedforward',FF),('shuffled',Recurrent))):
        torch.manual_seed(d.SEED+400+index)
        net=model_type()
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        for batch in inds:
            ids=torch.tensor(batch,dtype=torch.long)
            z=net(tx[ids]);loss=F.cross_entropy(z,y_shuffled[ids] if kind=='shuffled' else y[ids])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            p=net(dx);acc=float((p.argmax(-1)==dy).float().mean());loss=float(F.cross_entropy(p,dy))
        count=sum(t.numel() for t in net.parameters())
        assert count==m.params(kind)
        ck={'schema':'rrc24.weights.v1','kind':kind,'steps':steps,'optimizer':'torch.Adam.fp32',
            'seed':d.SEED,'parameter_count':count,'train_cases':len(train),'dev_cases':len(dev),
            'train_roster_sha256':d.fingerprint(train),'dev_roster_sha256':d.fingerprint(dev),
            'source_sha256':source_hashes(),'training_families':list(d.TRAIN_FAMILIES),
            'train_uses_holdout':False,'target':'base operator identity only, not delay or amplitude',
            'batch':BATCH,'lr':LR,'torch_version':torch.__version__,
            'dev_class_accuracy':acc,'dev_loss':loss,
            'state_dict':{k:v.detach().tolist() for k,v in net.state_dict().items()}}
        out[kind]=ck
        if verbose:print(kind,'parameters',count,'dev accuracy',round(acc,4),flush=True)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=HERE/'results');p.add_argument('--steps',type=int,default=STEPS)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    for k,v in train(steps=a.steps).items():
        path=a.out_dir/f'rrc24_{k}.json'
        path.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
        print('wrote',path)
if __name__=='__main__':main()
