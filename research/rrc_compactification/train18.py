"""RRC-18 CPU training of small recurrent, FF and shuffled class predictors.

Class labels derive from exact polynomial normal forms, restricted to the
training roster. Inference sees raw demonstrations and a public topology.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import learn18 as l, model18 as m
except ImportError:
    import learn18 as l
    import model18 as m

HERE=Path(__file__).resolve().parent
SEED=l.SEED
STEPS=900
BATCH=128
LR=0.003

def sources():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('certify18.py','learn18.py','model18.py','train18.py')}

def modules():
    from torch import nn
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.GRU(9,m.HIDDEN,batch_first=True)
            self.head=nn.Linear(m.HIDDEN,64)
        def forward(self,x):return self.head(self.rnn(x)[0][:,-1,:])
    class Feedforward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(56,m.FF_HIDDEN)
            self.head=nn.Linear(m.FF_HIDDEN,64)
        def forward(self,x):return self.head(self.fc1(x).tanh())
    return {'recurrent':Recurrent,'feedforward':Feedforward,'shuffled':Recurrent}

def train(*,steps=STEPS,train_per=l.TRAIN_PER,dev_per=l.DEV_PER,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=l.roster(include_test=False,train_per=train_per,dev_per=dev_per)
    classes=l.class_map()
    training=data['train'];dev=data['dev']
    x_seq=torch.tensor([l.sequence(c) for c in training],dtype=torch.float32)
    x_flat=torch.tensor([l.flat(c) for c in training],dtype=torch.float32)
    d_seq=torch.tensor([l.sequence(c) for c in dev],dtype=torch.float32)
    d_flat=torch.tensor([l.flat(c) for c in dev],dtype=torch.float32)
    y=torch.tensor(l.labels(training,classes),dtype=torch.long)
    dy=torch.tensor(l.labels(dev,classes),dtype=torch.long)
    mask={topo:[g in classes[topo]['representatives'] for g in l.g.GRAPHS] for topo in l.TOPOLOGIES}
    tm=torch.tensor([mask[c.topology] for c in training],dtype=torch.bool)
    dm=torch.tensor([mask[c.topology] for c in dev],dtype=torch.bool)
    assert bool(tm[torch.arange(len(y)),y].all())
    rng=random.Random(SEED+451)
    batches=[[rng.randrange(len(training)) for _ in range(BATCH)] for _ in range(steps)]
    permuted=y.clone()
    shuffler=random.Random(SEED+881)
    for topology in l.TOPOLOGIES:
        indices=[i for i,c in enumerate(training) if c.topology==topology]
        values=[int(permuted[i]) for i in indices]
        shuffler.shuffle(values)
        for i,val in zip(indices,values):permuted[i]=val
    checkpoints={}
    for i,(kind,cls) in enumerate(modules().items()):
        torch.manual_seed(SEED+3100+i)
        net=cls();optimizer=torch.optim.Adam(net.parameters(),lr=LR)
        x=x_flat if kind=='feedforward' else x_seq
        xd=d_flat if kind=='feedforward' else d_seq
        target=permuted if kind=='shuffled' else y
        for ids_list in batches:
            ids=torch.tensor(ids_list,dtype=torch.long)
            logits=net(x[ids]).masked_fill(~tm[ids],-1e4)
            loss=F.cross_entropy(logits,target[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            prediction=net(xd).masked_fill(~dm,-1e4)
            dev_loss=float(F.cross_entropy(prediction,dy))
            dev_top1=float((prediction.argmax(1)==dy).float().mean())
            dev_top12=float((prediction.topk(12,dim=1).indices==dy[:,None]).any(1).float().mean())
        count=sum(v.numel() for v in net.parameters())
        assert count==m.parameter_count(kind)
        checkpoints[kind]={
            'schema':'rrc18.weights.v1','kind':kind,'seed':SEED,'optimizer_steps':steps,
            'learning_rate':LR,'batch_size':BATCH,'parameter_count':count,
            'train_per_topology':train_per,'dev_per_topology':dev_per,
            'source_sha256':sources(),'train_roster_sha256':l.digest(training),
            'dev_roster_sha256':l.digest(dev),
            'supervision':'canonical exact finite-degree polynomial class, source graph labels only for training',
            'inference_candidate_executions':0,'inference_receives_hidden_graph':False,
            'inference_receives_future':False,'product_source_in_training':False,
            'torch_version':torch.__version__,'dev_cross_entropy':dev_loss,
            'dev_canonical_top1':dev_top1,'dev_canonical_top12':dev_top12,
            'state_dict':{k:v.detach().tolist() for k,v in net.state_dict().items()}}
        if verbose:print(kind,count,'dev loss',round(dev_loss,4),'top1',round(dev_top1,4),'top12',round(dev_top12,4),flush=True)
    return checkpoints

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',type=Path,default=HERE/'results')
    ap.add_argument('--steps',type=int,default=STEPS)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for kind,ck in train(steps=args.steps).items():
        path=args.out_dir/f'rrc18_{kind}.json'
        path.write_text(json.dumps(ck,indent=2,sort_keys=True)+'\n')
        print('wrote',path)
if __name__=='__main__':main()
