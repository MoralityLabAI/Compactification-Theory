"""CPU-only RRC-09 training. Held-out primitive families are NEVER built here."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import synth09 as ds
except ImportError:
    import synth09 as ds

HERE=Path(__file__).resolve().parent
STEPS=450
BATCH=96
LR=0.005


def generate(*,steps:int=STEPS):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(ds.SEED+123)
    cases=ds.generate(include_test=False)
    train=cases['train'];dev=cases['dev']
    labels=[ds.canonical_tokens(c) for c in train]
    mapping={token:i for i,token in enumerate((ds.PAD,)+ds.OPS)}
    targets=torch.tensor([[mapping[p[0]] if p else 0,
                           mapping[p[1]] if len(p)>1 else 0] for p in labels],dtype=torch.long)
    X=torch.tensor([ds.feature_pairs(c) for c in train],dtype=torch.float32)
    Xdev=torch.tensor([ds.feature_pairs(c) for c in dev],dtype=torch.float32)
    dev_labels=[ds.canonical_tokens(c) for c in dev]
    devtarget=torch.tensor([[mapping[p[0]] if p else 0,mapping[p[1]] if len(p)>1 else 0]
                            for p in dev_labels],dtype=torch.long)

    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.GRU(input_size=2,hidden_size=20,batch_first=True)
            self.head=nn.Linear(20,12)
        def forward(self,x):return self.head(self.rnn(x)[0][:,-1])

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(2*(ds.PREFIX+1),33)
            self.head=nn.Linear(33,12)
        def forward(self,x):return self.head(torch.tanh(self.fc1(x.reshape(x.shape[0],-1))))

    def fit(kind,permuted=False):
        torch.manual_seed(ds.SEED + (500 if kind=='feedforward' else 600 if permuted else 700))
        model=FeedForward() if kind=='feedforward' else Recurrent()
        rng=random.Random(ds.SEED+456) # matched optimization minibatches
        n=sum(v.numel() for v in model.parameters())
        actual_targets=targets.clone()
        if permuted:
            perm=torch.randperm(len(train),generator=torch.Generator().manual_seed(ds.SEED+901))
            actual_targets=actual_targets[perm]
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        model.train()
        for step in range(steps):
            ids=torch.tensor(rng.sample(range(len(train)),BATCH),dtype=torch.long)
            optimizer.zero_grad()
            logits=model(X[ids]).reshape(-1,2,6)
            loss=F.cross_entropy(logits[:,0,:],actual_targets[ids,0])+F.cross_entropy(logits[:,1,:],actual_targets[ids,1])
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            logits=model(Xdev).reshape(-1,2,6)
            val_loss=float((F.cross_entropy(logits[:,0,:],devtarget[:,0])+
                          F.cross_entropy(logits[:,1,:],devtarget[:,1])).item())
            dev_acc=float((logits.argmax(dim=-1)==devtarget).all(dim=1).float().mean().item())
        state={key:v.detach().tolist() for key,v in model.state_dict().items()}
        return {'schema':'rrc09.direct.v1','kind':kind,'seed':ds.SEED,'steps':steps,
                'batch_size':BATCH,'learning_rate':LR,'train_size':len(train),
                'training_operators':list(ds.TRAIN_OPS),'heldout_operators':['C','D'],
                'state_dict':state,'parameter_count':n,
                'dev_loss':val_loss,'dev_program_label_accuracy':dev_acc,
                'training_roster_sha256':ds.roster_digest(train,include_full=False),
                'train09_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
                'synth09_sha256':sha256((HERE/'synth09.py').read_bytes()).hexdigest(),
                'train_excludes_test_generators':True}

    recurrent=fit('recurrent')
    feedforward=fit('feedforward')
    shuffled=fit('shuffled_recurrent',permuted=True)

    # Generic execution-guided low-degree ranker. Training examples consist
    # solely of public candidate executions from N/S/R program grammar.
    features=[];labels=[]
    for c in train:
        for program in ds.all_programs(ds.TRAIN_OPS):
            features.append(ds.candidate_features(c.source,c.target,program)+(1,))
            labels.append(int(ds.execute(c.source[:ds.PREFIX+1],program)==c.target[:ds.PREFIX+1]))
    x=torch.tensor(features,dtype=torch.float64)
    y=torch.tensor(labels,dtype=torch.float64)
    param=torch.nn.Parameter(torch.zeros(6,dtype=torch.float64))
    optimizer=torch.optim.Adam([param],lr=.12)
    pos=float(y.sum().item());neg=len(labels)-pos
    for _ in range(360):
        optimizer.zero_grad()
        logits=x@param
        loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(neg/max(1,pos),dtype=torch.float64))
        loss.backward();optimizer.step()
    ranker={'schema':'rrc09.ranker.v1','weights':[float(v) for v in param.detach().tolist()],
            'train_examples':len(train),'positive_pairs':int(pos),'negative_pairs':int(neg),
            'trained_candidate_ops':list(ds.TRAIN_OPS),'test_candidate_ops':list(ds.OPS),
            'optimizer_steps':360,'interpreter_access':True,'oracle_target_prefix_access':True,
            'training_roster_sha256':ds.roster_digest(train,include_full=False),
            'train09_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'synth09_sha256':sha256((HERE/'synth09.py').read_bytes()).hexdigest()}
    return {'recurrent':recurrent,'feedforward':feedforward,'shuffled_recurrent':shuffled,'guided_ranker':ranker}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--steps',type=int,default=STEPS)
    args=p.parse_args()
    output=generate(steps=args.steps)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,content in output.items():
        path=args.out_dir/f'rrc09_{name}.json'
        path.write_text(json.dumps(content,sort_keys=True,indent=2)+'\n')
        print(name,'parameters',content.get('parameter_count',6),'dev_accuracy',content.get('dev_program_label_accuracy'),'->',path)

if __name__=='__main__':main()
