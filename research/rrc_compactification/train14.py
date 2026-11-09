"""RRC-14 supervised CPU policy imitation from familiar-graph edit traces.

All counterfactual neighbor executions and hidden canonical starts are confined
within the training roster. Test generator and hidden test graph labels are
never constructed in fitting. Optimizer and Python/torch metadata are frozen.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import graph14 as b, policy14 as p, graph13 as prior, policy13 as prior_policy
except ImportError:
    import graph14 as b
    import policy14 as p
    import graph13 as prior
    import policy13 as prior_policy

HERE=Path(__file__).resolve().parent
SEED=b.SEED
STEPS=420
BATCH=96
LR=0.004
LENGTH=4


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('graph14.py','policy14.py','train14.py','graph11.py','graph12.py','graph13.py','policy13.py')}


def initial_checkpoint():
    path=HERE/'results'/'rrc13_active.json'
    ck=json.loads(path.read_text())
    if ck['kind']!='active' or ck['parameter_count']!=6148:raise AssertionError('bad initializer checkpoint')
    return ck,sha256(path.read_bytes()).hexdigest()


def start_graph(case,prior_checkpoint):
    """A frozen feedback-disabled RRC-13 proposal (zero interpreter calls)."""
    return tuple(prior_policy.infer(case,prior_checkpoint,feedback_enabled=False)['graph'])


def trajectories(cases,prior_checkpoint,seed,*,shuffled=False):
    """One near-graph and one frozen-RRC13 trajectory per training case.

    Teacher chooses the best next neighbor using nine *training-only*
    counterfactual graph executions at each step, with deterministic ties.
    """
    rng=random.Random(seed)
    rows=[];counterfactual=0
    for c in cases:
        for version in (0,1):
            if version==0:
                current=start_graph(c,prior_checkpoint)
            else:
                current=list(c.hidden_graph)
                for position in rng.sample(range(3),2):
                    current[position]=rng.choice([x for x in b.OPS if x!=current[position]])
                current=tuple(current)
            X=[];Y=[];M=[];visited=set()
            for _ in range(LENGTH):
                state=b.feedback(c,current)
                X.append(b.action_features(c,current,state))
                if state['accepted']:
                    Y.append(0);M.append(0)
                else:
                    # All nine neighbor outputs are training-only supervision.
                    choice,next_graph=b.teacher(c,current)
                    Y.append(choice);M.append(1)
                    counterfactual+=9
                    current=next_graph
                visited.add(current)
            rows.append((X,Y,M))
    if shuffled:
        # Permute labels only among nonpadding positions across training rows.
        ids=[(r,t) for r,(_,_,m) in enumerate(rows) for t,v in enumerate(m) if v]
        labels=[rows[r][1][t] for r,t in ids]
        rng.shuffle(labels)
        for (r,t),label in zip(ids,labels):rows[r][1][t]=label
    return rows,counterfactual


def modules():
    import torch
    from torch import nn
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell=nn.GRUCell(p.FEATURES,p.HIDDEN)
            self.head=nn.Linear(p.HIDDEN,p.ACTIONS)
        def forward(self,x):
            h=torch.zeros((x.shape[0],p.HIDDEN),dtype=x.dtype,device=x.device)
            outs=[]
            for j in range(x.shape[1]):
                h=self.cell(x[:,j,:],h)
                outs.append(self.head(h))
            return torch.stack(outs,dim=1)
    class Feedforward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(p.FEATURES,p.FF_HIDDEN)
            self.head=nn.Linear(p.FF_HIDDEN,p.ACTIONS)
        def forward(self,x):return self.head(torch.tanh(self.fc1(x)))
    return {'recurrent':Recurrent,'feedforward':Feedforward,'shuffled':Recurrent}


def fit(*,steps=STEPS,train_count=b.TRAIN_N,dev_count=b.DEV_N,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    groups=b.dataset(include_test=False,train_count=train_count,dev_count=dev_count)
    ck,prior_sha=initial_checkpoint()
    train,teacher_ops=trajectories(groups['train'],ck,SEED+99)
    dev,dev_ops=trajectories(groups['dev'],ck,SEED+199)
    shuffled,_=trajectories(groups['train'],ck,SEED+99,shuffled=True)
    def tensors(rows):
        return tuple(torch.tensor([r[i] for r in rows],dtype=(torch.float32 if i==0 else torch.long if i==1 else torch.float32)) for i in range(3))
    tx,ty,tm=tensors(train)
    dx,dy,dm=tensors(dev)
    sx,sy,sm=tensors(shuffled)
    assert torch.equal(tx,sx) and torch.equal(tm,sm)
    rng=random.Random(SEED+345)
    minibatches=[[rng.randrange(len(train)) for _ in range(BATCH)] for _ in range(steps)]
    result={}
    for idx,(kind,klass) in enumerate(modules().items()):
        torch.manual_seed(SEED+700+idx)
        model=klass()
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        x,y,m=(sx,sy,sm) if kind=='shuffled' else (tx,ty,tm)
        for inds in minibatches:
            inds=torch.tensor(inds)
            z=model(x[inds])
            weights=m[inds]
            loss=(F.cross_entropy(z.reshape(-1,p.ACTIONS),y[inds].reshape(-1),reduction='none')*weights.reshape(-1)).sum()/weights.sum().clamp_min(1)
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            z=model(dx)
            dev_loss=float((F.cross_entropy(z.reshape(-1,p.ACTIONS),dy.reshape(-1),reduction='none')*dm.reshape(-1)).sum()/dm.sum().clamp_min(1))
            dev_acc=float(((z.argmax(-1)==dy).float()*dm).sum()/dm.sum().clamp_min(1))
        out={'schema':'rrc14.weights.v1','kind':kind,'seed':SEED,'steps':steps,
             'train_count':len(groups['train']),'dev_count':len(groups['dev']),
             'trajectories_per_case':2,'teacher_steps':LENGTH,'optimizer':'torch.Adam.fp32',
             'learning_rate':LR,'batch_size':BATCH,'hidden':p.HIDDEN if kind!='feedforward' else p.FF_HIDDEN,
             'parameters':sum(z.numel() for z in model.parameters()),
             'source_hashes':source_hashes(),'initializer_sha256':prior_sha,
             'train_roster_sha256':b.digest(groups['train']),
             'dev_roster_sha256':b.digest(groups['dev']),
             'teacher_neighbor_executions':teacher_ops,
             'teacher_future_access':False,'test_generated_during_training':False,
             'development_loss':dev_loss,'development_action_accuracy':dev_acc,
             'torch_version':torch.__version__,
             'state_dict':{k:v.detach().tolist() for k,v in model.state_dict().items()}}
        result[kind]=out
        if verbose:print(kind,'params',out['parameters'],'dev loss',round(dev_loss,4),
                         'action acc',round(dev_acc,3),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out-dir',type=Path,default=HERE/'results')
    parser.add_argument('--steps',type=int,default=STEPS)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    models=fit(steps=args.steps)
    for k,model in models.items():
        path=args.out_dir/f'rrc14_{k}.json'
        path.write_text(json.dumps(model,sort_keys=True,indent=2)+'\n')
        print('wrote',path)

if __name__=='__main__':main()
