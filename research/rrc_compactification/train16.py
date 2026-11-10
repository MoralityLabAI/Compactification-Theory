"""RRC-16 CPU fine-tuning on fork, cascade and additive graph-edit traces.

The Hadamard join, its test cases and future labels are never generated in
training. The prior RRC-14 checkpoints and frozen RRC-13 initializer are pinned.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import topology16 as b, policy16 as p, graph14 as old, train14
except ImportError:
    import topology16 as b
    import policy16 as p
    import graph14 as old
    import train14

HERE=Path(__file__).resolve().parent
STEPS=260
BATCH=64
LR=0.002
EPISODE_STEPS=4


def source_hashes():
    names=('topology16.py','policy16.py','train16.py','topology15.py','graph11.py',
           'graph12.py','graph13.py','graph14.py','policy13.py','policy14.py','train14.py')
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def prior_models():
    try:
        from .bench14 import checkpoints
    except ImportError:
        from bench14 import checkpoints
    models,digests,initializer=checkpoints()
    return models,digests,initializer


def trajectories(cases,initializer,seed,*,shuffled=False):
    """Teacher best-neighbor actions, using train-only candidate executions.

    Two trajectories per case, from frozen-RRC13 and a corrupt known graph.
    No interpreter output is supplied for unevaluated candidates at inference.
    """
    rng=random.Random(seed)
    rows=[]; queries=0
    for case in cases:
        if case.topology not in b.TRAIN_TOPOLOGIES:raise AssertionError('held-out join in training')
        for version in range(2):
            if version==0:current=train14.start_graph(case,initializer)
            else:
                graph=list(case.hidden_graph)
                for pos in rng.sample(range(3),2):
                    graph[pos]=rng.choice([q for q in b.g.OPS if q!=graph[pos]])
                current=tuple(graph)
            obs=[];actions=[];mask=[]
            for _ in range(EPISODE_STEPS):
                signal=b.feedback(case,current);queries+=1
                obs.append(b.features(case,current,signal))
                if signal['accepted']:
                    actions.append(0);mask.append(0)
                else:
                    best=None
                    for index,(candidate,_,_) in enumerate(old.neighborhood(current)):
                        reply=b.feedback(case,candidate);queries+=1
                        entry=(reply['rank'],-index,candidate)
                        if best is None or entry>best:best=entry
                    actions.append(-best[1]);mask.append(1)
                    current=best[2]
            rows.append([obs,actions,mask])
    if shuffled:
        nonpadding=[(i,j) for i,(_,_,mask) in enumerate(rows) for j,flag in enumerate(mask) if flag]
        values=[rows[i][1][j] for i,j in nonpadding]
        rng.shuffle(values)
        for (i,j),value in zip(nonpadding,values):rows[i][1][j]=value
    return rows,queries


def modules():
    import torch
    from torch import nn
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell=nn.GRUCell(p.FEATURES,p.HIDDEN)
            self.head=nn.Linear(p.HIDDEN,p.ACTIONS)
        def forward(self,inputs):
            state=torch.zeros((len(inputs),p.HIDDEN),dtype=inputs.dtype,device=inputs.device)
            output=[]
            for j in range(inputs.shape[1]):
                state=self.cell(inputs[:,j,:],state)
                output.append(self.head(state))
            return torch.stack(output,dim=1)
    class Feedforward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(p.FEATURES,p.FF_HIDDEN)
            self.head=nn.Linear(p.FF_HIDDEN,p.ACTIONS)
        def forward(self,inputs):
            return self.head(torch.tanh(self.fc1(inputs)))
    return {'recurrent':Recurrent,'feedforward':Feedforward,'shuffled':Recurrent}


def preinitialize(net, checkpoint):
    """Preserve the RRC-14 132-feature weights; zero only new feature columns."""
    import torch
    old_state=checkpoint['state_dict']
    with torch.no_grad():
        for name,tensor in net.state_dict().items():
            prior=torch.tensor(old_state[name],dtype=tensor.dtype)
            if name in ('cell.weight_ih','fc1.weight'):
                if tensor.shape[0]!=prior.shape[0] or prior.shape[1]!=132 or tensor.shape[1]!=p.FEATURES:
                    raise AssertionError('unsupported pretraining shape')
                tensor[:,:132].copy_(prior)
                tensor[:,132:].zero_()
            else:
                if tensor.shape!=prior.shape:raise AssertionError('pretrained tensor shape drift')
                tensor.copy_(prior)


def fit(*,steps=STEPS,train_per=b.TRAIN_PER_TOPOLOGY,dev_per=b.DEV_PER_TOPOLOGY,verbose=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models,digests,initializer=prior_models()
    groups=b.roster(include_test=False,train_per=train_per,dev_per=dev_per)
    assert all(c.topology in b.TRAIN_TOPOLOGIES for cases in groups.values() for c in cases)
    train,train_queries=trajectories(groups['train'],initializer,b.SEED+1000)
    dev,dev_queries=trajectories(groups['dev'],initializer,b.SEED+2000)
    shuffled,_=trajectories(groups['train'],initializer,b.SEED+1000,shuffled=True)
    def tensors(rows):
        return (torch.tensor([r[0] for r in rows],dtype=torch.float32),
                torch.tensor([r[1] for r in rows],dtype=torch.long),
                torch.tensor([r[2] for r in rows],dtype=torch.float32))
    tx,ty,tm=tensors(train);dx,dy,dm=tensors(dev);sx,sy,sm=tensors(shuffled)
    assert torch.equal(tx,sx) and torch.equal(tm,sm)
    rng=random.Random(b.SEED+333)
    batches=[[rng.randrange(len(train)) for _ in range(BATCH)] for _ in range(steps)]
    outputs={}
    classes=modules()
    for offset,kind in enumerate(('recurrent','feedforward','shuffled')):
        torch.manual_seed(b.SEED+3000+offset)
        net=classes[kind]()
        base=models['feedforward' if kind=='feedforward' else 'recurrent']
        preinitialize(net,base)
        optimizer=torch.optim.Adam(net.parameters(),lr=LR)
        inputs,labels,masks=(sx,sy,sm) if kind=='shuffled' else (tx,ty,tm)
        for step,indices in enumerate(batches):
            index=torch.tensor(indices,dtype=torch.long)
            logits=net(inputs[index])
            mask=masks[index]
            loss=(F.cross_entropy(logits.reshape(-1,p.ACTIONS),labels[index].reshape(-1),reduction='none')
                  *mask.reshape(-1)).sum()/mask.sum().clamp_min(1)
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            values=net(dx)
            loss=(F.cross_entropy(values.reshape(-1,p.ACTIONS),dy.reshape(-1),reduction='none')
                  *dm.reshape(-1)).sum()/dm.sum().clamp_min(1)
            accuracy=((values.argmax(-1)==dy).float()*dm).sum()/dm.sum().clamp_min(1)
        checkpoint={'schema':'rrc16.weights.v1','kind':kind,'seed':b.SEED,
                    'optimization_steps':steps,'batch_size':BATCH,'learning_rate':LR,
                    'train_cases':len(groups['train']),'dev_cases':len(groups['dev']),
                    'topologies_trained':list(b.TRAIN_TOPOLOGIES),
                    'heldout_join_in_training':False,'training_examples_per_case':2,
                    'supervision_queries_train':train_queries,
                    'supervision_queries_dev':dev_queries,
                    'source_sha256':source_hashes(),
                    'pretrained_sha256':digests['feedforward' if kind=='feedforward' else 'recurrent'],
                    'initializer_sha256':digests['rrc13_frozen_initializer'],
                    'train_digest':b.digest(groups['train']),'dev_digest':b.digest(groups['dev']),
                    'parameter_count':sum(v.numel() for v in net.parameters()),
                    'pretrained_132_feature_weights':True,
                    'topology_features':list(b.topology_descriptor('hadamard')),
                    'development_teacher_action_loss':float(loss.item()),
                    'development_teacher_action_accuracy':float(accuracy.item()),
                    'torch_version':torch.__version__,
                    'state_dict':{name:t.detach().tolist() for name,t in net.state_dict().items()}}
        if checkpoint['parameter_count']!=p.parameter_count(kind):raise AssertionError('param accounting')
        outputs[kind]=checkpoint
        if verbose:print(kind,'params',checkpoint['parameter_count'],'dev loss',round(loss.item(),4),
                         'dev teacher action accuracy',round(accuracy.item(),4),flush=True)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'results')
    parser.add_argument('--steps',type=int,default=STEPS)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for kind,ck in fit(steps=args.steps).items():
        path=args.out_dir/f'rrc16_{kind}.json'
        path.write_text(json.dumps(ck,sort_keys=True,indent=2)+'\n')
        print('saved',path)
if __name__=='__main__':main()
