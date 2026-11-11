"""OIC-22 fault-aware 145-param sensor-selection policy, teacher distillation.

Source of teacher actions: exact finite joint Bayes oracle on n=4,5 for equal
mixture clean + one uniformly placed flip on optimal-policy trees.
Off-tree contexts and n=6 use the archived OIC-21 scorer as an additional
privileged teacher. Its earlier training cost is disclosed separately. The
recurrent belief is a supplied exact cyclic hypothesis table.
"""
from __future__ import annotations
import argparse,json,random,time,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from oic22_filter import OnlineFilter, BoundedFlipObserver, archived_scorers, scorer_values
from oic22_exact import MixedFaultOracle

HERE=Path(__file__).resolve().parent
TRAIN_NS=(4,5,6)
STEPS=500;BATCH=96;LR=.01
ALPHA=.5

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(7,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.network(x).squeeze(-1)


def posterior_from_masks(o,t,c0,c1,alpha=ALPHA):
    """Exact weighted posterior under mixture clean vs exactly-one flip."""
    weights0=(1-alpha)+alpha*(o.Q-t)/o.Q
    weights1=alpha/o.Q
    total=o.n*o.n
    w=np.fromiter((weights0 if ((c0>>i)&1) else weights1 if ((c1>>i)&1) else 0.
                   for i in range(total)),count=total,dtype=np.float64)
    if not w.sum():raise AssertionError('impossible history')
    return w/w.sum()



def _optimal_tree_action_labels(o):
    """Certified exact action values only for the complete optimal-policy tree."""
    o.optimal()
    queue=[(0,o.all,0)]
    values={}
    while queue:
        t,m0,m1=queue.pop()
        key=(t,m0,m1)
        if t==o.Q or key in values:continue
        scores=o.action_values(t,m0,m1)
        values[key]=scores
        marker=int(np.argmax(scores))
        for _,a,b in o.successors(m0,m1,t,marker):
            if a|b:queue.append((t+1,a,b))
    return values

def make_contexts(seed,split,episodes_per_n=120):
    rng=random.Random(20261111+seed*919+int(split=='heldout')*2003)
    refs={n:MixedFaultOracle(n,n+2) for n in TRAIN_NS}
    sensors={n:OnlineFilter(n) for n in TRAIN_NS}
    oracle_tree={n:_optimal_tree_action_labels(refs[n]) for n in (4,5)}
    old_teacher=archived_scorers()[(split,seed)]
    contexts=[];stats={}
    for n in TRAIN_NS:
        o=refs[n];sensor=sensors[n]
        phases=list(range(n)) if split=='full' else [z for z in range(n) if z not in (1,n-2)]
        seen={}
        for episode in range(episodes_per_n):
            d=rng.randrange(n);z=rng.choice(phases)
            fault=rng.randrange(o.Q) if episode%2 else None
            c0,c1=o.all,0
            mode=episode%4
            for t in range(o.Q):
                key=(t,c0,c1)
                if key not in seen:
                    w=posterior_from_masks(o,t,c0,c1)
                    f=sensor.features(w,t)
                    vals=(oracle_tree[n][key] if n in (4,5) and key in oracle_tree[n] else scorer_values(f,old_teacher).tolist())
                    top=max(vals)
                    seen[key]={'features':f.tolist(),'teacher_values':vals,
                               'optimal':[int(v==top) for v in vals],
                               'n':n,'t':t}
                x=seen[key]
                if mode==0:m=x['teacher_values'].index(max(x['teacher_values']))
                elif mode==1:m=int(np.argmax([row[0]+1e-5*row[1] for row in x['features']]))
                elif mode==2:m=rng.randrange(n)
                else:m=t%n
                y=int((z+d*t)%n==m)
                if t==fault:y=1-y
                c0,c1=o.after(c0,c1,t,m,y)
                if not (c0|c1):raise AssertionError('true hypothesis vanished')
        contexts.extend(seen.values())
        stats[str(n)]={'teacher':'exact_mixed_Bayes_optimal_tree_plus_archived_OIC21_off_tree' if n<=5 else 'archived_OIC21_distillation_teacher',
                       'contexts':len(seen),'episodes':episodes_per_n,
                       'phase_train':phases,'teacher_cache_states':o.optimal_value.cache_info().currsize,'oracle_tree_contexts':len(oracle_tree[n]) if n in (4,5) else 0}
    return contexts,stats


def distill(seed,split,steps=STEPS,episodes_per_n=120):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    start=time.perf_counter()
    contexts,stats=make_contexts(seed,split,episodes_per_n)
    rng=random.Random(221100+seed*439+int(split=='heldout')*13)
    # Pad to 6 sensors, masking invalid columns. Keep exactly 145 params.
    xs=[]; ys=[]; validity=[]
    max_n=6
    for case in contexts:
        n=case['n']
        f=case['features']+[[0.0]*7]*(max_n-n)
        target=np.array(case['optimal'],dtype=np.float32)
        target/=target.sum()
        ys.append(np.pad(target,(0,max_n-n)))
        xs.append(f)
        validity.append([1]*n+[0]*(max_n-n))
    X=torch.tensor(np.array(xs),dtype=torch.float32)
    Y=torch.tensor(np.array(ys),dtype=torch.float32)
    valid=torch.tensor(np.array(validity),dtype=torch.bool)
    model=ScoreNet()
    optim=torch.optim.Adam(model.parameters(),lr=LR)
    losses=[]
    for step in range(steps):
        idx=torch.tensor([rng.randrange(len(X)) for _ in range(BATCH)],dtype=torch.int64)
        logits=model(X[idx]).masked_fill(~valid[idx],-1e9)
        loss=-(Y[idx]*F.log_softmax(logits,dim=-1)).sum(dim=1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optim.step()
        if step in (0,steps//2,steps-1):losses.append(float(loss.detach().item()))
    record={'seed':seed,'split':split,'train_steps':steps,'batch':BATCH,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'learning_rate':LR,'alpha':ALPHA,'teacher':stats,'contexts':len(contexts),
            'losses':losses,'elapsed_sec':round(time.perf_counter()-start,2)}
    weights={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}
    return weights,record


def evaluate(weights,n,alpha=ALPHA,include_faults=True):
    observer=BoundedFlipObserver(n,weights,alpha=alpha)
    return observer.panels(kinds=('clean','flip','erase') if include_faults else ('clean',))


def one_fit(seed,split,steps=STEPS,episodes_per_n=120):
    weights,record=distill(seed,split,steps,episodes_per_n)
    evals={}
    for n in (4,5,6,7,8):
        evals[str(n)]=evaluate(weights,n,include_faults=n in (4,6,8))
    return {'meta':record,'weights':weights,'evaluation':evals}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=('full','heldout'),required=True)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--episodes-per-n',type=int,default=120)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    fit=one_fit(a.seed,a.split,a.steps,a.episodes_per_n)
    fit['software']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(fit,sort_keys=True,indent=2)+'\n')
    print('saved',a.out,fit['meta'],flush=True)
    for n,vals in fit['evaluation'].items():print(n,[(p['kind'],round(p['accuracy'],4)) for p in vals],flush=True)

if __name__=='__main__':main()
