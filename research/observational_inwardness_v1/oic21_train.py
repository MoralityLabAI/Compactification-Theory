"""OIC-21: teacher-distilled adaptive sensor policy with exact posterior.

Structure-assisted design: the action-value teacher and belief updates enumerate
all (d,z0) candidates, with no true hidden phase supplied at inference. The
neural policy only scores candidate *sensor positions*. The supervised teacher
is a privileged training signal whose compute cost is included in receipts.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from oic21_exact import ExactActiveOracle
from oic21_policy import feature_matrix, greedy_policy, evaluate_policy

TR_N=(4,5,6)
TR_STEPS=500
BATCH=96
LR=0.01
FEAT=7

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(FEAT,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.network(x).squeeze(-1)


def qa_oracle(o,mask,t):
    return [o.best_adaptive(yes,t+1)+o.best_adaptive(no,t+1)
            for m in range(o.n) for yes,no in (o.split(mask,t,m),)]


def make_teacher_contexts(seed:int,split:str,episodes_per_n:int=120):
    if split not in ('full','heldout'):raise ValueError('bad split')
    generator=random.Random(18000+seed*113+int(split=='heldout')*1003)
    examples=[]
    stats={}
    for n in TR_N:
        Q=n+2
        o=ExactActiveOracle(n,Q)
        start=time.perf_counter()
        optimum=o.adaptive_accuracy()  # cache exact teacher recurrence for later labeling
        phases=list(range(n)) if split=='full' else [z for z in range(n) if z not in (1,n-2)]
        reachable={}
        for episode in range(episodes_per_n):
            d=generator.randrange(n);z=generator.choice(phases)
            mask=o.all
            policy_id=episode%4
            for t in range(Q):
                key=(mask,t)
                if key not in reachable:
                    qs=qa_oracle(o,mask,t)
                    best=max(qs)
                    f=feature_matrix(o,mask,t)
                    reachable[key]={'n':n,'t':t,'features':[list(row) for row in f],
                                    'optimal':[int(x==best) for x in qs],
                                    'teacher_values':qs}
                if policy_id==0:
                    m=generator.randrange(n)
                elif policy_id==1:
                    m=greedy_policy(o,mask,t)
                elif policy_id==2:
                    m=max(range(n),key=lambda j:(reachable[key]['teacher_values'][j],-j))
                else:
                    m=t%n
                hit,no=o.split(mask,t,m)
                mask=hit if (z+d*t)%n==m else no
                if not mask:raise AssertionError('empty posterior given true hypothesis')
        examples.extend(reachable.values())
        stats[str(n)]={'horizon':Q,'oracle_accuracy':optimum,
                       'teacher_contexts':len(reachable),'oracle_states':o.best_adaptive.cache_info().currsize,
                       'teacher_seconds':round(time.perf_counter()-start,3)}
    return examples,stats


def distill(seed:int,split:str,steps:int=TR_STEPS,episodes_per_n:int=120):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng=random.Random(223300+seed*919+int(split=='heldout')*17)
    begin=time.perf_counter()
    data,teacher_stats=make_teacher_contexts(seed,split,episodes_per_n)
    model=ScoreNet()
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    losses=[]
    for step in range(steps):
        batch=[data[rng.randrange(len(data))] for _ in range(BATCH)]
        # ragged set of sensor positions; process contexts separately
        loss=0
        for example in batch:
            x=torch.tensor(example['features'],dtype=torch.float32)
            logits=model(x)
            target=torch.tensor(example['optimal'],dtype=torch.float32)
            target=target/target.sum()
            loss=loss-torch.sum(target*F.log_softmax(logits,dim=0))
        loss=loss/BATCH
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if step in (0,steps//2,steps-1):losses.append(round(float(loss.detach().item()),6))
    return model,{'seed':seed,'split':split,'parameter_count':sum(p.numel() for p in model.parameters()),
                  'training_steps':steps,'batch_size':BATCH,'learning_rate':LR,'training_examples':len(data),
                  'loss_first_mid_last':losses,'teacher':teacher_stats,
                  'elapsed_seconds':round(time.perf_counter()-begin,3)}


def model_policy(model):
    def query(o,mask,t):
        vals=torch.tensor(feature_matrix(o,mask,t),dtype=torch.float32)
        with torch.no_grad():s=model(vals)
        return int(torch.argmax(s).item())
    return query


def learned_evaluation(model,n,horizon):
    return evaluate_policy(n,horizon,model_policy(model))


def run(seed,split,episodes_per_n=120,steps=TR_STEPS):
    model,meta=distill(seed,split,episodes_per_n=episodes_per_n,steps=steps)
    evalu=[]
    for n in (4,5,6,7,8,12,16,24,32):
        horizon=n+2
        data=learned_evaluation(model,n,horizon)
        evalu.append(data)
        print('fit',seed,split,'n',n,'acc',round(data['shift_accuracy'],4),'leaves',data['leaves'],flush=True)
    state={key:value.detach().cpu().tolist() for key,value in model.state_dict().items()}
    return {'metadata':meta,'evaluation':evalu,'weights':state}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--split',choices=('full','heldout'),default='full')
    parser.add_argument('--steps',type=int,default=TR_STEPS)
    parser.add_argument('--episodes-per-n',type=int,default=120)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    x=run(a.seed,a.split,a.episodes_per_n,a.steps)
    x['software']={'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print('saved',a.out,'seconds',x['metadata']['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
