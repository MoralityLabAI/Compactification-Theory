"""SHAFT-COMP-23: information-matched, behavioral-only decoder comparison.

Frozen preregistration: SHAFT_COMP_23_PROTOCOL.md, commit fae6c5c6843ee8f2a8f85686012ee559dc37d281.
All models are small in-memory synthetic GRUs with known training goals. Decoder
fitting receives ONLY native frozen-model two-query behavioral bits; all
canonical task-state labels are withheld until posthoc evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.shaft_compactification.experiments import architecture_vocab_transfer as base
from research.shaft_compactification.experiments import aux_latch_learning as aux
from research.shaft_compactification.experiments import factorized_decoder as prev

REGISTRATION = 'fae6c5c6843ee8f2a8f85686012ee559dc37d281'
SEEDS = {
    'repeat_toggle': {'calibration':(2301,), 'heldout':(2302,2303)},
    'successor_toggle': {'calibration':(2311,), 'heldout':(2312,2313)},
}
V = 6
GOALS = 64
HIDDEN = 24
PROBE_DEPTHS=(4,8,12)
TEST_DEPTHS=(12,40)
CONTEXTS_PER_GOAL=12
EVAL_PER_GOAL=16
DECODER_STEPS=600
BASE_STEPS=3200
LR=.04
GATE=.90


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def states(model,family,seed,depth,per_goal,training):
    """Independent frozen-model states. Labels g,l,p returned for audit ONLY."""
    goals=torch.arange(GOALS).repeat_interleave(per_goal)
    source=(23000000+seed*101+depth*131+(2011 if training else 901107))
    gen=torch.Generator().manual_seed(source)
    goals=goals[torch.randperm(len(goals),generator=gen)]
    with torch.no_grad():
        state, latch, previous = prev.comp20.prefixes(model, goals, depth,
                 source+(301007 if training else 602011),family)
    return state.detach(),goals.detach(),torch.from_numpy(latch.copy()).long(),torch.from_numpy(previous.copy()).long()


class Factorized900(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.Tanh(),nn.Linear(HIDDEN,12))

    def raw(self,state):
        x=self.layers(state)
        return x[:,:6], x[:,6:]

    def forward(self,state,family):
        goal_logits, previous_logits=self.raw(state)
        s=torch.sigmoid(goal_logits)[:,None,:]  # [n,1,6]
        previous_probs=torch.softmax(previous_logits,dim=1)
        if family=='repeat_toggle':
            trigger=previous_probs
        elif family=='successor_toggle':
            trigger=previous_probs.roll(shifts=1,dims=1)
        else: raise ValueError(family)
        t=trigger[:,:,None]  # [n,6,1] (first query a)
        probability=s*(1-t)+(1-s)*t
        return probability.reshape(len(state),V*V).clamp(1e-7,1-1e-7)

    @torch.no_grad()
    def latent(self,state):
        goal,prev_logits=self.raw(state)
        return goal>=0,prev_logits.argmax(dim=1)


def create_heads(seed):
    result={}
    for idx,name in enumerate(('direct36','factorized900','directmlp890')):
        torch.manual_seed(230000+seed*100+idx*733)
        if name=='direct36':head=nn.Linear(HIDDEN,V*V)
        elif name=='factorized900':head=Factorized900()
        else:head=nn.Sequential(nn.Linear(HIDDEN,14),nn.Tanh(),nn.Linear(14,V*V))
        result[name]=head
    assert sum(p.numel() for p in result['direct36'].parameters())==900
    assert sum(p.numel() for p in result['factorized900'].parameters())==900
    assert sum(p.numel() for p in result['directmlp890'].parameters())==890
    return result


def train_heads(model,family,seed,steps=DECODER_STEPS,per_goal=CONTEXTS_PER_GOAL):
    xs=[]
    for depth in PROBE_DEPTHS:
        x,_,_,_=states(model,family,seed,depth,per_goal,True)
        xs.append(x)
    x=torch.cat(xs)
    with torch.no_grad():
        target=(base.counterfactual(model,x,2)>=0).float()
    result=create_heads(seed)
    losses={}
    for name,head in result.items():
        opt=torch.optim.Adam(head.parameters(),lr=LR)
        for step in range(steps):
            if name=='factorized900':
                loss=F.binary_cross_entropy(head(x,family),target)
            else:
                loss=F.binary_cross_entropy_with_logits(head(x),target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        losses[name]=float(loss.detach())
    return result,losses,len(x)


@torch.no_grad()
def evaluate_cell(model,heads,family,seed,depth,per_goal=EVAL_PER_GOAL):
    x,g,l,p=states(model,family,seed,depth,per_goal,False)
    native=base.counterfactual(model,x,2)>=0
    exact=prev.predicted_two(prev.bits_from_goal(g).bool(),p,family,latch=l)
    predicted={
        'direct36':heads['direct36'](x)>=0,
        'factorized900':heads['factorized900'](x,family)>=.5,
        'directmlp890':heads['directmlp890'](x)>=0,
        'oracle':exact,
    }
    effective_pred,prev_pred=heads['factorized900'].latent(x)
    effective_true=prev.bits_from_goal(g).bool() ^ l.bool()[:,None]
    diagnostics={
        'latent_effective_bit_correct':(effective_pred==effective_true).float().mean().item(),
        'latent_effective_exact_correct':(effective_pred==effective_true).all(dim=1).float().mean().item(),
        'latent_previous_class_correct':(prev_pred==p).float().mean().item(),
        'oracle_native_bit_correct':(exact==native).float().mean().item(),
        'oracle_native_exact_correct':(exact==native).all(dim=1).float().mean().item(),
    }
    metrics={}
    records=[]
    for name,pred in predicted.items():
        match_native=pred==native
        match_exact=pred==exact
        metrics[name]={
            'native_bit_accuracy':float(match_native.float().mean()),
            'native_exact_accuracy':float(match_native.all(dim=1).float().mean()),
            'automaton_bit_accuracy':float(match_exact.float().mean()),
            'automaton_exact_accuracy':float(match_exact.all(dim=1).float().mean()),
            'native_bits_correct':int(match_native.sum()),
            'native_exact_correct':int(match_native.all(dim=1).sum()),
        }
    for i in range(len(g)):
        rec={'seed':int(seed),'family':family,'depth':int(depth),'index':int(i),
             'goal_audit_only':int(g[i]),'latch_audit_only':int(l[i]),'previous_audit_only':int(p[i]),
             'effective_exact_audit_only':int((effective_pred[i]==effective_true[i]).all()),
             'previous_correct_audit_only':int(prev_pred[i]==p[i]),
             'oracle_native_exact':int((exact[i]==native[i]).all())}
        for name,pred in predicted.items():
            rec[f'{name}_native_bits_correct']=int((pred[i]==native[i]).sum())
            rec[f'{name}_native_exact']=int((pred[i]==native[i]).all())
            rec[f'{name}_automaton_bits_correct']=int((pred[i]==exact[i]).sum())
            rec[f'{name}_automaton_exact']=int((pred[i]==exact[i]).all())
        records.append(rec)
    return {'seed':seed,'family':family,'depth':depth,'states':len(g),
            'methods':metrics,'diagnostics':diagnostics},records


def base_train(family,seed,steps=BASE_STEPS,show_every=800):
    model,latch_head,previous_head,losses=aux.train_one(family,seed,'aux_state',steps=steps,show_every=show_every)
    score12=aux.evaluate(model,family,seed,12)
    score32=aux.evaluate(model,family,seed,32)
    receipt={
        'family':family,'seed':seed,'arm':'aux_state','train_steps':steps,
        'model_sha256':base.stable_sha(model),
        'train_loss':losses,
        'task_accuracy_d12':score12['bit_accuracy'],'task_accuracy_d32':score32['bit_accuracy'],
        'gate_passed':score12['bit_accuracy']>=GATE,
        'train_only_latch_head_sha256':base.stable_sha(latch_head),
        'train_only_prev_head_sha256':base.stable_sha(previous_head),
        'aux_state_accuracy':aux.aux_accuracy(model,latch_head,previous_head,family,seed),
    }
    return model,latch_head,previous_head,receipt


def run_one(family,seed,phase,steps=BASE_STEPS,probe_steps=DECODER_STEPS,per_goal=EVAL_PER_GOAL,show_every=800):
    if phase != 'engineering':
        if seed not in SEEDS[family][phase]:raise ValueError('unregistered model seed')
        if steps != BASE_STEPS or probe_steps != DECODER_STEPS or per_goal != EVAL_PER_GOAL:
            raise ValueError('registered test requires original steps and evaluation count')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    model,latch,previous,br=base_train(family,seed,steps,show_every)
    result={
        'experiment':'SHAFT-COMP-23','registration_commit':REGISTRATION,
        'torch':str(torch.__version__),'numpy':str(np.__version__),
        'phase':phase,'family':family,'seed':seed,'base':br,'probe':None,
        'cells':[],'rows':[],
    }
    if br['gate_passed']:
        heads,losses,ntrain=train_heads(model,family,seed,probe_steps)
        result['probe']={
            'train_examples':ntrain,'training_steps':probe_steps,'losses':losses,
            'heads':{k:{'parameters':sum(p.numel() for p in h.parameters()),
                        'sha256':base.stable_sha(h)} for k,h in heads.items()},
            'supervision':'the SAME frozen native two-query binary responses for all three heads',
        }
        for depth in TEST_DEPTHS:
            m,rows=evaluate_cell(model,heads,family,seed,depth,per_goal)
            result['cells'].append(m)
            result['rows'].extend(rows)
            print(json.dumps({'seed':seed,'family':family,'phase':phase,
                'depth':depth,'native_bit_accuracy':{k:round(v['native_bit_accuracy'],4)
                for k,v in m['methods'].items()},'exact':{k:round(v['native_exact_accuracy'],4)
                for k,v in m['methods'].items()},'diag':m['diagnostics']}),flush=True)
    return result, (model,latch,previous)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=sorted(SEEDS),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--phase',choices=('calibration','heldout','engineering'),required=True)
    parser.add_argument('--steps',type=int,default=BASE_STEPS)
    parser.add_argument('--probe-steps',type=int,default=DECODER_STEPS)
    parser.add_argument('--per-goal',type=int,default=EVAL_PER_GOAL)
    parser.add_argument('--show-every',type=int,default=800)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--calibration-freeze',type=Path)
    parser.add_argument('--freeze-sha256',type=str)
    args=parser.parse_args()
    if args.phase=='heldout':
        if not args.calibration_freeze or not args.freeze_sha256:
            raise ValueError('heldout requires preregistered calibration freeze and SHA256')
        if sha256_file(args.calibration_freeze)!=args.freeze_sha256:
            raise ValueError('calibration freeze SHA256 mismatch')
        data=json.loads(args.calibration_freeze.read_text())
        if data.get('registration_commit')!=REGISTRATION or data.get('experiment')!='SHAFT-COMP-23':
            raise ValueError('calibration freeze is not from this registered protocol')
    result,(base_model,latch,prevhead)=run_one(args.family,args.seed,args.phase,args.steps,args.probe_steps,args.per_goal,args.show_every)
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload,end='')
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'base':base_model.state_dict(),'latch_head':latch.state_dict(),
                    'prev_head':prevhead.state_dict(),'family':args.family,'seed':args.seed},args.checkpoint)

if __name__=='__main__':main()
