"""SHAFT-COMP-21: supervised temporal-state training on six-token latch tasks.

Registration: b4596c1b5c0e4a63fc4dc3a3d03af524ad041f10.
Local synthetic TaskMemoryGRUs only. Full native model retained throughout.
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

V=6
GOALS=64
HIDDEN=24
STEPS=3200
BATCH=128
LENGTH=12
LR=.01
REGISTRATION="b4596c1b5c0e4a63fc4dc3a3d03af524ad041f10"
FAMILIES={
    "repeat_toggle":{"calibration":(2201,),"heldout":(2202,2203)},
    "successor_toggle":{"calibration":(2211,),"heldout":(2212,2213)},
}
ARMS=("output_only","aux_state")


def automaton_labels(goals,tokens,family):
    prev=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),dim=1)
    if family=="repeat_toggle":
        trigger=(tokens==prev).long()
    elif family=="successor_toggle":
        trigger=(tokens==(prev+1)%V).long()
    else:
        raise ValueError("unknown family")
    latch_post=trigger.cumsum(dim=1)%2
    latch_pre=(trigger.cumsum(dim=1)-trigger)%2
    target=(((goals[:,None]>>tokens)&1)^latch_pre).float()
    return target,latch_post.float(),tokens.long()


def step_sequence(model,goals,tokens):
    states=[]
    logits=[]
    h=model.embedding(goals)
    for t in range(tokens.shape[1]):
        x=F.one_hot(tokens[:,t],V).float()
        h=model.step(h,x)
        states.append(h)
        logits.append(model.score(h,x))
    return torch.stack(logits,1),torch.stack(states,1)


def build_frozen_pair(seed):
    torch.manual_seed(seed)
    model=base.RecurrentMemory("gru",V)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+999_001)
        latch_head=nn.Linear(HIDDEN,1)
        prev_head=nn.Linear(HIDDEN,V)
    return model,latch_head,prev_head


def train_one(family,seed,arm,steps=STEPS,show_every=800):
    if family not in FAMILIES or arm not in ARMS:
        raise ValueError("unregistered arm")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    model,latch_head,prev_head=build_frozen_pair(seed)
    if arm=="output_only":
        params=list(model.parameters())
    else:
        params=list(model.parameters())+list(latch_head.parameters())+list(prev_head.parameters())
    opt=torch.optim.Adam(params,lr=LR)
    generator=torch.Generator().manual_seed(10000+seed)
    model.train()
    for iteration in range(steps):
        goals=torch.randint(GOALS,(BATCH,),generator=generator)
        tokens=torch.randint(V,(BATCH,LENGTH),generator=generator)
        target,latch,previous=automaton_labels(goals,tokens,family)
        logits,states=step_sequence(model,goals,tokens)
        task_loss=F.binary_cross_entropy_with_logits(logits,target)
        if arm=="aux_state":
            latch_loss=F.binary_cross_entropy_with_logits(latch_head(states).squeeze(-1),latch)
            prev_loss=F.cross_entropy(prev_head(states).reshape(-1,V),previous.reshape(-1))
            loss=task_loss+latch_loss+.25*prev_loss
        else:
            latch_loss=task_loss.detach().new_zeros(())
            prev_loss=task_loss.detach().new_zeros(())
            loss=task_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params,2.)
        opt.step()
        if show_every and (iteration+1)%show_every==0:
            print(json.dumps({
                "seed":seed,"family":family,"arm":arm,"step":iteration+1,
                "task_loss":round(float(task_loss.detach()),5),
                "latch_loss":round(float(latch_loss.detach()),5),
                "previous_loss":round(float(prev_loss.detach()),5),
            }),flush=True)
    model.eval()
    latch_head.eval()
    prev_head.eval()
    return model,latch_head,prev_head,{
        "task":float(task_loss.detach()),"latch":float(latch_loss.detach()),
        "previous":float(prev_loss.detach()),"total":float(loss.detach())}


@torch.no_grad()
def evaluate(model,family,seed,length,count=1024):
    generator=torch.Generator().manual_seed(9_000_000+seed+43*length)
    goals=torch.randint(GOALS,(count,),generator=generator)
    tokens=torch.randint(V,(count,length),generator=generator)
    target,_,_=automaton_labels(goals,tokens,family)
    logits,_=step_sequence(model,goals,tokens)
    hits=((logits>=0)==target.bool())
    return {"bit_accuracy":float(hits.float().mean()),"bits_correct":int(hits.sum().item()),
            "bits_total":int(count*length),
            "bce":float(F.binary_cross_entropy_with_logits(logits,target))}


@torch.no_grad()
def aux_accuracy(model,latch_head,prev_head,family,seed,length=12,count=512):
    generator=torch.Generator().manual_seed(9_800_000+seed)
    goals=torch.randint(GOALS,(count,),generator=generator)
    tokens=torch.randint(V,(count,length),generator=generator)
    _,latch,previous=automaton_labels(goals,tokens,family)
    _,states=step_sequence(model,goals,tokens)
    return {
        "latch_bit_accuracy":float(((latch_head(states).squeeze(-1)>=0)==latch.bool()).float().mean()),
        "previous_symbol_accuracy":float((prev_head(states).argmax(-1)==previous).float().mean())}


def record(family,seed,phase,arm,steps=STEPS,show_every=800):
    model,latch,prev,loss=train_one(family,seed,arm,steps,show_every)
    result={
        "experiment":"SHAFT-COMP-21","registration_commit":REGISTRATION,
        "family":family,"seed":seed,"phase":phase,"arm":arm,
        "train_steps":steps,"train_loss":loss,
        "accuracy_d12":evaluate(model,family,seed,12),
        "accuracy_d32":evaluate(model,family,seed,32),
        "auxiliary_head_accuracy":aux_accuracy(model,latch,prev,family,seed),
        "model_sha256":base.stable_sha(model),
        "aux_latch_head_sha256":base.stable_sha(latch),
        "aux_previous_head_sha256":base.stable_sha(prev),
        "parameter_count_native":sum(p.numel() for p in model.parameters()),
        "torch":str(torch.__version__),"numpy":str(np.__version__),
    }
    result["probe_gate_passed"]=result["accuracy_d12"]["bit_accuracy"]>=.90
    return result,(model,latch,prev)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",choices=tuple(FAMILIES),required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--phase",choices=("calibration","heldout","engineering"),required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--steps",type=int,default=STEPS)
    parser.add_argument("--show-every",type=int,default=800)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--gate-file",type=Path)
    parser.add_argument("--gate-sha256")
    a=parser.parse_args()
    if a.phase!="engineering":
        if a.seed not in FAMILIES[a.family][a.phase]:
            raise ValueError("unregistered model seed")
        if a.steps!=STEPS:
            raise ValueError("registered models require 3200 steps")
    if a.phase=="heldout":
        if not a.gate_file or not a.gate_sha256:
            raise ValueError("heldout requires frozen calibration gate")
        raw=a.gate_file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=a.gate_sha256:
            raise ValueError("calibration SHA-256 mismatch")
        if json.loads(raw).get("registration_commit")!=REGISTRATION:
            raise ValueError("calibration registration mismatch")
    result,mods=record(a.family,a.seed,a.phase,a.arm,a.steps,a.show_every)
    payload=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(payload)
    else:
        print(payload,end="")
    if a.weights:
        a.weights.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"base":mods[0].state_dict(),"latch_head":mods[1].state_dict(),
                    "prev_head":mods[2].state_dict(),
                    "config":{"family":a.family,"arm":a.arm,"seed":a.seed,"steps":a.steps}},
                   a.weights)


if __name__=="__main__":
    main()
