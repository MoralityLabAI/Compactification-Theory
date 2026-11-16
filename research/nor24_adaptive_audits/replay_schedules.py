"""Independent NOR-24 challenge plans, without importing experiment.py."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from math import comb
import random

F=("tail","magnitude","lag","uniform")
K=F+("off_template",)
A=("uniform","static_prior","posterior","shuffled_prior","oracle_support")

def seed(*fields):
    return int.from_bytes(sha256("|".join(str(x) for x in fields).encode("ascii")).digest(),"big")

def regions(y,header):
    m=len(y)
    t=set(range(m-96,m))
    mag=set(sorted(range(m),key=lambda i:(-abs(y[i]),i))[:96])
    lag=set(range(header.delay+header.spacing,header.delay+header.spacing+96))
    out=dict(tail=t,magnitude=mag,lag=lag,uniform=set(range(m)))
    out["off_template"]=set(range(m))-(t|mag|lag)
    assert len(out["off_template"])>=16
    return out

def attack(digest,kind,r,regions):
    rng=random.Random(seed(202611240072,digest,kind,r))
    return set(rng.sample(sorted(regions[kind]),r))

def plan(regions,prior,digest,trial):
    m=len(regions["uniform"])
    rng=random.Random(seed(202611240073,digest,"uniform",trial))
    q=list(range(m));rng.shuffle(q)
    uniform=tuple(q[:64])

    weight=[0.0]*m
    for kind in F:
        w=prior[kind]/len(regions[kind])
        for i in regions[kind]:weight[i]+=w
    rng=random.Random(seed(202611240073,digest,"static_prior",trial))
    static=tuple(sorted(range(m),key=lambda i:(-weight[i],rng.random()))[:64])

    rng=random.Random(seed(202611240073,digest,"shuffled_prior",trial))
    permutation=list(range(m));rng.shuffle(permutation)
    shuffled=tuple(permutation[i] for i in static)

    rng=random.Random(seed(202611240073,digest,"posterior",trial))
    groups={}
    for i in range(m):
        member=tuple(f for f in F if i in regions[f])
        groups.setdefault(member,[]).append(i)
    for pool in groups.values():rng.shuffle(pool)
    used=Counter({f:0 for f in F})
    denominators={f:comb(len(regions[f]),4) for f in F}
    posterior=[]
    for _ in range(64):
        likelihood={f:prior[f]*comb(len(regions[f])-used[f],4)/denominators[f]
                    for f in F}
        total=sum(likelihood.values())
        best=-1.
        choices=[]
        for member,pool in groups.items():
            if not pool:continue
            s=sum(likelihood[f]*4/(len(regions[f])-used[f]) for f in member)/total
            if s>best+1e-15:
                best=s;choices=[member]
            elif abs(s-best)<=1e-15:
                choices.append(member)
        selected=rng.choice(choices)
        posterior.append(groups[selected].pop())
        for f in selected:used[f]+=1
    out=dict(uniform=uniform,static_prior=static,
             posterior=tuple(posterior),shuffled_prior=shuffled)
    for kind in K:
        rng=random.Random(seed(202611240073,digest,"oracle_support",kind,trial))
        q=sorted(regions[kind]);rng.shuffle(q)
        out[("oracle_support",kind)]=tuple(q[:64])
    return out
