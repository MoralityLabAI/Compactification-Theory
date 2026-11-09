"""RRC-10: learn a NEW finite multiplier from demonstrations of its I/O behavior.

The hidden operator is T_H(F)=F*H, where H is an unknown formal polynomial.
The model is given paired raw input/output demos (no H or operator tag).
The public architecture supplies coefficient convolution and its triangular
causal state. It learns only two integer update coefficients from training.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Sequence

SEED=2026110910
DEMO=24
HORIZON=48
TRAIN_K=tuple(range(1,9))
TEST_K=(9,10,11,12)
WRAPPERS=('I','N','R','S')


def mul(a:Sequence[int],b:Sequence[int])->tuple[int,...]:
    if len(a)!=len(b):raise ValueError('length mismatch')
    t=len(a)-1
    out=[0]*(t+1)
    for i,v in enumerate(a):
        if v:
            for j in range(t+1-i):out[i+j]+=v*b[j]
    return tuple(out)


def invert(f:Sequence[int])->tuple[int,...]:
    if not f or f[0]!=1:raise ValueError('unit constant required')
    h=[1]
    for n in range(1,len(f)):
        h.append(-sum(f[j]*h[n-j] for j in range(1,n+1)))
    return tuple(h)


def apply_wrapper(f:Sequence[int],op:str)->tuple[int,...]:
    if op=='I':return tuple(f)
    if op=='N':return tuple((-1 if n%2 else 1)*v for n,v in enumerate(f))
    if op=='R':return invert(f)
    if op=='S':return tuple(f[n//2] if n%2==0 else 0 for n in range(len(f)))
    raise ValueError('unknown wrapper')


def source(rng:random.Random)->tuple[int,...]:
    # Integer series units independent of their secret multiplier.
    a=[1]+[rng.randint(-2,3) for _ in range(HORIZON)]
    return tuple(a)


def multiplier(k:int,kind:str,*,late:bool=False)->tuple[int,...]:
    if not 1<=k<=HORIZON:raise ValueError('k out of range')
    h=[1]+[0]*HORIZON
    h[k]+=1
    if kind=='single':pass
    elif kind=='double':
        if 2*k>HORIZON:raise ValueError('support beyond horizon')
        h[2*k]+=1
    elif kind=='signed':
        if 2*k>HORIZON:raise ValueError('support beyond horizon')
        h[k]=-1;h[2*k]+=1
    elif kind=='triple':
        if 2*k>HORIZON:raise ValueError('support beyond horizon')
        h[2*k]+=2
    else:raise ValueError('kind')
    if late:h[DEMO+5]+=1
    return tuple(h)


@dataclass(frozen=True)
class Episode:
    id:str
    split:str
    kind:str
    k:int
    wrapper:str
    demos:tuple[tuple[tuple[int,...],tuple[int,...]],...]
    query:tuple[int,...]
    future:tuple[int,...]
    pair:str|None=None


def episodes(*,include_test:bool=True)->dict[str,list[Episode]]:
    groups={'train':[],'dev':[]}
    if include_test:groups.update({'new_multiplier':[],'new_composition':[],'late_collision':[],'conflict':[]})
    def make(split,index,k,kind,wrapper='I',late=False,pair=None,conflict=False):
        rng=random.Random(SEED+{'train':17,'dev':27,'new_multiplier':47,'new_composition':57,'late_collision':67,'conflict':77}[split]*100000+index)
        h=multiplier(k,kind,late=late)
        demos=tuple((f:=source(rng),mul(f,h)) for _ in range(2))
        if conflict:
            # Corrupt the second demo at a visible degree; no common multiplier exists.
            d=list(demos[1][1]);d[2]+=1
            demos=(demos[0],(demos[1][0],tuple(d)))
        x=source(rng);future=apply_wrapper(mul(x,h),wrapper)
        return Episode(f'{split}-{index:03d}',split,kind,k,wrapper,demos,x,future,pair)
    for i in range(128):
        groups['train'].append(make('train',i,TRAIN_K[i%len(TRAIN_K)],'single'))
    for i in range(24):
        groups['dev'].append(make('dev',i,TRAIN_K[i%len(TRAIN_K)],'single',WRAPPERS[i%len(WRAPPERS)]))
    if not include_test:return groups
    for i in range(48):
        groups['new_multiplier'].append(make('new_multiplier',i,TEST_K[i%4],('double','signed','triple')[i%3]))
        groups['new_composition'].append(make('new_composition',i,TEST_K[i%4],('double','signed','triple')[i%3],WRAPPERS[1+(i%3)]))
    for j in range(8):
        a=make('late_collision',j,TEST_K[j%4],('double','signed')[j%2],pair=f'collision-{j}')
        rng_h=multiplier(a.k,a.kind,late=True)
        # Clone identical visible demos and query, replace the hidden continuation.
        b=Episode('late_collision-'+str(j+8),a.split,a.kind,a.k,a.wrapper,a.demos,a.query,apply_wrapper(mul(a.query,rng_h),a.wrapper),a.pair)
        groups['late_collision'] += [a,b]
    for i in range(16):
        groups['conflict'].append(make('conflict',i,TEST_K[i%4],'double',conflict=True))
    return groups


def recovery(demo:tuple[Sequence[int],Sequence[int]],weights:Sequence[int],degree:int=DEMO)->tuple[int,...]:
    """Two learned gains in supplied causal basis: h[n]=u*g[n]+v*Σf[j]h[n-j]."""
    if len(weights)!=2:raise ValueError('two coefficients required')
    f,g=demo
    if len(f)<degree+1 or len(g)<degree+1 or f[0]!=1 or g[0]!=1:raise ValueError('bad demo')
    h=[1]
    for n in range(1,degree+1):
        h.append(weights[0]*g[n]+weights[1]*sum(f[j]*h[n-j] for j in range(1,n+1)))
    return tuple(h)


def local_recovery(demo:tuple[Sequence[int],Sequence[int]],weights:Sequence[int],degree:int=DEMO)->tuple[int,...]:
    f,g=demo
    return tuple([1]+[weights[0]*g[n]+weights[1]*f[n] for n in range(1,degree+1)])


def propose(episode:Episode,weights:Sequence[int],*,local:bool=False)->dict:
    func=local_recovery if local else recovery
    inferred=[func(d,weights) for d in episode.demos]
    if inferred[0]!=inferred[1]:
        return {'accepted':False,'reason':'demo_disagreement','first_fault':next(n for n in range(DEMO+1) if inferred[0][n]!=inferred[1][n])}
    h=inferred[0]+(0,)*(HORIZON-DEMO)
    if any(mul(f[:DEMO+1],h[:DEMO+1])!=g[:DEMO+1] for f,g in episode.demos):
        return {'accepted':False,'reason':'demo_verifier_rejection','first_fault':None}
    predicted=apply_wrapper(mul(episode.query,h),episode.wrapper)
    return {'accepted':True,'reason':'finite_demo_verified','first_fault':None,'prediction':predicted,'operator':h}


def reference(episode:Episode)->dict:
    return propose(episode,(1,-1))


def digest(groups:dict[str,list[Episode]])->dict[str,str]:
    def record(e):return [e.id,e.split,e.kind,e.k,e.wrapper,e.demos,e.query,e.future,e.pair]
    return {name:sha256(json.dumps([record(e) for e in group],separators=(',',':')).encode()).hexdigest() for name,group in groups.items()}
