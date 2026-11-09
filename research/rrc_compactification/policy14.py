"""RRC-14 portable GRU and matched feedforward action logits.

Every policy input is 132 public features from a currently evaluated graph.
An action selects one of nine reversible single-token edits. No graph
interpreter or future target is accessible to the policy head.
"""
from __future__ import annotations
import math
try:
    from . import graph14 as b
except ImportError:
    import graph14 as b

FEATURES=132
HIDDEN=16
FF_HIDDEN=52
ACTIONS=9


def sigmoid(x):
    if x>=0:
        v=math.exp(-x);return 1/(1+v)
    v=math.exp(x);return v/(1+v)


def affine(w,b,x):
    return [t+sum(a*v for a,v in zip(row,x)) for row,t in zip(w,b)]


def step(x,checkpoint,h=None):
    if len(x)!=FEATURES:raise ValueError('invalid input feature shape')
    kind=checkpoint['kind'];s=checkpoint['state_dict']
    if kind in ('recurrent','shuffled'):
        if h is None:h=[0.0]*HIDDEN
        ix=affine(s['cell.weight_ih'],s['cell.bias_ih'],x)
        hx=affine(s['cell.weight_hh'],s['cell.bias_hh'],h)
        nh=[]
        for i in range(HIDDEN):
            r=sigmoid(ix[i]+hx[i])
            z=sigmoid(ix[i+HIDDEN]+hx[i+HIDDEN])
            n=math.tanh(ix[i+2*HIDDEN]+r*hx[i+2*HIDDEN])
            nh.append((1-z)*n+z*h[i])
        return affine(s['head.weight'],s['head.bias'],nh),nh
    if kind=='feedforward':
        z=[math.tanh(v) for v in affine(s['fc1.weight'],s['fc1.bias'],x)]
        return affine(s['head.weight'],s['head.bias'],z),None
    raise ValueError('unknown policy kind')


def mac_proxy(kind,visits):
    if kind in ('recurrent','shuffled'):
        return visits*(3*HIDDEN*(FEATURES+HIDDEN)+ACTIONS*HIDDEN)
    if kind=='feedforward':return visits*(FEATURES*FF_HIDDEN+FF_HIDDEN*ACTIONS)
    raise ValueError('unsupported kind')


def ranked_neighbors(graph,logits):
    if len(logits)!=ACTIONS:raise ValueError('wrong action count')
    return sorted(((float(-logits[i]),i,child) for i,(child,_,_) in enumerate(b.neighborhood(graph))))
