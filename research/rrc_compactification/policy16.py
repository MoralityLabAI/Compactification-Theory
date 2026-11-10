"""RRC-16 portable topology-conditioned edit policies and exact parameter counts.

The network sees only candidate feedback already paid for, the current
three-token graph, the demo coefficient prefixes, and public topology features.
It cannot call an unseen-graph interpreter or inspect query futures.
"""
from __future__ import annotations
import math
try:
    from . import topology16 as b
except ImportError:
    import topology16 as b

FEATURES=135
HIDDEN=16
FF_HIDDEN=52
ACTIONS=9


def affine(weight,bias,vector):
    return [v+sum(a*x for a,x in zip(row,vector)) for row,v in zip(weight,bias)]


def sigmoid(x):
    if x >= 0:
        e=math.exp(-x);return 1/(1+e)
    e=math.exp(x);return e/(1+e)


def step(vector,checkpoint,hidden=None):
    if len(vector)!=FEATURES:raise ValueError('wrong input shape')
    s=checkpoint['state_dict'];kind=checkpoint['kind']
    if kind in ('recurrent','shuffled'):
        h=[0.0]*HIDDEN if hidden is None else hidden
        x=affine(s['cell.weight_ih'],s['cell.bias_ih'],vector)
        hh=affine(s['cell.weight_hh'],s['cell.bias_hh'],h)
        output=[]
        for i in range(HIDDEN):
            r=sigmoid(x[i]+hh[i]);z=sigmoid(x[i+HIDDEN]+hh[i+HIDDEN])
            n=math.tanh(x[i+2*HIDDEN]+r*hh[i+2*HIDDEN])
            output.append((1-z)*n+z*h[i])
        return affine(s['head.weight'],s['head.bias'],output),output
    if kind=='feedforward':
        z=[math.tanh(x) for x in affine(s['fc1.weight'],s['fc1.bias'],vector)]
        return affine(s['head.weight'],s['head.bias'],z),None
    raise ValueError('unsupported policy')


def parameter_count(kind):
    if kind in ('recurrent','shuffled'):
        return 3*HIDDEN*(FEATURES+HIDDEN)+6*HIDDEN+HIDDEN*ACTIONS+ACTIONS
    if kind=='feedforward':return FEATURES*FF_HIDDEN+FF_HIDDEN+ACTIONS*FF_HIDDEN+ACTIONS
    raise ValueError('unsupported kind')


def model_macs(kind,steps=1):
    if kind in ('recurrent','shuffled'):
        return steps*(3*HIDDEN*(FEATURES+HIDDEN)+HIDDEN*ACTIONS)
    if kind=='feedforward':return steps*(FEATURES*FF_HIDDEN+FF_HIDDEN*ACTIONS)
    raise ValueError('unsupported kind')
