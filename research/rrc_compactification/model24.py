"""Portable no-interpreter model for RRC-24 ordering of four base operators."""
from __future__ import annotations
import math
try:
    from . import delay24 as d
except ImportError:
    import delay24 as d

HIDDEN=11
FF_HIDDEN=8


def params(kind):
    if kind in ('recurrent','shuffled'):
        return 3*HIDDEN*(4+HIDDEN)+6*HIDDEN+(HIDDEN+1)*4
    if kind=='feedforward':return (17*4+1)*FF_HIDDEN+(FF_HIDDEN+1)*4
    raise ValueError(kind)


def sigmoid(v):
    if v>=0:
        e=math.exp(-v);return 1/(1+e)
    e=math.exp(v);return e/(1+e)


def affine(w,b,x):return [s+sum(a*c for a,c in zip(row,x)) for row,s in zip(w,b)]


def logits(case,checkpoint):
    s=checkpoint['state_dict'];kind=checkpoint['kind'];seq=d.learn_features(case)
    if kind=='feedforward':
        vec=[z for row in seq for z in row]
        hidden=[math.tanh(x) for x in affine(s['fc.weight'],s['fc.bias'],vec)]
        return affine(s['head.weight'],s['head.bias'],hidden)
    if kind not in ('recurrent','shuffled'):raise ValueError(kind)
    h=[0.]*HIDDEN
    for row in seq:
        x=affine(s['rnn.weight_ih_l0'],s['rnn.bias_ih_l0'],row)
        v=affine(s['rnn.weight_hh_l0'],s['rnn.bias_hh_l0'],h)
        out=[]
        for i in range(HIDDEN):
            reset=sigmoid(x[i]+v[i]);z=sigmoid(x[i+HIDDEN]+v[i+HIDDEN])
            u=math.tanh(x[i+2*HIDDEN]+reset*v[i+2*HIDDEN])
            out.append((1-z)*u+z*h[i])
        h=out
    return affine(s['head.weight'],s['head.bias'],h)


def order(case,checkpoint):
    z=logits(case,checkpoint)
    return tuple(d.OPS[j] for j in sorted(range(4),key=lambda j:(-z[j],j)))


def macs(kind):
    if kind in ('recurrent','shuffled'):return 17*3*HIDDEN*(4+HIDDEN)+4*HIDDEN
    if kind=='feedforward':return 17*4*FF_HIDDEN+4*FF_HIDDEN
    raise ValueError(kind)
