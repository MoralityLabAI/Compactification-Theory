"""RRC-25: portable 1000-parameter GRU and 995-parameter MLP template ranks."""
from __future__ import annotations
import math
try:
    from . import compound25 as b
except ImportError:
    import compound25 as b

HIDDEN = 12
FF_HIDDEN = 11
VOCAB=16


def parameters(kind):
    if kind in ('recurrent','shuffled'):
        return 3*HIDDEN*(8+HIDDEN)+6*HIDDEN+(HIDDEN+1)*VOCAB
    if kind=='feedforward':return (9*8+1)*FF_HIDDEN+(FF_HIDDEN+1)*VOCAB
    raise ValueError(kind)


def affine(w,bias,x):return [v+sum(a*z for a,z in zip(row,x)) for row,v in zip(w,bias)]


def sigmoid(x):
    if x>=0:
        z=math.exp(-x);return 1/(1+z)
    z=math.exp(x);return z/(1+z)


def logits(case,header,checkpoint):
    kind=checkpoint['kind'];s=checkpoint['state_dict'];x=b.features(case,header)
    if kind=='feedforward':
        h=[math.tanh(v) for v in affine(s['fc.weight'],s['fc.bias'],[v for row in x for v in row])]
        return affine(s['head.weight'],s['head.bias'],h)
    if kind not in ('recurrent','shuffled'):raise ValueError('bad model kind')
    h=[0.]*HIDDEN
    for row in x:
        a=affine(s['rnn.weight_ih_l0'],s['rnn.bias_ih_l0'],row)
        v=affine(s['rnn.weight_hh_l0'],s['rnn.bias_hh_l0'],h)
        out=[]
        for i in range(HIDDEN):
            reset=sigmoid(a[i]+v[i]);update=sigmoid(a[i+HIDDEN]+v[i+HIDDEN])
            new=math.tanh(a[i+2*HIDDEN]+reset*v[i+2*HIDDEN])
            out.append((1-update)*new+update*h[i])
        h=out
    return affine(s['head.weight'],s['head.bias'],h)


def macs(kind):
    if kind in ('recurrent','shuffled'):return 9*3*HIDDEN*(8+HIDDEN)+VOCAB*HIDDEN
    if kind=='feedforward':return 9*8*FF_HIDDEN+VOCAB*FF_HIDDEN
    raise ValueError(kind)
