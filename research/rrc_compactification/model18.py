"""RRC-18 portable GRU and parameter-matched feedforward class scorers."""
from __future__ import annotations
import math
try:
    from . import learn18 as l
except ImportError:
    import learn18 as l

HIDDEN=32
FF_HIDDEN=51
V=64


def parameter_count(kind):
    if kind in ('recurrent','shuffled'):
        return 3*HIDDEN*(9+HIDDEN)+6*HIDDEN+(HIDDEN+1)*V
    if kind=='feedforward':
        return 56*FF_HIDDEN+FF_HIDDEN+(FF_HIDDEN+1)*V
    raise ValueError(kind)


def linear(weight,bias,x):
    return [v+sum(a*b for a,b in zip(row,x)) for row,v in zip(weight,bias)]


def sigmoid(x):
    if x>=0:
        a=math.exp(-x);return 1/(1+a)
    a=math.exp(x);return a/(1+a)


def logits(case,checkpoint):
    kind=checkpoint['kind'];state=checkpoint['state_dict']
    if kind=='feedforward':
        h=[math.tanh(z) for z in linear(state['fc1.weight'],state['fc1.bias'],l.flat(case))]
        return linear(state['head.weight'],state['head.bias'],h)
    if kind not in ('recurrent','shuffled'):raise ValueError('unknown checkpoint kind')
    h=[0.]*HIDDEN
    for x in l.sequence(case):
        a=linear(state['rnn.weight_ih_l0'],state['rnn.bias_ih_l0'],x)
        b=linear(state['rnn.weight_hh_l0'],state['rnn.bias_hh_l0'],h)
        h=[(1-sigmoid(a[i+HIDDEN]+b[i+HIDDEN]))*
                math.tanh(a[i+2*HIDDEN]+sigmoid(a[i]+b[i])*b[i+2*HIDDEN])+
                sigmoid(a[i+HIDDEN]+b[i+HIDDEN])*h[i]
            for i in range(HIDDEN)]
    return linear(state['head.weight'],state['head.bias'],h)
