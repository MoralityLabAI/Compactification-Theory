"""RRC-09 portable, PyTorch-free inference from frozen CPU checkpoints."""
from __future__ import annotations

import math
from typing import Sequence

try:
    from .synth09 import PAD,OPS,feature_pairs,decode_tokens
except ImportError:
    from synth09 import PAD,OPS,feature_pairs,decode_tokens


def _dense(x:Sequence[float],W:Sequence[Sequence[float]],b:Sequence[float]):
    return [bias+sum(w*v for w,v in zip(row,x)) for row,bias in zip(W,b)]


def _sigmoid(z:float):
    if z>=0:
        x=math.exp(-z)
        return 1/(1+x)
    x=math.exp(z)
    return x/(1+x)


def _argmax(xs:Sequence[float]):
    return max(range(len(xs)),key=lambda i:xs[i])


def direct_logits(case,checkpoint,*,sweeps:int=1)->list[float]:
    """No candidate interpreter, program labels or Euler oracle are accessible."""
    if sweeps not in (1,2,4):raise ValueError('registered sweeps are 1,2,4')
    features=feature_pairs(case)
    kind=checkpoint['kind'];weights=checkpoint['state_dict']
    if kind=='feedforward':
        if sweeps!=1:raise ValueError('feedforward has one inference pass')
        flat=[v for pair in features for v in pair]
        hidden=[math.tanh(x) for x in _dense(flat,weights['fc1.weight'],weights['fc1.bias'])]
        return _dense(hidden,weights['head.weight'],weights['head.bias'])
    if kind not in ('recurrent','shuffled_recurrent'):
        raise ValueError('unsupported checkpoint')
    ih=weights['rnn.weight_ih_l0'];hh=weights['rnn.weight_hh_l0']
    bi=weights['rnn.bias_ih_l0'];bh=weights['rnn.bias_hh_l0']
    hidden_size=len(hh[0]);hidden=[0.0]*hidden_size
    for _ in range(sweeps):
        for x in features:
            wi=_dense(x,ih,bi)
            wh=_dense(hidden,hh,bh)
            r=[_sigmoid(wi[k]+wh[k]) for k in range(hidden_size)]
            z=[_sigmoid(wi[hidden_size+k]+wh[hidden_size+k]) for k in range(hidden_size)]
            n=[math.tanh(wi[2*hidden_size+k]+r[k]*wh[2*hidden_size+k]) for k in range(hidden_size)]
            hidden=[(1-z[k])*n[k]+z[k]*hidden[k] for k in range(hidden_size)]
    return _dense(hidden,weights['head.weight'],weights['head.bias'])


def direct_program(case,checkpoint,*,sweeps:int=1):
    logits=direct_logits(case,checkpoint,sweeps=sweeps)
    return decode_tokens((_argmax(logits[:len(OPS)+1]),
                          _argmax(logits[len(OPS)+1:])))
