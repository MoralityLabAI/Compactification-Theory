"""RRC-06 portable CPU inference: no teacher, divisor graph or exact solver.

Saved PyTorch GRU/MLP weights are evaluated using Python standard-library
operations. All inputs are log1p-scaled *raw* partition coefficients.
"""
from __future__ import annotations

import math
from typing import Sequence

try:
    from .rrc06 import PREFIX,raw_features
except ImportError:
    from rrc06 import PREFIX,raw_features

INPUT=3
RECURRENT_HIDDEN=23
FEEDFORWARD_HIDDEN=4
DEFAULT_LOOPS=2


def parameter_count(kind:str)->int:
    if kind in ('recurrent','shuffled_recurrent'):
        h=RECURRENT_HIDDEN
        return 3*h*(INPUT+h)+6*h+(4*h+4)
    if kind=='feedforward':
        m=FEEDFORWARD_HIDDEN
        return (PREFIX+1)*INPUT*m+m+m*(PREFIX+1)*4+(PREFIX+1)*4
    raise ValueError('unknown kind')


def _sigmoid(z:float)->float:
    if z<0:
        q=math.exp(z)
        return q/(1+q)
    return 1/(1+math.exp(-z))


def _linear(w:Sequence[Sequence[float]], b:Sequence[float],x:Sequence[float])->list[float]:
    return [v+sum(wij*xj for wij,xj in zip(wi,x)) for wi,v in zip(w,b)]


def portable_forward(features:Sequence[Sequence[float]],checkpoint:dict,
                     *,loops:int|None=None)->list[list[float]]:
    """Only locally computed features and saved parameters influence output."""
    if len(features)!=PREFIX+1 or any(len(x)!=INPUT for x in features):
        raise ValueError('expected full raw coefficient feature sequence')
    kind=checkpoint['kind']
    if checkpoint['parameter_count']!=parameter_count(kind):
        raise ValueError('checkpoint parameter count mismatch')
    state=checkpoint['weights']
    if kind=='feedforward':
        if loops is not None:raise ValueError('feedforward has no loop budget')
        flat=[v for sample in features for v in sample]
        hidden=[math.tanh(v) for v in _linear(state['l1.weight'],state['l1.bias'],flat)]
        z=_linear(state['l2.weight'],state['l2.bias'],hidden)
        return [z[i*4:i*4+4] for i in range(PREFIX+1)]
    depth=checkpoint['training_depth'] if loops is None else loops
    if not isinstance(depth,int) or not 0<=depth<=8:raise ValueError('invalid loop depth')
    if depth==0:return [[0.,1.,0.,0.] for _ in range(PREFIX+1)]
    h=[0.]*RECURRENT_HIDDEN
    w_ih=state['gru.weight_ih_l0'];w_hh=state['gru.weight_hh_l0']
    b_ih=state['gru.bias_ih_l0'];b_hh=state['gru.bias_hh_l0']
    n=RECURRENT_HIDDEN
    logits=[]
    for _ in range(depth):
        logits=[]
        for sample in features:
            inp=_linear(w_ih,b_ih,sample)
            prev=_linear(w_hh,b_hh,h)
            r=[_sigmoid(inp[i]+prev[i]) for i in range(n)]
            z=[_sigmoid(inp[n+i]+prev[n+i]) for i in range(n)]
            new=[math.tanh(inp[2*n+i]+r[i]*prev[2*n+i]) for i in range(n)]
            h=[(1-z[i])*new[i]+z[i]*h[i] for i in range(n)]
            logits.append(_linear(state['head.weight'],state['head.bias'],h))
    return logits


def infer_raw(prefix:Sequence[int],checkpoint:dict,*,loops:int|None=None)->list[list[float]]:
    return portable_forward(raw_features(prefix),checkpoint,loops=loops)
