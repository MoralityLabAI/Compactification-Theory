"""Portable raw-only RRC-07 refinement network, independent of exact Euler oracles.

The initial logits come from frozen RRC-06. Each extra pass sees the same raw
features, its own preceding class probabilities and recurrent hidden state.
Neither evaluator residuals nor future coefficients enter this forward pass.
"""
from __future__ import annotations

import math
from typing import Sequence

try:
    from .raw_student06 import _linear, _sigmoid
    from .rrc06 import PREFIX
except ImportError:
    from raw_student06 import _linear, _sigmoid
    from rrc06 import PREFIX

HIDDEN=32
FF_HIDDEN=7
INPUT=9
BASE_INPUT=6
CLASSES=3
DEFAULT_LOOPS=2


def parameter_count(kind:str)->int:
    if kind in ('recurrent','shuffled_recurrent'):
        return 3*HIDDEN*(INPUT+HIDDEN)+6*HIDDEN+(CLASSES*HIDDEN+CLASSES)
    if kind=='feedforward':
        return (PREFIX+1)*BASE_INPUT*FF_HIDDEN+FF_HIDDEN+FF_HIDDEN*(PREFIX+1)*CLASSES+(PREFIX+1)*CLASSES
    raise ValueError('invalid network kind')


def softmax(row:Sequence[float])->list[float]:
    shift=max(row)
    exp=[math.exp(float(x)-shift) for x in row]
    s=sum(exp)
    return [v/s for v in exp]


def portable_forward(features:Sequence[Sequence[float]],
                     base_logits:Sequence[Sequence[float]],ckpt:dict,
                     *,passes:int|None=None)->list[list[float]]:
    if len(features)!=PREFIX+1 or len(base_logits)!=PREFIX+1 or any(len(x)!=3 for x in features) or any(len(x)!=4 for x in base_logits):
        raise ValueError('expected raw features (65,3), base RRC-06 logits (65,4)')
    kind=ckpt['kind']
    if ckpt['parameter_count']!=parameter_count(kind):raise ValueError('parameter count mismatch')
    weight=ckpt['weights'];base=[list(z[:3]) for z in base_logits]
    if any(not math.isfinite(v) for row in base for v in row):raise ValueError('nonfinite base')
    base_p=[softmax(v) for v in base]
    if kind=='feedforward':
        if passes is not None:raise ValueError('feedforward has no pass count')
        flat=[z for row,probs in zip(features,base_p) for z in list(row)+probs]
        hidden=[math.tanh(x) for x in _linear(weight['l1.weight'],weight['l1.bias'],flat)]
        z=_linear(weight['l2.weight'],weight['l2.bias'],hidden)
        return [[base[i][j]+z[3*i+j] for j in range(3)] for i in range(PREFIX+1)]
    passes=ckpt['training_passes'] if passes is None else passes
    if not isinstance(passes,int) or passes not in (0,1,2,4,8):
        raise ValueError('unsupported depth')
    if passes==0:return base
    n=HIDDEN
    w_ih=weight['gru.weight_ih_l0'];w_hh=weight['gru.weight_hh_l0']
    b_ih=weight['gru.bias_ih_l0'];b_hh=weight['gru.bias_hh_l0']
    h=[0.]*n
    current=base
    for _ in range(passes):
        next_logits=[]
        for i in range(PREFIX+1):
            sample=list(features[i])+base_p[i]+softmax(current[i])
            inp=_linear(w_ih,b_ih,sample)
            prev=_linear(w_hh,b_hh,h)
            r=[_sigmoid(inp[j]+prev[j]) for j in range(n)]
            z=[_sigmoid(inp[n+j]+prev[n+j]) for j in range(n)]
            h_t=[math.tanh(inp[2*n+j]+r[j]*prev[2*n+j]) for j in range(n)]
            h=[(1-z[j])*h_t[j]+z[j]*h[j] for j in range(n)]
            corr=_linear(weight['head.weight'],weight['head.bias'],h)
            next_logits.append([base[i][j]+corr[j] for j in range(3)])
        current=next_logits
    return current


def propose_int(logits:Sequence[Sequence[float]])->list[int]:
    if len(logits)!=PREFIX+1 or any(len(row)!=3 for row in logits):
        raise ValueError('expected per-coordinate ternary logits')
    return [0]+[max(range(3),key=lambda i:logits[n][i])-1
                 for n in range(1,PREFIX+1)]
