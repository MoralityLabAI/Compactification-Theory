"""Portable bounded recurrent graph constructor, no candidate-catalog search.

The decoder commits left, right, then post operations. In active mode it gets
one public partial-program execution after each of the first two choices.
"""
from __future__ import annotations
import math
try:
    from . import graph13 as b
except ImportError:
    import graph13 as b

HIDDEN=24
INPUT=4+b.FEEDBACK_SIZE+3


def sigmoid(x):
    if x>=0:
        e=math.exp(-x);return 1.0/(1.0+e)
    e=math.exp(x);return e/(1.0+e)


def affine(matrix,bias,vec):
    return [sum(row[i]*vec[i] for i in range(len(vec)))+t for row,t in zip(matrix,bias)]


def gru_step(x,h,state):
    ih=affine(state['cell.weight_ih'],state['cell.bias_ih'],x)
    hh=affine(state['cell.weight_hh'],state['cell.bias_hh'],h)
    size=len(h)
    return [(1.0-sigmoid(ih[k+size]+hh[k+size])) *
            math.tanh(ih[k+2*size]+sigmoid(ih[k]+hh[k])*hh[k+2*size])+
            sigmoid(ih[k+size]+hh[k+size])*h[k] for k in range(size)]


def infer(case,checkpoint,*,feedback_enabled=None,forced_prefix=()):
    """One graph is proposed without exhaustive candidate execution.

    forced_prefix is used only for unit-test parity; normal inference takes
    model argmax decisions. It cannot read future, hidden graph or graph labels.
    """
    if checkpoint['kind'] not in ('active','passive'):raise ValueError('wrong checkpoint kind')
    if not 0<=len(forced_prefix)<=3 or any(x not in b.OPS for x in forced_prefix):
        raise ValueError('invalid forced prefix')
    if feedback_enabled is None:feedback_enabled=(checkpoint['kind']=='active')
    s=checkpoint['state_dict'];hidden=checkpoint['hidden_size']
    h=[math.tanh(v) for v in affine(s['encoder.weight'],s['encoder.bias'],b.raw_features(case))]
    previous=[0.]*4
    feedback=[0.]*b.FEEDBACK_SIZE
    chosen=[];logits=[];queries=0
    for stage in range(3):
        step=[float(int(j==stage)) for j in range(3)]
        h=gru_step(previous+feedback+step,h,s)
        z=affine(s['head.weight'],s['head.bias'],h)
        logits.append(z)
        op=forced_prefix[stage] if stage<len(forced_prefix) else b.OPS[max(range(4),key=lambda j:z[j])]
        chosen.append(op)
        previous=[float(int(v==op)) for v in b.OPS]
        if stage<2:
            if feedback_enabled:
                feedback=list(b.residual_feedback(case,chosen[0],chosen[1] if stage==1 else None))
                queries+=1
            else:feedback=[0.]*b.FEEDBACK_SIZE
    return {'graph':tuple(chosen),'logits':logits,'partial_queries':queries,
            'model_mac_proxy':inference_macs(checkpoint)}


def inference_macs(checkpoint):
    h=checkpoint['hidden_size']
    return 104*h+3*(3*h*(INPUT+h)+4*h)
