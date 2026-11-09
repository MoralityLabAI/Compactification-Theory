"""Optional NumPy CPU accelerator for the unchanged RRC-06 / RRC-07 GRU equations.

Raw-feature-only inference. Scalar standard-library implementations remain
canonical for parity checks. Float64 NumPy evaluation is used for bounded CI.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from .rrc07 import raw_features, PREFIX
    from .student07 import parameter_count
except ImportError:
    from rrc07 import raw_features, PREFIX
    from student07 import parameter_count


def sigmoid(a):
    # The clipped domain is well inside floating-point overflow at frozen weights.
    return np.where(a>=0,1/(1+np.exp(-np.maximum(a,0))),
                    np.exp(np.minimum(a,0))/(1+np.exp(np.minimum(a,0))))


def cell(x,h,wih,whh,bih,bhh):
    v=wih @ x+bih
    z=whh @ h+bhh
    n=len(h)
    reset=sigmoid(v[:n]+z[:n])
    keep=sigmoid(v[n:2*n]+z[n:2*n])
    candidate=np.tanh(v[2*n:]+reset*z[2*n:])
    return (1-keep)*candidate+keep*h


def fast_base(prefix, ckpt):
    if ckpt['kind']!='recurrent' or ckpt['training_depth']!=2:
        raise ValueError('frozen base identity mismatch')
    features=raw_features(prefix)
    x=np.asarray(features,dtype=np.float64)
    w={k:np.asarray(v,dtype=np.float64) for k,v in ckpt['weights'].items()}
    h=np.zeros(w['gru.weight_hh_l0'].shape[1])
    out=None
    for _ in range(ckpt['training_depth']):
        out=[]
        for i in range(PREFIX+1):
            h=cell(x[i],h,w['gru.weight_ih_l0'],w['gru.weight_hh_l0'],
                   w['gru.bias_ih_l0'],w['gru.bias_hh_l0'])
            out.append((w['head.weight'] @ h+w['head.bias']).tolist())
    return features,out


def fast_forward(features,base_logits,ckpt,passes=None):
    kind=ckpt['kind']
    if ckpt['parameter_count']!=parameter_count(kind):
        raise ValueError('bad checkpoint parameter count')
    x=np.asarray(features,dtype=np.float64)
    base=np.asarray(base_logits,dtype=np.float64)[:,:3]
    base_p=np.exp(base-base.max(axis=1,keepdims=True))
    base_p/=base_p.sum(axis=1,keepdims=True)
    w={k:np.asarray(v,dtype=np.float64) for k,v in ckpt['weights'].items()}
    if kind=='feedforward':
        if passes is not None:raise ValueError('feedforward has no scan count')
        inputs=np.concatenate((x,base_p),axis=1).ravel()
        hidden=np.tanh(w['l1.weight'] @ inputs + w['l1.bias'])
        correction=w['l2.weight'] @ hidden+w['l2.bias']
        return (base+correction.reshape(PREFIX+1,3)).tolist()
    depth=ckpt['training_passes'] if passes is None else passes
    if depth not in (0,1,2,4,8):raise ValueError('invalid depth')
    if depth==0:return base.tolist()
    h=np.zeros(w['gru.weight_hh_l0'].shape[1])
    current=base.copy()
    for _ in range(depth):
        previous=np.exp(current-current.max(axis=1,keepdims=True))
        previous/=previous.sum(axis=1,keepdims=True)
        sample=np.concatenate((x,base_p,previous),axis=1)
        out=[]
        for i in range(PREFIX+1):
            h=cell(sample[i],h,w['gru.weight_ih_l0'],w['gru.weight_hh_l0'],
                   w['gru.bias_ih_l0'],w['gru.bias_hh_l0'])
            out.append((base[i]+w['head.weight'] @ h+w['head.bias']).tolist())
        current=np.asarray(out,dtype=np.float64)
    return current.tolist()
