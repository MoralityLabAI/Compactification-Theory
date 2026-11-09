"""Portable pure-Python direct recurrent/feedforward graph classifiers for RRC-12."""
from __future__ import annotations
import math
try:
    from .graph12 import OPS,features
except ImportError:
    from graph12 import OPS,features


def sigmoid(x):
    if x>=0:
        e=math.exp(-x);return 1/(1+e)
    e=math.exp(x);return e/(1+e)


def affine(weights, bias, v):
    return [b+sum(a*x for a,x in zip(row,v)) for row,b in zip(weights,bias)]


def recurrent_logits(inputs, checkpoint, scans=1):
    if scans<1:raise ValueError('at least one complete observation scan required')
    s=checkpoint['state_dict']; hid=checkpoint['hidden_size']; h=[0.0]*hid
    ih=s['rnn.weight_ih_l0'];hh=s['rnn.weight_hh_l0'];bi=s['rnn.bias_ih_l0'];bh=s['rnn.bias_hh_l0']
    # A repeated scan never changes the number of learned weights.
    for _ in range(scans):
        for x in inputs:
            ix=affine(ih,bi,x);hx=affine(hh,bh,h)
            next_h=[]
            for j in range(hid):
                r=sigmoid(ix[j]+hx[j]);z=sigmoid(ix[j+hid]+hx[j+hid]);
                n=math.tanh(ix[j+2*hid]+r*hx[j+2*hid])
                next_h.append((1-z)*n+z*h[j])
            h=next_h
    return affine(s['head.weight'],s['head.bias'],h)


def feedforward_logits(inputs, checkpoint):
    s=checkpoint['state_dict'];flat=[v for row in inputs for v in row]
    h=[math.tanh(x) for x in affine(s['fc1.weight'],s['fc1.bias'],flat)]
    return affine(s['head.weight'],s['head.bias'],h)


def logits(case,checkpoint,*,scans=1):
    x=features(case)
    kind=checkpoint['kind']
    if kind in ('recurrent','shuffled_recurrent'):
        return recurrent_logits(x,checkpoint,scans=scans)
    if kind=='feedforward':
        if scans!=1:raise ValueError('feedforward has one pass')
        return feedforward_logits(x,checkpoint)
    raise ValueError('bad model kind')


def predict(case,checkpoint,*,scans=1):
    y=logits(case,checkpoint,scans=scans)
    tokens=tuple(OPS[max(range(4),key=lambda i:y[4*j+i])] for j in range(3))
    return tokens


def inference_mac(checkpoint,scans=1):
    if checkpoint['kind'] in ('recurrent','shuffled_recurrent'):
        h=checkpoint['hidden_size'];i=4
        return scans*26*3*h*(i+h)+12*h
    return 26*4*checkpoint['hidden_size']+12*checkpoint['hidden_size']
