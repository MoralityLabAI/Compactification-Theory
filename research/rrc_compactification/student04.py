"""RRC-04 portable raw-coefficient program proposers.

All inputs are fixed scalar functions of f_0..f_64. No Euler inverse, program
labels, future data, exact synthesizer outputs, or source family ID reaches
neural inference. Frozen PyTorch weights are replayed with Python stdlib.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
try:
    from .program04 import Program, PREFIX, MAX_PERIOD, simple_features
except ImportError:
    from program04 import Program, PREFIX, MAX_PERIOD, simple_features

RNN_H=24
FF_H=21
INPUT_RNN=3
INPUT_FF=130
OUTPUT=27
RECUR_PARAMS=3*RNN_H*(INPUT_RNN+RNN_H+2)+RNN_H*(RNN_H+1)+OUTPUT*(RNN_H+1)
FF_PARAMS=FF_H*(INPUT_FF+1)+OUTPUT*(FF_H+1)


def dense(matrix, bias, x):
    return [sum(v*w for v,w in zip(row,x)) + b for row,b in zip(matrix,bias)]


def logistic(v):
    if v<-40:return 0.0
    if v>40:return 1.0
    return 1.0/(1.0+math.exp(-v))


def check_matrix(value,rows,cols):
    return isinstance(value,list) and len(value)==rows and all(
        isinstance(row,list) and len(row)==cols and all(isinstance(v,(int,float)) and math.isfinite(v) for v in row)
        for row in value)


def check_bias(value,rows):
    return isinstance(value,list) and len(value)==rows and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value)


class ProgramStudent:
    def __init__(self,checkpoint:dict):
        if checkpoint.get('schema')!='rrc04.student.v1':
            raise ValueError('unsupported student schema')
        self.variant=checkpoint.get('variant')
        if self.variant not in ('recurrent','feedforward','shuffled_recurrent'):
            raise ValueError('unexpected student kind')
        self.w=checkpoint['weights']
        if self.variant!='feedforward':
            shapes={
                'gru_ih_w':(3*RNN_H,INPUT_RNN),
                'gru_hh_w':(3*RNN_H,RNN_H),
                'refine_w':(RNN_H,RNN_H),
                'head_w':(OUTPUT,RNN_H),
            }
            biases={'gru_ih_b':3*RNN_H,'gru_hh_b':3*RNN_H,
                    'refine_b':RNN_H,'head_b':OUTPUT}
            expected=RECUR_PARAMS
        else:
            shapes={'encoder_w':(FF_H,INPUT_FF),'head_w':(OUTPUT,FF_H)}
            biases={'encoder_b':FF_H,'head_b':OUTPUT}
            expected=FF_PARAMS
        if set(self.w) != set(shapes)|set(biases):
            raise ValueError('unexpected checkpoint tensor names')
        if any(not check_matrix(self.w[k],*dims) for k,dims in shapes.items()):
            raise ValueError('invalid weight matrix')
        if any(not check_bias(self.w[k],count) for k,count in biases.items()):
            raise ValueError('invalid bias')
        if checkpoint.get('parameters')!=expected:
            raise ValueError('parameter count mismatch')
        self.checkpoint=checkpoint

    @classmethod
    def load(cls,path:Path|str)->'ProgramStudent':
        return cls(json.loads(Path(path).read_text()))

    @property
    def parameters(self):
        return self.checkpoint['parameters']

    def encode(self,prefix):
        features=simple_features(prefix)
        if self.variant=='feedforward':
            vector=[v for row in features for v in row[:2]]
            return [math.tanh(v) for v in dense(self.w['encoder_w'],self.w['encoder_b'],vector)]
        h=[0.0]*RNN_H
        ihw,ihb=self.w['gru_ih_w'],self.w['gru_ih_b']
        hhw,hhb=self.w['gru_hh_w'],self.w['gru_hh_b']
        for x in features:
            i=dense(ihw,ihb,x)
            z=dense(hhw,hhb,h)
            reset=[logistic(i[k]+z[k]) for k in range(RNN_H)]
            update=[logistic(i[RNN_H+k]+z[RNN_H+k]) for k in range(RNN_H)]
            proposal=[math.tanh(i[2*RNN_H+k]+reset[k]*z[2*RNN_H+k])
                      for k in range(RNN_H)]
            h=[(1-u)*p+u*previous for u,p,previous in zip(update,proposal,h)]
        return h

    def from_embedding(self,embedding:list[float],depth:int=2)->list[float]:
        if depth not in (0,1,2,4,8): raise ValueError('invalid recurrent depth')
        h=embedding.copy()
        if self.variant!='feedforward':
            for _ in range(depth):
                z=dense(self.w['refine_w'],self.w['refine_b'],h)
                h=[hi+0.20*math.tanh(v) for hi,v in zip(h,z)]
        return dense(self.w['head_w'],self.w['head_b'],h)

    def logits(self,prefix,depth:int=2)->list[float]:
        return self.from_embedding(self.encode(prefix),depth)

    @staticmethod
    def decode(output):
        if len(output)!=OUTPUT:raise ValueError('expected 27 logits')
        cap=max(range(3),key=lambda k:output[k])
        period=1+max(range(MAX_PERIOD),key=lambda k:output[3+k])
        mask=tuple(r for r in range(period) if output[15+r]>=0)
        return Program(cap,period,mask)

    def predict(self,prefix,depth:int=2)->Program:
        return self.decode(self.logits(prefix,depth))

    def estimated_macs(self,depth:int=2)->int:
        if self.variant=='feedforward':
            return FF_H*INPUT_FF + OUTPUT*FF_H
        return (PREFIX+1)*3*RNN_H*(INPUT_RNN+RNN_H) + depth*RNN_H*RNN_H + OUTPUT*RNN_H
