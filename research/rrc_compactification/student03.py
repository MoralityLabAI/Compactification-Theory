"""RRC-03 portable 353-parameter policies: sequential RNN and full-history MLP.

Both consume the same public (query index, binary response) history and known
hypothesis count. No true hypothesis, Euler exponents, exact interval or future
coefficients are passed into either policy. Standard-library inference.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HIDDEN = 16
INPUT = 4
PACKED = 20
MAX_STEPS = 6
PARAMETERS = 353


def rnn_inputs(history: tuple[tuple[int, int], ...], size: int) -> tuple[tuple[float, ...], ...]:
    if not 2 <= size <= 65 or len(history) > MAX_STEPS:
        raise ValueError('invalid size/history')
    x = []
    for step,(threshold, response) in enumerate(history):
        if not 0 <= threshold < size-1 or response not in (0, 1):
            raise ValueError('invalid query/response')
        x.append((threshold/(size-2) if size>2 else 0.0,
                  float(2*response-1), (step+1)/MAX_STEPS, (size-1)/64))
    return tuple(x)


def packed_inputs(history: tuple[tuple[int, int], ...], size: int) -> tuple[float, ...]:
    feedback = rnn_inputs(history,size)
    rows = []
    for i in range(MAX_STEPS):
        if i < len(feedback):
            rows += [feedback[i][0],feedback[i][1],1.0]
        else:
            rows += [0.0,0.0,0.0]
    rows += [(size-1)/64, len(feedback)/MAX_STEPS]
    assert len(rows)==PACKED
    return tuple(rows)


def dense(w: list[list[float]], b: list[float], x) -> list[float]:
    return [sum(y*z for y,z in zip(row,x))+bias for row,bias in zip(w,b)]


def sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-max(-40.0,min(40.0,x))))


class QueryStudent:
    def __init__(self, checkpoint: dict):
        if checkpoint.get('schema') != 'rrc03.student.v1':
            raise ValueError('unsupported checkpoint schema')
        variant = checkpoint.get('variant')
        if variant not in ('recurrent','feedforward'):
            raise ValueError('invalid model variant')
        self.variant = variant
        self.checkpoint = checkpoint
        spec = ([('in',HIDDEN,INPUT),('state',HIDDEN,HIDDEN),('out',1,HIDDEN)]
                if variant=='recurrent' else [('fc',HIDDEN,PACKED),('out',1,HIDDEN)])
        w=checkpoint.get('weights',{})
        self.w,self.b={},{}
        count=0
        for name,rows,cols in spec:
            a,b=w.get(name+'_w'),w.get(name+'_b')
            if name=='state' and variant=='recurrent':
                if not isinstance(a,list) or len(a)!=rows or any(len(row)!=cols for row in a) or b not in (None,[]):
                    raise ValueError('invalid recurrent state weight shape')
                self.w[name],self.b[name]=a,[0.0]*rows
                count+=rows*cols
            else:
                if not isinstance(a,list) or len(a)!=rows or any(len(row)!=cols for row in a) or not isinstance(b,list) or len(b)!=rows:
                    raise ValueError('invalid weight tensor shape: '+name)
                self.w[name],self.b[name]=a,b
                count+=rows*cols+rows
        if count!=PARAMETERS or count!=checkpoint.get('parameter_count'):
            raise ValueError('parameter budget mismatch')

    @classmethod
    def load(cls,path: str|Path):
        return cls(json.loads(Path(path).read_text()))

    def predict_fraction(self, history: tuple[tuple[int,int],...], size: int) -> float:
        if self.variant=='recurrent':
            h=[0.0]*HIDDEN
            for x in rnn_inputs(history,size):
                inputs=dense(self.w['in'],self.b['in'],x)
                memory=dense(self.w['state'],self.b['state'],h)
                h=[math.tanh(i+j) for i,j in zip(inputs,memory)]
        else:
            h=[math.tanh(v) for v in dense(self.w['fc'],self.b['fc'],packed_inputs(history,size))]
        return sigmoid(dense(self.w['out'],self.b['out'],h)[0])

    def action(self,history:tuple[tuple[int,int],...],size:int,budget:int) -> int:
        return max(0,min(size-2,int(self.predict_fraction(history,size)*(size-2)+.5)))

    def macs_for_history(self, history_length: int) -> int:
        if self.variant=='recurrent':
            return history_length*HIDDEN*(INPUT+HIDDEN)+HIDDEN
        return HIDDEN*PACKED+HIDDEN
