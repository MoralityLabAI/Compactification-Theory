"""Tiny RRC-02 CPU checkpoint inference with no third-party dependencies.

The trainer uses PyTorch, but the exported weights can be replayed by pure
Python. The student receives six public summaries of Euler exponents for a
candidate residue class; every recurrent pass sees the *same* evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

FEATURES = 6
RECURRENT_HIDDEN = 12
FF_HIDDEN = (18, 11)


def make_features(observed: list[int]) -> tuple[float, ...]:
    if len(observed) < 2:
        raise ValueError('candidate class must have >=2 observations')
    a = [max(-3.0, min(3.0, float(v))) for v in observed]
    count = len(a)
    return (
        sum(a)/(2*count),
        sum(v*v for v in a)/(4*count),
        a[0]/2,
        a[-1]/2,
        count/32,
        sum(abs(a[i]-a[i-1]) for i in range(1,count))/(2*(count-1)),
    )


def dense(weights: list[list[float]], bias: list[float], x: list[float] | tuple[float, ...]) -> list[float]:
    return [sum(w*v for w,v in zip(row,x)) + b for row,b in zip(weights,bias)]


def checked_shape(mat: list[list[float]], rows: int, cols: int) -> bool:
    return len(mat) == rows and all(len(row) == cols for row in mat)


class ResidueStudent:
    def __init__(self, checkpoint: dict):
        self.checkpoint = checkpoint
        if checkpoint.get('schema') != 'rrc02.tiny-student.v1':
            raise ValueError('wrong checkpoint schema')
        variant = checkpoint.get('variant')
        if variant not in ('recurrent','feedforward'):
            raise ValueError('invalid variant')
        self.variant = variant
        w = checkpoint['weights']
        if variant == 'recurrent':
            specification = (
                ('enc', RECURRENT_HIDDEN, FEATURES),
                ('loop', RECURRENT_HIDDEN, RECURRENT_HIDDEN),
                ('ctx', RECURRENT_HIDDEN, FEATURES),
                ('out', 3, RECURRENT_HIDDEN),
            )
        else:
            specification = (
                ('fc1', FF_HIDDEN[0], FEATURES),
                ('fc2', FF_HIDDEN[1], FF_HIDDEN[0]),
                ('out', 3, FF_HIDDEN[1]),
            )
        self.w = {}
        self.b = {}
        for name,rows,cols in specification:
            mat,bias = w[name+'_w'],w[name+'_b']
            if not checked_shape(mat,rows,cols) or len(bias)!=rows:
                raise ValueError(f'wrong shape for {name}')
            self.w[name] = mat
            self.b[name] = bias
        self.parameter_count = sum(len(mat)*len(mat[0]) + len(self.b[name])
                                   for name,mat in self.w.items())
        if self.parameter_count != checkpoint.get('parameter_count'):
            raise ValueError('parameter count mismatch')

    @classmethod
    def load(cls, path: str | Path) -> 'ResidueStudent':
        return cls(json.loads(Path(path).read_text()))

    def logits(self, x: tuple[float, ...], depth: int) -> list[float]:
        if len(x)!=FEATURES or depth not in (0, 1, 2, 4, 8):
            raise ValueError('invalid feature vector or recurrent depth')
        if self.variant == 'feedforward':
            if depth != 0:
                raise ValueError('feedforward only supports depth=0')
            h = [math.tanh(v) for v in dense(self.w['fc1'],self.b['fc1'],x)]
            h = [math.tanh(v) for v in dense(self.w['fc2'],self.b['fc2'],h)]
            return dense(self.w['out'],self.b['out'],h)
        h = [math.tanh(v) for v in dense(self.w['enc'],self.b['enc'],x)]
        context = dense(self.w['ctx'],self.b['ctx'],x)
        for _ in range(depth):
            recurrent = dense(self.w['loop'],self.b['loop'],h)
            h = [math.tanh(a+b) for a,b in zip(recurrent,context)]
        return dense(self.w['out'],self.b['out'],h)

    def predict(self, x: tuple[float, ...], depth: int) -> int:
        v = self.logits(x,depth)
        return max(range(3), key=lambda i:v[i])
