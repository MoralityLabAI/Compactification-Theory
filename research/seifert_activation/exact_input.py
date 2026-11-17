"""SCM-08: rational enclosures for the idealized RMP embedding input.

Stored embedding/LayerNorm weights and a stored float32 sinusoidal frequency
TABLE are interpreted as exact real constants. LayerNorm and trigonometric
functions are evaluated as real functions with rational outward bounds.
This contract does NOT certify PyTorch floating point execution, nor that
CPU/GPU generated positional frequency tables agree bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from math import factorial
from functools import lru_cache
from typing import Sequence

import torch
from torch import nn

from rmp_lipschitz import rational, sqrt_lower, sqrt_upper, ceiling_decimal


@dataclass(frozen=True)
class InputEnclosure:
    center: torch.Tensor
    radii: tuple[Fraction, ...]
    exact_centers: tuple[tuple[Fraction, ...], ...]
    frequency_sha256: str
    scope: str = 'ideal_real_with_exact_dyadic_frequency_table'


def atan_interval(reciprocal: int, terms: int = 32) -> tuple[Fraction, Fraction]:
    """Alternating arctangent series, with its next-term remainder."""
    if reciprocal < 1 or not (1 <= terms <= 100):
        raise ValueError('invalid arctangent parameters')
    x = Fraction(1, reciprocal)
    partial = sum(((-1)**k * x**(2*k+1) / (2*k+1) for k in range(terms)), Fraction(0))
    tail = x**(2*terms+1) / (2*terms+1)
    return ((partial, partial+tail) if terms % 2 == 0
            else (partial-tail, partial))


@lru_cache(maxsize=16)
def pi_interval(terms: int = 32) -> tuple[Fraction, Fraction]:
    """Machin identity pi/4=4 atan(1/5)-atan(1/239), rigorous rational."""
    a,b = atan_interval(5, terms)
    c,d = atan_interval(239, terms)
    return 16*a-4*d, 16*b-4*c


def _taylor_sincos(x: Fraction, *, terms: int = 24) -> tuple[tuple[Fraction,Fraction],tuple[Fraction,Fraction]]:
    """Sin, cos on [-pi,pi] by polynomials and absolute Taylor remainders."""
    x=rational(x)
    if abs(x) > Fraction(22,7):
        raise ValueError('reduce angle before invoking the Taylor checker')
    sine = sum(((-1)**k*x**(2*k+1)/factorial(2*k+1) for k in range(terms)),Fraction(0))
    cosine = sum(((-1)**k*x**(2*k)/factorial(2*k) for k in range(terms)),Fraction(0))
    # Lagrange remainder; every derivative of sin/cos has absolute value <=1.
    es=abs(x)**(2*terms)/factorial(2*terms)
    ec=abs(x)**(2*terms-1)/factorial(2*terms-1)
    return (sine-es,sine+es),(cosine-ec,cosine+ec)


@lru_cache(maxsize=100000)
def trig_rational(argument: Fraction, kind: str, *, terms: int = 24) -> tuple[Fraction,Fraction]:
    """Exact-rational center and error bounding sin/cos of rational input."""
    if kind not in ('sin','cos'):
        raise ValueError('only sin/cos supported')
    x=rational(argument)
    pi_lo,pi_hi=pi_interval(32)
    pi_exact_mid=(pi_lo+pi_hi)/2
    grid=10**20
    pi_mid=Fraction((pi_exact_mid*grid+Fraction(1,2)).numerator//
                    (pi_exact_mid*grid+Fraction(1,2)).denominator,grid)
    pi_error=abs(pi_exact_mid-pi_mid)+(pi_hi-pi_lo)/2
    ratio=x/(2*pi_mid)
    k=(ratio+Fraction(1,2)).numerator//(ratio+Fraction(1,2)).denominator
    midpoint=x-2*k*pi_mid
    uncertainty=2*abs(k)*pi_error
    if abs(midpoint)>Fraction(22,7):
        raise ValueError('period reduction failed')
    sine,cosine=_taylor_sincos(midpoint,terms=terms)
    lo,hi = sine if kind=='sin' else cosine
    # Global 1-Lipschitz sin and cos justify the interval shift.
    center=(lo+hi)/2
    rounded=Fraction((center*10**18+Fraction(1,2)).numerator//
                   (center*10**18+Fraction(1,2)).denominator,10**18)
    error=ceiling_decimal((hi-lo)/2+uncertainty+abs(center-rounded),digits=18)
    return rounded,error


def exact_float_vector(values: torch.Tensor) -> tuple[Fraction,...]:
    if values.ndim!=1 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError('require finite 1D floating point tensor')
    return tuple(rational(float(value)) for value in values.detach().cpu().tolist())


def exact_layernorm_point(norm: nn.LayerNorm, raw: Sequence[Fraction]) -> tuple[tuple[Fraction,...],tuple[Fraction,...]]:
    """Independent exact-real LayerNorm enclosure using rational sqrt intervals."""
    if not isinstance(norm,nn.LayerNorm) or len(norm.normalized_shape)!=1 or not norm.elementwise_affine:
        raise ValueError('require 1D affine LayerNorm')
    d=len(raw)
    if norm.normalized_shape[0]!=d or norm.weight is None or norm.bias is None:
        raise ValueError('invalid affine LayerNorm dimensions')
    x=tuple(rational(value) for value in raw)
    mean=sum(x,Fraction(0))/d
    variance=sum(((value-mean)**2 for value in x),Fraction(0))/d
    s=variance+rational(float(norm.eps))
    if s<=0:
        raise ValueError('LayerNorm epsilon invalid')
    lo=sqrt_lower(s,digits=20)
    hi=sqrt_upper(s,digits=20)
    if lo<=0:
        raise ValueError('LayerNorm interval resolution too coarse')
    invlo,invhi=1/hi,1/lo
    inv=(invlo+invhi)/2
    inv_error=(invhi-invlo)/2
    gamma=exact_float_vector(norm.weight)
    bias=exact_float_vector(norm.bias)
    raw_centers=tuple(g*(value-mean)*inv+b for g,value,b in zip(gamma,x,bias))
    raw_errors=tuple(abs(g*(value-mean))*inv_error for g,value in zip(gamma,x))
    rounded=tuple(Fraction((v*10**18+Fraction(1,2)).numerator//
                   (v*10**18+Fraction(1,2)).denominator,10**18) for v in raw_centers)
    errors=tuple(ceiling_decimal(e+abs(c-r),digits=18)
                 for e,c,r in zip(raw_errors,raw_centers,rounded))
    return rounded,errors


def encode_anchors(exact_centers: Sequence[Sequence[Fraction]],
                   coordinate_errors: Sequence[Sequence[Fraction]]) -> tuple[torch.Tensor,tuple[Fraction,...]]:
    """Export float64 center with an *exact bound* on its rounding error."""
    if len(exact_centers)!=len(coordinate_errors) or not exact_centers:
        raise ValueError('inconsistent center rows')
    out=[]; radii=[]; width=len(exact_centers[0])
    for center,error in zip(exact_centers,coordinate_errors):
        if len(center)!=width or len(error)!=width:
            raise ValueError('nonrectangular center or error')
        vec=[]; sq=Fraction(0)
        for x,e in zip(center,error):
            xx=rational(x);ee=rational(e)
            if ee<0:
                raise ValueError('negative claimed error')
            ff=float(xx)
            if not math.isfinite(ff):
                raise ValueError('center conversion overflow')
            vec.append(ff)
            distance=abs(rational(ff)-xx)+ee
            sq+=distance*distance
        out.append(vec)
        radii.append(sqrt_upper(sq,digits=18))
    return torch.tensor(out,dtype=torch.float64),tuple(radii)


def rmp_input_enclosure(model: nn.Module, tokens: torch.Tensor, *,
                         positional_frequency_table: torch.Tensor | None = None) -> InputEnclosure:
    """Anchor ideal LN(embedding)+sin/cos with rational, source-bound error.

    RMP uses float32 frequencies produced from an exponential table. The
    supplied/reconstructed table is an explicit certificate assumption and is
    hashed; real-valued sin/cos are certified for these exact stored values.
    """
    if getattr(model,'carry_policy','reset')!='reset':
        raise ValueError('RMP input proof requires reset carry policy')
    if tokens.ndim != 1 or tokens.dtype not in (torch.int32,torch.int64,torch.int16):
        raise ValueError('one integer token row required')
    if not all(hasattr(model,k) for k in ('embedding','input_norm','use_sinusoidal_positions','hidden_size')):
        raise ValueError('not an RMP-style embedder')
    if not isinstance(model.embedding,nn.Embedding):
        raise ValueError('expected embedding module')
    d=model.hidden_size
    if d!=model.embedding.embedding_dim:
        raise ValueError('dimension mismatch')
    if any(int(t)<0 or int(t)>=model.embedding.num_embeddings for t in tokens.tolist()):
        raise ValueError('token outside embedding')
    if model.use_sinusoidal_positions:
        if positional_frequency_table is None:
            even=torch.arange(0,d,2,dtype=torch.float32)
            positional_frequency_table=torch.exp(-math.log(10_000.0)*even/d)
        frequency=exact_float_vector(positional_frequency_table)
        if len(frequency)!=(d+1)//2:
            raise ValueError('incorrect frequency table length')
    else:
        if positional_frequency_table is not None:
            raise ValueError('position table supplied for model without position encoding')
        frequency=()
    sha=hashlib.sha256(b''.join(str(x).encode('ascii')+b';' for x in frequency)).hexdigest()
    centers=[]; errs=[]
    for pos,index in enumerate(tokens.tolist()):
        raw=exact_float_vector(model.embedding.weight[int(index)])
        cn,en=exact_layernorm_point(model.input_norm,raw)
        if frequency:
            trig=[trig_rational(pos*frequency[j//2], 'sin' if j%2==0 else 'cos') for j in range(d)]
            cn=tuple(v+c for v,(c,e) in zip(cn,trig))
            en=tuple(v+e for v,(c,e) in zip(en,trig))
        centers.append(cn)
        errs.append(en)
    tensor,radii=encode_anchors(centers,errs)
    return InputEnclosure(tensor,radii,tuple(centers),sha)
