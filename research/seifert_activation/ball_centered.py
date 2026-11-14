"""SCM-10: causal, tokenwise validated centers for the RMP ideal-real graph.

Dense point evaluation uses SCM-10's IEEE/BLAS conservative interval backend.
Variation from the point center is bounded in Euclidean norm, using exact
source-weight spectral upper bounds, the local LayerNorm derivative, and the
L_infinity->L_1 softmax Jacobian inequality. This avoids exponentially wide
coordinatewise interval boxes at 64-token, 128-hidden recurrence scale.

Result scope is conditional on the stated binary64/BLAS arithmetic model;
physical PyTorch/CUDA traces and pretrained model data are not certified.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
import torch
from torch import nn

from exact_input import InputEnclosure
from fast_centered import ArrayI, _block, _extract, _from_fraction, _validate_model, _weights, _norm_point, dot_fixed
from rmp_lipschitz import (spectral_upper,vector_upper,rational,sqrt_lower,
                           sqrt_upper,ceiling_decimal)


def _local_ln_lipschitz(center:torch.Tensor, radius:Fraction, ln:nn.LayerNorm)->Fraction:
    """Exact-real L-Lipschitz bound on closed Euclidean ball around center."""
    x=[rational(float(y)) for y in center.tolist()]
    d=len(x)
    mu=sum(x,Fraction(0))/d
    projected=sum(((v-mu)**2 for v in x),Fraction(0))
    E=rational(radius)
    if E<0:
        raise ValueError('invalid negative error bound')
    lower=max(Fraction(0),sqrt_lower(projected,digits=20)-E)
    gamma=max(abs(rational(float(v))) for v in ln.weight.detach().cpu().tolist())
    eps=rational(float(ln.eps))
    den=sqrt_lower(eps+lower*lower/d,digits=20)
    if den<=0:
        raise ValueError('LayerNorm denominator lower bound missing')
    return ceiling_decimal(gamma/den,digits=20)


def _ln_image_upper(ln:nn.LayerNorm)->Fraction:
    """Per-token absolute bound on LN image for every real input."""
    gamma=max(abs(rational(float(v))) for v in ln.weight.detach().cpu().tolist())
    beta=vector_upper(ln.bias)
    return sqrt_upper(Fraction(len(ln.weight)))*gamma+beta


@dataclass(frozen=True)
class BlockNorms:
    qkv_heads:tuple[tuple[Fraction,Fraction,Fraction],...]
    output:Fraction
    up:Fraction
    down:Fraction
    max_normalized:Fraction
    head_dim:int
    alpha:Fraction
    beta:Fraction


def _block_norms(block:nn.Module)->BlockNorms:
    from rmp_lipschitz import _head_matrices
    heads,projection=_head_matrices(block)
    bounds=tuple(tuple(min(spectral_upper(w),exact_gram_spectral_upper(w)) for w in group) for group in heads)
    return BlockNorms(bounds,min(spectral_upper(projection),exact_gram_spectral_upper(projection)),
         min(spectral_upper(block.mlp[0].weight),exact_gram_spectral_upper(block.mlp[0].weight)),
         min(spectral_upper(block.mlp[2].weight),exact_gram_spectral_upper(block.mlp[2].weight)),
         _ln_image_upper(block.norm1),block.head_size,
         rational(float(block.alpha)),rational(float(block.beta)))


def _certified_qkv_reference(block:nn.Module,center:torch.Tensor,weights:dict):
    """Source-validated Q/K/V point norms, retaining causal token identity.

    Each number is an upper bound on the true real Euclidean head vector at
    the supplied exact binary64 center. Uses the same proved point interval
    arithmetic as the SCM-10 matrix audit.
    """
    reference=_norm_point(ArrayI.point(center.numpy()),weights['g1'],weights['b1'],weights['eps1'])
    qkv=dot_fixed(reference,weights['wqkv'])
    n,d=center.shape; heads=weights['h'];hd=weights['d']
    result=[]
    for h in range(heads):
        row=[]
        for shift in (0,d,2*d):
            lo=qkv.lo[:,shift+h*hd:shift+(h+1)*hd]
            hi=qkv.hi[:,shift+h*hd:shift+(h+1)*hd]
            norms=[]
            for a,b in zip(lo,hi):
                sq=sum((max(abs(Fraction.from_float(float(x))),abs(Fraction.from_float(float(y))))**2
                         for x,y in zip(a,b)),Fraction(0))
                norms.append(sqrt_upper(sq,digits=20))
            row.append(tuple(norms))
        result.append(tuple(row))
    return tuple(result)


def _change_bound(block:nn.Module, weight:BlockNorms,
                  pre:torch.Tensor,errors:Sequence[Fraction],
                  mid:torch.Tensor,mid_round:Sequence[Fraction], qkv_reference):
    """Bound B(X)-B(C) and M(X)-M(C) for causal RMP, tokenwise.

    Pure source bound; `mid` is a verified point enclosure of M(C).
    """
    n=len(errors)
    if pre.shape!=mid.shape or pre.shape[0]!=n:
        raise ValueError('source center dimensions')
    errs=[rational(e) for e in errors]
    first=[_local_ln_lipschitz(pre[i],errs[i],block.norm1)*errs[i] for i in range(n)]
    maxprior=Fraction(0)
    mid_errors=[];post_errors=[]
    root_head_inv=sqrt_upper(Fraction(1,weight.head_dim),digits=20)
    for i in range(n):
        maxprior=max(maxprior,first[i])
        perhead=[]
        for h,(nq,nk,nv) in enumerate(weight.qkv_heads):
            dq=nq*first[i]
            dk=nk*maxprior
            # q0,k0,v0 norms bounded on all real token states.
            q0=qkv_reference[h][0][i]
            k0=max(qkv_reference[h][1][:i+1])
            v0=max(qkv_reference[h][2][:i+1])
            ds=(dq*k0 + q0*dk + dq*dk)*root_head_inv
            dv=nv*maxprior
            dy=dv+v0*ds
            perhead.append(dy)
        datt=weight.output*sqrt_upper(sum((y*y for y in perhead),Fraction(0)),digits=20)
        dm=abs(weight.alpha)*errs[i]+abs(weight.beta)*datt
        mid_errors.append(ceiling_decimal(dm,digits=20))
        layer2radius=dm+rational(mid_round[i])
        l2=_local_ln_lipschitz(mid[i],layer2radius,block.norm2)
        g=Fraction(5,4)
        post_delta=(abs(weight.alpha)+abs(weight.beta)*weight.down*g*weight.up*l2)*dm
        post_errors.append(ceiling_decimal(post_delta,digits=20))
    return tuple(mid_errors),tuple(post_errors)


@dataclass(frozen=True)
class BallTubes:
    pre_centers:tuple[torch.Tensor,...]
    mid_centers:tuple[torch.Tensor,...]
    post_centers:tuple[torch.Tensor,...]
    pre_center_errors:tuple[tuple[Fraction,...],...]
    mid_center_errors:tuple[tuple[Fraction,...],...]
    post_center_errors:tuple[tuple[Fraction,...],...]
    input_scope:str
    frequency_sha256:str
    max_sequence:int
    max_hidden:int
    method:str='scm10_ieee_cpu_roundoff_plus_causal_L2_balls_exact_real'
    floating_execution_certified:bool=False


def verified_ball_tubes(model:nn.Module,source:InputEnclosure,* ,
                        fixed_input_reinjection:bool=False,
                        max_seq:int=64,max_dim:int=128,max_visits:int=8)->BallTubes:
    _validate_model(model,source,max_seq=max_seq,max_dim=max_dim,max_visits=max_visits)
    # Initial center is stored binary64 and hence an exact real point.
    center=source.center.detach().cpu().to(torch.float64)
    errors=tuple(rational(x) for x in source.radii)
    inp=center.clone();inp_errors=errors
    pre,mid,post=[],[],[]
    perr,merr,oerr=[],[],[]
    blocks={};norms={}
    for idx in model.module_indices:
        block=model.blocks[idx]
        if idx not in blocks:
            blocks[idx]=_weights(block)
            norms[idx]=_block_norms(block)
        pre.append(center);perr.append(errors)
        point=ArrayI.point(center.numpy())
        middle,out=_block(point,blocks[idx])
        cmid,eroundmid=_extract(middle)
        new,eroundpost=_extract(out)
        qref=_certified_qkv_reference(block,center,blocks[idx])
        dmid,dpost=_change_bound(block,norms[idx],center,errors,cmid,eroundmid,qref)
        mr=tuple(eroundmid[i]+dmid[i] for i in range(len(errors)))
        if fixed_input_reinjection:
            # Exact real reinjection adds the original input center and its
            # independently certified uncertainty. Include center add roundoff.
            reinjected=ArrayI.point(new.numpy())+ArrayI.point(inp.numpy())
            new2,readd=_extract(reinjected)
            nr=tuple(eroundpost[i]+dpost[i]+inp_errors[i]+readd[i]
                     for i in range(len(errors)))
            new=new2
        else:
            nr=tuple(eroundpost[i]+dpost[i] for i in range(len(errors)))
        mid.append(cmid);merr.append(mr)
        post.append(new);oerr.append(nr)
        center,errors=new,nr
    return BallTubes(tuple(pre),tuple(mid),tuple(post),
        tuple(perr),tuple(merr),tuple(oerr),source.scope,source.frequency_sha256,
        max_seq,max_dim)

# See SCM10_MATH.md: exact-dyadic Gram/Gershgorin source-norm certificate.
def exact_gram_spectral_upper(matrix:torch.Tensor)->Fraction:
    """||W||_2 <= sqrt(||W^T W||_infty), with *integer* Gram products.

    Select the smaller Gram dimension; every stored binary float is converted
    to an exact integer under a common power-of-two scaling. This calculation
    makes no floating-point linear algebra assumptions.
    """
    if matrix.ndim!=2 or matrix.numel()<1 or matrix.numel()>500_000 or not matrix.is_floating_point():
        raise ValueError('unsupported matrix for certified Gram')
    xs=matrix.detach().cpu().to(torch.float64).numpy()
    if not np.isfinite(xs).all():
        raise ValueError('nonfinite checkpoint weights')
    entries=xs.ravel().tolist()
    shifts=[];scale=0
    for v in entries:
        p,q=float(v).as_integer_ratio()
        s=q.bit_length()-1
        shifts.append((p,s));scale=max(scale,s)
    nums=[p<<(scale-s) for p,s in shifts]
    Z=np.array(nums,dtype=object).reshape(xs.shape)
    A=Z if Z.shape[0]<=Z.shape[1] else Z.T
    gram=A@A.T
    rows=max((sum((abs(v) for v in row),0) for row in gram),default=0)
    return sqrt_upper(Fraction(rows,1<<(2*scale)),digits=20)
