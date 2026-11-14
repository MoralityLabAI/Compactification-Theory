"""SCM-11: source-certified local causal attention response bounds.

Builds on SCM-10's CPU IEEE-binary64 interval reference. Treats stored model
weights and centers as exact dyadic values and uses outward interval error
bounds for point Q/K/V and softmax probabilities. The ideal-real theorem is
conditional on the pinned CPU/BLAS verifier arithmetic contract. It does not
certify CUDA execution or an observed link.

The main sharpening is to exploit the zero-sum probability change and retain
the reference attention mass on tokens with nonzero value variation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ball_centered import (_block_norms, _certified_qkv_reference,
                           _local_ln_lipschitz, BlockNorms)
from fast_centered import (ArrayI, _block, _extract, _norm_point, _weights,
                           _validate_model, dot_fixed, dot_pair, exp_interval,
                           _down, _up, _from_fraction)
from rmp_lipschitz import (rational, sqrt_upper, ceiling_decimal, spectral_upper, block_bound)
from validated_forward import centered_null_disk


@dataclass(frozen=True)
class HeadReference:
    qnorm: tuple[Fraction, ...]
    knorm: tuple[Fraction, ...]
    vnorm: tuple[Fraction, ...]
    anchor_radii: tuple[Fraction, ...]
    # upper bound for the clean reference attention weights at each query row
    probability_upper: tuple[tuple[Fraction, ...], ...]


def _norm_upper_from_endpoints(lo: np.ndarray, hi: np.ndarray) -> Fraction:
    """Exact dyadic bound for Euclidean norm of every enclosed vector."""
    sq = Fraction(0)
    for a, b in zip(lo, hi):
        x = Fraction.from_float(float(a))
        y = Fraction.from_float(float(b))
        sq += max(abs(x), abs(y)) ** 2
    return sqrt_upper(sq, digits=20)


def _point_probabilities(scores: ArrayI) -> tuple[tuple[Fraction,...],...]:
    """Upper-bound the source-reference row probabilities, all via intervals."""
    n = scores.lo.shape[0]
    result=[]
    for i in range(n):
        upper_shift = float(np.max(scores.hi[i, :i+1]))
        # Subtract a fixed EXACT binary64 shift. All ideal scores are <= shift.
        lower=_down(scores.lo[i,:i+1] - upper_shift)
        upper=np.minimum(0.,_up(scores.hi[i,:i+1] - upper_shift))
        expv=exp_interval(ArrayI(lower,upper))
        if np.any(expv.lo <= 0.):
            raise ValueError('reference softmax denominator enclosure nonpositive')
        prob=expv/expv.sum(axis=-1)
        result.append(tuple(Fraction.from_float(float(min(1.,max(0.,x))))
                            for x in prob.hi))
    return tuple(result)


def reference_attention(block:nn.Module, center:torch.Tensor,
                        weights:dict|None=None) -> tuple[HeadReference,...]:
    """Certify Q/K/V norms, prefix value diameters and attention masses."""
    if center.ndim!=2 or not center.is_floating_point() or not torch.isfinite(center).all():
        raise ValueError('finite [tokens,hidden] source center required')
    w=_weights(block) if weights is None else weights
    x=center.detach().cpu().to(torch.float64).numpy()
    normed=_norm_point(ArrayI.point(x),w['g1'],w['b1'],w['eps1'])
    qkv=dot_fixed(normed,w['wqkv'])
    n,d=x.shape
    heads=[]
    old=_certified_qkv_reference(block,center,w)
    inv=ArrayI.point(1.)/ArrayI.point(w['d']).sqrt()
    for h in range(w['h']):
        start=h*w['d'];end=start+w['d']
        Q=ArrayI(qkv.lo[:,start:end],qkv.hi[:,start:end])
        K=ArrayI(qkv.lo[:,d+start:d+end],qkv.hi[:,d+start:d+end])
        V=ArrayI(qkv.lo[:,2*d+start:2*d+end],qkv.hi[:,2*d+start:2*d+end])
        score=dot_pair(Q,K,scale=inv)
        qnorm,knorm,vnorm=old[h]
        # The probability delta has coefficient max_j ||v0_j-a||.
        # Anchors a=0 and a=v0_0 are compared. The latter is represented
        # with its certified interval; its j=0 deviation is EXACTLY zero.
        anchored=[]
        maxnorm=Fraction(0); maxdist=Fraction(0)
        for i in range(n):
            maxnorm=max(maxnorm,vnorm[i])
            if i>0:
                lo=_down(V.lo[i]-V.hi[0])
                hi=_up(V.hi[i]-V.lo[0])
                maxdist=max(maxdist,_norm_upper_from_endpoints(lo,hi))
            anchored.append(min(maxnorm,maxdist))
        heads.append(HeadReference(tuple(qnorm),tuple(knorm),tuple(vnorm),
                                   tuple(anchored),_point_probabilities(score)))
    return tuple(heads)


def attention_response_bound(ref:HeadReference, *, prefix:int,
                             delta_query:Fraction,
                             delta_keys:Sequence[Fraction],
                             delta_values:Sequence[Fraction],
                             query_norm:Fraction|None=None,
                             key_norms:Sequence[Fraction]|None=None,
                             baseline_query_delta:Fraction=Fraction(0),
                             baseline_key_delta:Sequence[Fraction]|None=None,
                             baseline_value_delta:Sequence[Fraction]|None=None,
                             head_dim:int) -> dict:
    """Valid ball around a causal head output under a finite perturbation.

    Let s0,v0 be the certified SOURCE reference and sb,vb the unknown clean
    baseline. Score variation sb->s is bounded by DS; s0->sb by DB. Then

      ||head(s,v)-head(sb,vb)|| <=
          min(max(dv),sum(p0_upper*dv)+(DB+DS)*range(dv)/2)
          + (R0+max(dbv))*DS.

    This follows from the L_inf->L_1 softmax Jacobian bound and the
    zero-sum change in probabilities. No assumption of baseline equality
    with the source-reference center is needed.
    """
    i=prefix
    if i<0 or i>=len(ref.qnorm) or len(delta_keys)<i+1 or len(delta_values)<i+1:
        raise ValueError('bad causal prefix')
    if head_dim<=0:
        raise ValueError('invalid head width')
    if baseline_key_delta is None:
        baseline_key_delta=[Fraction(0)]*(i+1)
    if baseline_value_delta is None:
        baseline_value_delta=[Fraction(0)]*(i+1)
    if min([delta_query,baseline_query_delta,*delta_keys[:i+1],
            *delta_values[:i+1],*baseline_key_delta[:i+1],
            *baseline_value_delta[:i+1]])<0:
        raise ValueError('negative uncertainty')
    if query_norm is None:
        query_norm=ref.qnorm[i]
    if key_norms is None:
        key_norms=ref.knorm
    scale=sqrt_upper(Fraction(1,head_dim),digits=20)
    ds=Fraction(0);db=Fraction(0)
    for j in range(i+1):
        dbq=baseline_query_delta; dbk=baseline_key_delta[j]
        # Source-reference Q,K plus (unknown) clean baseline perturbation.
        db=max(db,scale*(dbq*ref.knorm[j]+ref.qnorm[i]*dbk+dbq*dbk))
        dq=delta_query;dk=delta_keys[j]
        ds=max(ds,scale*(dq*(ref.knorm[j]+dbk)+
                         (ref.qnorm[i]+dbq)*dk+dq*dk))
    dv=[rational(x) for x in delta_values[:i+1]]
    max_dv=max(dv);min_dv=min(dv)
    p_up=ref.probability_upper[i]
    if len(p_up)!=i+1:
        raise ValueError('reference attention prefix mismatch')
    weighted=sum((p_up[j]*dv[j] for j in range(i+1)),Fraction(0))
    value_term=min(max_dv, weighted+(db+ds)*(max_dv-min_dv)/2)
    radius=ref.anchor_radii[i]+max(baseline_value_delta[:i+1])
    return {'bound':value_term+radius*ds,
            'score_response':ds,'score_center_error':db,
            'value_response':value_term,'value_anchor_radius':radius,
            'max_value_change':max_dv}


def _change_bound_refined(block:nn.Module,norms:BlockNorms,
                          pre:torch.Tensor, clean_errors:Sequence[Fraction],
                          response:Sequence[Fraction],mid:torch.Tensor,
                          mid_clean_errors:Sequence[Fraction],
                          references:Sequence[HeadReference]) -> tuple[Fraction,...]:
    """Uniform response ball for a full intervention disk at one block."""
    n=len(response)
    if (pre.shape!=mid.shape or pre.shape[0]!=n or len(clean_errors)!=n or
         len(mid_clean_errors)!=n or len(references)!=len(norms.qkv_heads)):
        raise ValueError('response reference shape mismatch')
    first_base=[_local_ln_lipschitz(pre[j],rational(clean_errors[j]),block.norm1)
                *rational(clean_errors[j]) for j in range(n)]
    first_response=[_local_ln_lipschitz(pre[j],rational(clean_errors[j])+rational(response[j]),block.norm1)
                    *rational(response[j]) for j in range(n)]
    mid_errors=[];post_errors=[]
    for i in range(n):
        heads=[]
        for h,(nq,nk,nv) in enumerate(norms.qkv_heads):
            v=attention_response_bound(references[h],prefix=i,
                delta_query=nq*first_response[i],
                delta_keys=[nk*x for x in first_response[:i+1]],
                delta_values=[nv*x for x in first_response[:i+1]],
                baseline_query_delta=nq*first_base[i],
                baseline_key_delta=[nk*x for x in first_base[:i+1]],
                baseline_value_delta=[nv*x for x in first_base[:i+1]],
                head_dim=norms.head_dim)
            heads.append(v['bound'])
        datt=norms.output*sqrt_upper(sum((v*v for v in heads),Fraction(0)),digits=20)
        dm=abs(norms.alpha)*rational(response[i])+abs(norms.beta)*datt
        mid_errors.append(ceiling_decimal(dm,digits=20))
        l2=_local_ln_lipschitz(mid[i],rational(mid_clean_errors[i])+dm,block.norm2)
        dp=(abs(norms.alpha)+abs(norms.beta)*norms.down*
            Fraction(5,4)*norms.up*l2)*dm
        post_errors.append(ceiling_decimal(dp,digits=20))
    return tuple(post_errors)


def refined_disk_bounds(model:nn.Module,tubes,*,chart:torch.Tensor,axes:torch.Tensor,
                        amplitude,observed_tokens:Sequence[int],
                        injected_token:int,inject_after_visit:int=0,
                        margin='1/10000000')->dict:
    """Carry bounded *response* throughout the registered intervention disk.

    The clean centers and center errors must have been obtained from the
    source-locked SCM-10 verifier, not from neural activation captures.
    The resulting null-braid passes remain conditional on CPU IEEE/BLAS
    arithmetic and exact-real model semantics.
    """
    from fast_centered import ieee_preflight
    if model.training:
        raise ValueError('eval model required')
    n,hidden=tubes.pre_centers[0].shape
    if tubes.input_scope != 'ideal_real_with_exact_dyadic_frequency_table':
        raise ValueError('source-locked exact-real input enclosure required')
    if (n>tubes.max_sequence or hidden>tubes.max_hidden or
        len(model.module_indices)>8 or model.hidden_size!=hidden):
        raise ValueError('source-locked shape mismatch')
    if hidden & (hidden-1):
        raise ValueError('source-locked fast backend requires power-of-two hidden dimension')
    if getattr(tubes,'method',None)!='scm10_ieee_cpu_roundoff_plus_causal_L2_balls_exact_real':
        raise ValueError('SCM-10 source-certified BallTubes required')
    for idx in model.module_indices:
        block=model.blocks[idx]
        block_bound(block,sequence=n)
        if block.mlp[0].bias is not None or block.mlp[2].bias is not None:
            raise ValueError('biased MLP outside RMP contract')
    if chart.shape!=axes.shape or tuple(chart.shape)!=(hidden,2):
        raise ValueError('expected same-shape [hidden,2] chart and axes')
    if not (torch.isfinite(chart).all() and torch.isfinite(axes).all()):
        raise ValueError('nonfinite chart/axes')
    obs=tuple(int(x) for x in observed_tokens)
    if len(obs)<2 or len(set(obs))!=len(obs) or any(i<0 or i>=n for i in obs):
        raise ValueError('bad observed token indices')
    if not 0<=injected_token<n or not 0<=inject_after_visit<len(model.module_indices):
        raise ValueError('bad injection index')
    if any(not x.is_floating_point() for x in tubes.pre_centers):
        raise ValueError('invalid centers')
    ieee_preflight()
    base=rational(amplitude)
    if base<0:
        raise ValueError('negative patch amplitude')
    ax_norm=spectral_upper(axes)
    chart_norm=spectral_upper(chart)
    delta=[Fraction(0) for _ in range(n)]
    history=[];pair_bounds=[]
    caches={}
    for visit,idx in enumerate(model.module_indices):
        if visit>0 and visit>inject_after_visit:
            if idx not in caches:
                block=model.blocks[idx]
                w=_weights(block)
                caches[idx]=(_block_norms(block),w)
            norms,w=caches[idx]
            references=reference_attention(model.blocks[idx],tubes.pre_centers[visit],w)
            delta=list(_change_bound_refined(model.blocks[idx],norms,
                  tubes.pre_centers[visit],tubes.pre_center_errors[visit],delta,
                  tubes.mid_centers[visit],tubes.mid_center_errors[visit],references))
            if max(delta)>10**12:
                raise ValueError('unresolved: certified response exceeds registered useful-radius budget')
        if visit==inject_after_visit:
            delta[injected_token]+=base*ax_norm
        pairs={}
        for a in range(len(obs)):
            for b in range(a+1,len(obs)):
                pairs[f'{a},{b}']=chart_norm*(delta[obs[a]]+delta[obs[b]])
        pair_bounds.append(pairs)
        history.append(tuple(delta))
    certificates=centered_null_disk(tubes,chart,obs,pair_bounds,margin=margin)
    return {'schema':'scm11_attention_diameter_source_disk_v1',
      'proof_scope':'ideal_real_frozen_RMP_preLN_assumed_IEEE_binary64_cpu_BLAS',
      'bound_method':'certified_reference_value_diameter_and_softmax_mass',
      'visit_pair_bounds':tuple({key:str(val) for key,val in p.items()} for p in pair_bounds),
      'visit_response_radii':tuple(tuple(str(v) for v in row) for row in history),
      'null_braid_ideal_real':certificates,
      'floating_execution_certified':False,'observed_link_certified':False}
