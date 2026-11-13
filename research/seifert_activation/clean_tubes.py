"""SCM-08: unconditional clean-state enclosures from a certified input tube.

For RMP's exact-real pre-norm decoder, a causal attention head output is
always a convex combination of bounded value vectors. Its entire output row
therefore has a source-level norm bound without evaluating softmax. Exact
GELU also satisfies |GELU(z)|<=|z|. Together they give a finite clean
pre-LayerNorm and middle-state *tube*, anchored to an exact input center.

Mathematical status: certified for the pinned exact-real graph, given a
correct input enclosure and a frozen dyadic frequency table. The PyTorch
floating implementation is outside this proof contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
from itertools import combinations

import torch
from torch import nn

from exact_input import InputEnclosure, encode_anchors
from rmp_lipschitz import (_head_matrices,block_bound,layernorm_bounds,
                           rational,spectral_upper,sqrt_upper,sqrt_lower)
from local_jacobian import visit_token_bounds


@dataclass(frozen=True)
class CleanTubes:
    pre_centers: tuple[torch.Tensor,...]
    mid_centers: tuple[torch.Tensor,...]
    pre_center_errors: tuple[tuple[Fraction,...],...]
    mid_center_errors: tuple[tuple[Fraction,...],...]
    post_radii: tuple[tuple[Fraction,...],...]
    post_coefficients: tuple[Fraction,...]
    attention_bounds: tuple[Fraction,...]
    mlp_bounds: tuple[Fraction,...]
    input_scope: str
    frequency_sha256: str


def _block_constants(block: nn.Module, sequence: int) -> tuple[Fraction,Fraction,Fraction,Fraction]:
    """Return alpha,beta, attention-row bound, MLP-row bound."""
    if sequence<1:
        raise ValueError('empty sequence')
    block_bound(block,sequence=sequence)  # strict architectural validation
    if any(layer.bias is not None for layer in (block.mlp[0],block.mlp[2])):
        raise ValueError('biased MLP is outside pinned RMP source contract')
    _,M1=layernorm_bounds(block.norm1)
    _,M2=layernorm_bounds(block.norm2)
    heads,output=_head_matrices(block)
    values=[spectral_upper(v) for _,_,v in heads]
    A=spectral_upper(output)*sqrt_upper(sum((v*v for v in values),Fraction(0)))*M1
    H=spectral_upper(block.mlp[2].weight)*spectral_upper(block.mlp[0].weight)*M2
    alpha,beta=rational(float(block.alpha)),rational(float(block.beta))
    return alpha,beta,A,H


def _export_scaled(base: Sequence[Sequence[Fraction]], coeff: Fraction,
                   radii: Sequence[Fraction]):
    """Float64 center export includes exact rational-to-binary rounding error."""
    rows=[tuple(coeff*x for x in row) for row in base]
    zero=[[Fraction(0)]*len(row) for row in rows]
    centers,rounding=encode_anchors(rows,zero)
    # sqrt(sum (center_i-anchor_i)^2) <= r; adding the two radii is safe.
    errors=tuple(r+e for r,e in zip(radii,rounding))
    return centers,errors


def clean_state_tubes(model: nn.Module, source: InputEnclosure, *,
                      fixed_input_reinjection: bool = False) -> CleanTubes:
    """Prove pre/mid/after-visit tubes using input_anchor and weights alone.

    RMP's AnalysisModel config controls fixed_input_reinjection. The embedding
    appears again after every visit in FF-U when this is True. The error of
    the initial embedding tube must be added at each reinjection.
    """
    if model.training:
        raise ValueError('eval model required')
    if not hasattr(model,'blocks') or not hasattr(model,'module_indices'):
        raise ValueError('expected RMP recurrent decoder')
    if source.scope!='ideal_real_with_exact_dyadic_frequency_table':
        raise ValueError('unrecognized certified input contract')
    raw=source.exact_centers
    T=len(raw)
    if T<1 or source.center.shape!=(T,len(raw[0])) or len(source.radii)!=T:
        raise ValueError('inconsistent input tube')
    if any(radius<0 for radius in source.radii):
        raise ValueError('negative initial uncertainty')
    if any(len(row)!=model.hidden_size for row in raw):
        raise ValueError('hidden dimension mismatch')
    if len(model.module_indices)<1:
        raise ValueError('empty recurrent schedule')
    # Every anchor is a scalar multiple of the *exact* input anchor. This is
    # valid for any alpha, including RMP's alpha=1, and for fixed reinjection.
    coefficient=Fraction(1)
    radius=list(source.radii)
    pre=[]; mid=[]; pre_err=[]; mid_err=[]; posts=[]; coefficients=[]; attention=[]; mlp=[]
    cache={}
    for idx in model.module_indices:
        idx=int(idx)
        if idx<0 or idx>=len(model.blocks):
            raise ValueError('invalid module reference')
        if idx not in cache:
            cache[idx]=_block_constants(model.blocks[idx],T)
        a,b,A,H=cache[idx]
        pre_center,pre_error=_export_scaled(raw,coefficient,radius)
        pre.append(pre_center);pre_err.append(pre_error)
        mid_r=[abs(a)*r+abs(b)*A for r in radius]
        mid_center,mid_error=_export_scaled(raw,a*coefficient,mid_r)
        mid.append(mid_center);mid_err.append(mid_error)
        # B(X)=alpha^2 X+alpha*beta*Attn(LN1(X))
        #       +beta*MLP(LN2(alpha X+beta Attn)).
        new_radius=[a*a*r+abs(a*b)*A+abs(b)*H for r in radius]
        coefficient=a*a*coefficient
        if fixed_input_reinjection:
            coefficient+=1
            new_radius=[r+e for r,e in zip(new_radius,source.radii)]
        radius=new_radius
        posts.append(tuple(radius));coefficients.append(coefficient);attention.append(A);mlp.append(H)
    return CleanTubes(tuple(pre),tuple(mid),tuple(pre_err),tuple(mid_err),
                      tuple(posts),tuple(coefficients),tuple(attention),tuple(mlp),
                      source.scope,source.frequency_sha256)


def verified_local_pair_bounds(model:nn.Module, source:InputEnclosure, *,
                               chart:torch.Tensor,axes:torch.Tensor,
                               amplitude,observed_tokens:Sequence[int],
                               injected_token:int,inject_after_visit:int=0,
                               fixed_input_reinjection:bool=False,
                               clean:CleanTubes|None=None) -> dict:
    """Use independently bounded clean centers in SCM-07's local Jacobian.

    Outputs are conditional only on the *stated exact-real source contract*.
    This implementation does not certify the physical float forward pass or
    that the deployed positional frequency table matches the pinned table.
    """
    if clean is None:
        clean=clean_state_tubes(model,source,
                fixed_input_reinjection=fixed_input_reinjection)
    result=visit_token_bounds(model,sequence=len(source.radii),chart=chart,
                axes=axes,amplitude=amplitude,observed_tokens=observed_tokens,
                injected_token=injected_token,injected_after_visit=inject_after_visit,
                pre_centers=clean.pre_centers,mid_centers=clean.mid_centers,
                pre_center_errors=clean.pre_center_errors,
                mid_center_errors=clean.mid_center_errors)
    result['schema']='scm08_derived_ideal_clean_enclosures_v1'
    result['bound_scope']='derived_from_input_tube_and_stored_weights_exact_real'
    result['center_enclosures_verified_by_this_program']=True
    result['input_scope']=clean.input_scope
    result['positional_frequency_table_sha256']=clean.frequency_sha256
    result['floating_execution_certified']=False
    result['deployed_frequency_table_verified']=False
    result['source_global_fallback_used']=any(
        max(map(Fraction,diag.get('token_radii',['0'])))>Fraction(10)**10
        for diag in result['per_visit_tokens'])
    result['clean_enclosures']=[{
        'visit':k+1,
        'maximum_pre_error':str(max(clean.pre_center_errors[k])),
        'maximum_mid_error':str(max(clean.mid_center_errors[k])),
        'attention_row_upper':str(clean.attention_bounds[k]),
        'mlp_row_upper':str(clean.mlp_bounds[k]),
        'maximum_post_tube_radius':str(max(clean.post_radii[k])),
    } for k in range(len(clean.post_radii))]
    return result


def certified_injection_amplitude(source:InputEnclosure, tubes:CleanTubes, *,
                                  injected_token:int,after_visit:int,
                                  radius_fraction) -> Fraction:
    """Upper bound the actual source-real clean token RMS without float captures.

    RMP measures the injected direction radius from the clean state after the
    chosen visit. A certified RMS upper must cover that state, rather than
    treating the rounded RMS in the captured JSON as exact.
    """
    if not (0<=after_visit<len(tubes.post_radii) and 0<=injected_token<len(source.radii)):
        raise ValueError('invalid injected token or visit')
    scale=rational(radius_fraction)
    if scale<0:
        raise ValueError('negative relative radius')
    row=source.exact_centers[injected_token]
    coeff=tubes.post_coefficients[after_visit]
    norm=sqrt_upper(sum(((coeff*value)**2 for value in row),Fraction(0)),digits=16)
    error=tubes.post_radii[after_visit][injected_token]
    return scale*(norm+error)*sqrt_upper(Fraction(1,len(row)),digits=16)


def certified_zero_disk(source:InputEnclosure,tubes:CleanTubes,chart:torch.Tensor,
                        pair_bounds:dict,observed_tokens:Sequence[int], *,
                        margin='0') -> list[dict]:
    """Certified ideal null-braid test using exact anchor geometry alone.

    No float model evaluation is used. A true clean pair is enclosed about
    a scalar multiple of the exact input anchor, with per-token error radii.
    If lower pair clearance exceeds the perturbation-disk bound for every
    pair, the disk avoids projected collisions and its braid is null.
    """
    if chart.ndim!=2 or chart.shape[1]!=2:
        raise ValueError('expected two-dimensional readout chart')
    if len(observed_tokens)<2 or len(set(observed_tokens))!=len(observed_tokens):
        raise ValueError('distinct observed tokens required')
    if any(i<0 or i>=len(source.radii) for i in observed_tokens):
        raise ValueError('observed token outside input')
    if len(pair_bounds['per_visit_pair_bounds'])!=len(tubes.post_coefficients):
        raise ValueError('visit count mismatch')
    chart_values=[[rational(float(value)) for value in row] for row in chart.detach().cpu().tolist()]
    if len(chart_values)!=len(source.exact_centers[0]):
        raise ValueError('chart and hidden dimension mismatch')
    margin=rational(margin)
    if margin<0:
        raise ValueError('negative clearance margin')
    chart_norm=spectral_upper(chart)
    reports=[]
    for v,coeff in enumerate(tubes.post_coefficients):
        pairs={}; all_pass=True
        for i,j in combinations(range(len(observed_tokens)),2):
            ii,jj=observed_tokens[i],observed_tokens[j]
            difference=[coeff*(a-b) for a,b in zip(source.exact_centers[ii],source.exact_centers[jj])]
            projected=[sum((difference[k]*chart_values[k][axis] for k in range(len(difference))),Fraction(0))
                       for axis in range(2)]
            anchor_dist=sqrt_lower(sum((p*p for p in projected),Fraction(0)),digits=16)
            lower=max(Fraction(0),anchor_dist-chart_norm*(tubes.post_radii[v][ii]+tubes.post_radii[v][jj]))
            dynamic=Fraction(pair_bounds['per_visit_pair_bounds'][v][f'{i},{j}'])
            valid=lower>dynamic+margin
            all_pass &= valid
            pairs[f'{i},{j}']={'clean_pair_lower':str(lower),'perturbation_pair_upper':str(dynamic),
                              'proven_null_pair':bool(valid)}
        reports.append({'visit':v+1,'null_braid_for_ideal_real_model':bool(all_pass),
                        'pairs':pairs,'floating_execution_certified':False})
    return reports
