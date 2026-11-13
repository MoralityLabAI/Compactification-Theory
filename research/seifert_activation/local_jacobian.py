"""SCM-07: causal, tokenwise Lipschitz propagation for RMP explicit decoder.

Real-arithmetic inequalities use dyadic-rational weight norm upper bounds from
SCM-06. The center-free path is a global theorem for the pinned architecture.
The optional variance-adaptive path requires *independent certified* enclosures
of the ideal clean pre/mid states. Float captures alone are only exploratory.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Sequence

import torch
from torch import nn

from rmp_lipschitz import (
    _gamma_beta, _head_matrices, block_bound, layernorm_bounds, rational,
    spectral_upper, sqrt_lower, sqrt_upper, ceiling_decimal,
)


def _nonnegative(x: Any) -> Fraction:
    value = rational(x)
    if value < 0:
        raise ValueError('Nonnegative certified radius required')
    return value


def _vector_ratios(x: torch.Tensor) -> list[Fraction]:
    if x.ndim != 1 or not x.is_floating_point():
        raise ValueError('Center must be a floating vector of shape [hidden]')
    values = x.detach().cpu().tolist()
    return [rational(float(z)) for z in values]


def centered_norm_sq(x: torch.Tensor) -> Fraction:
    """Compute ||P x||_2^2 exactly over stored binary float coordinates."""
    values = _vector_ratios(x)
    mean = sum(values, Fraction(0)) / len(values)
    return sum(((value - mean)**2 for value in values), Fraction(0))


def local_layernorm_lipschitz(norm: nn.LayerNorm, center: torch.Tensor, radius: Any,
                              *, center_error: Any = 0) -> Fraction:
    """Conditional upper bound on ||D LN(x)|| for ||x-center||<=radius+error.

    `center_error` must separately bound discrepancy between this *recorded*
    center and the ideal real-valued center when used as a certified result.
    All computations treat the recorded tensor coordinates as exact rationals.
    """
    gamma, _, d, eps = _gamma_beta(norm)
    if center.numel() != d or center.ndim != 1:
        raise ValueError('Center and LayerNorm dimension mismatch')
    total_radius = _nonnegative(radius) + _nonnegative(center_error)
    mean_free_lower = sqrt_lower(centered_norm_sq(center), digits=14)
    lower = max(Fraction(0), mean_free_lower - total_radius)
    global_upper = layernorm_bounds(norm)[0]
    if lower == 0:
        return global_upper
    energy_lower = eps + lower*lower/d
    local = ceiling_decimal(gamma * sqrt_upper(1/energy_lower, digits=14), digits=12)
    return min(local, global_upper)


def _validate_centers(centers: torch.Tensor | None,
                      errors: Sequence[Any] | None, *, T: int, d: int, label: str):
    if centers is None:
        if errors is not None:
            raise ValueError(f'{label} error bounds need centers')
        return None
    if centers.shape != (T, d) or not centers.is_floating_point() or not torch.isfinite(centers).all():
        raise ValueError(f'Invalid {label} centers: need finite [{T},{d}]')
    if errors is None or len(errors) != T:
        raise ValueError(f'{label} centers require per-token certified-error claims')
    return [_nonnegative(x) for x in errors]


def block_token_enclosure(block: nn.Module, radii: Sequence[Any], *,
                          pre_center: torch.Tensor | None = None,
                          mid_center: torch.Tensor | None = None,
                          pre_center_errors: Sequence[Any] | None = None,
                          mid_center_errors: Sequence[Any] | None = None) -> dict:
    """Upper bound on each token's output displacement *and* directional derivative.

    For radius vector r, the result bounds B(X)-B(X0) for any input row
    displacements <=r; it also bounds directional derivatives whose input
    row norms are <=r. One block's causal triangular structure is preserved.

    Optional LayerNorm centers are **claims** about the ideal clean trajectory.
    Verified error envelopes are a separate obligation, not established here.
    An omitted center falls back to the source-locked global LN inequality.
    """
    T = len(radii)
    if T < 1:
        raise ValueError('Sequence must contain at least one token')
    input_radius = [_nonnegative(x) for x in radii]
    d = int(block.hidden_size)
    ec1 = _validate_centers(pre_center, pre_center_errors, T=T, d=d, label='pre')
    ec2 = _validate_centers(mid_center, mid_center_errors, T=T, d=d, label='mid')
    if not isinstance(block.norm1, nn.LayerNorm) or not isinstance(block.norm2, nn.LayerNorm):
        raise ValueError('Expected per-token LayerNorm')
    # Also checks exact GELU, unbiased QKV, pre-norm convention and dimensions.
    block_bound(block, sequence=T)
    L1_global = layernorm_bounds(block.norm1)[0]
    L2_global = layernorm_bounds(block.norm2)[0]
    _, M = layernorm_bounds(block.norm1)
    L1 = [local_layernorm_lipschitz(block.norm1, pre_center[j], input_radius[j], center_error=ec1[j])
          if pre_center is not None else L1_global for j in range(T)]
    ly = [L1[j]*input_radius[j] for j in range(T)]
    heads, out = _head_matrices(block)
    head_norms = [(spectral_upper(q), spectral_upper(k), spectral_upper(v)) for q,k,v in heads]
    o = spectral_upper(out)
    alpha, beta = abs(rational(float(block.alpha))), abs(rational(float(block.beta)))
    mlp_linear = (Fraction(5,4) * spectral_upper(block.mlp[0].weight)
                  * spectral_upper(block.mlp[2].weight))
    middle: list[Fraction] = []
    outputs: list[Fraction] = []
    L2: list[Fraction] = []
    for i in range(T):
        U = sqrt_upper(sum((v*v for v in ly[:i+1]), Fraction(0)), digits=12)
        root_m = sqrt_upper(Fraction(i+1), digits=12)
        head_sq = Fraction(0)
        for q, k, v in head_norms:
            # DP=J_softmax DS; ||J_softmax||_2<=1/2.
            # Each row attends only to indices <=i. Correctly retains the
            # complete Q and K derivatives, including diagonal self-attention.
            head = v*U + (q*k*v*M*M/2)*sqrt_upper(Fraction(1,int(block.head_size)),digits=12)*(
                Fraction(i+1)*ly[i] + root_m*U)
            head_sq += head*head
        d_att = o * sqrt_upper(head_sq, digits=12)
        rmid = ceiling_decimal(alpha*input_radius[i] + beta*d_att, digits=12)
        middle.append(rmid)
        ln2 = (local_layernorm_lipschitz(block.norm2, mid_center[i], rmid,
                  center_error=ec2[i]) if mid_center is not None else L2_global)
        L2.append(ln2)
        outputs.append(ceiling_decimal((alpha + beta*mlp_linear*ln2)*rmid, digits=12))
    return {
        'output_radii': tuple(outputs),
        'middle_radii': tuple(middle),
        'norm1_lipschitz': tuple(L1),
        'norm2_lipschitz': tuple(L2),
        'method': 'tokenwise_local_ln' if (pre_center is not None or mid_center is not None)
                  else 'tokenwise_source_global',
        'state_enclosure_independently_validated': False,
    }


def visit_token_bounds(model: nn.Module, *, sequence: int,
                       chart: torch.Tensor, axes: torch.Tensor,
                       amplitude: Any, observed_tokens: Sequence[int],
                       injected_token: int,
                       injected_after_visit: int = 0,
                       pre_centers: Sequence[torch.Tensor] | None = None,
                       mid_centers: Sequence[torch.Tensor] | None = None,
                       pre_center_errors: Sequence[Sequence[Any]] | None = None,
                       mid_center_errors: Sequence[Sequence[Any]] | None = None) -> dict:
    """RMP source-global or conditional center-local angular pair bounds.

    The per-token radius is simultaneously a bound on the perturbation-disk
    excursion and on the angular directional derivative (both start <=a||A||).
    RMP input re-injection adds a fixed token embedding; it changes the clean
    centers but never the difference/derivative propagation.
    """
    if model.training:
        raise ValueError('Require eval() and deterministic RMP model')
    if not hasattr(model, 'blocks') or not hasattr(model,'module_indices'):
        raise ValueError('Expected RMP model with explicit schedule')
    visits = len(model.module_indices)
    if not (0 <= injected_after_visit < visits and 0 <= injected_token < sequence):
        raise ValueError('Invalid injection visit or token')
    obs = tuple(int(x) for x in observed_tokens)
    if len(obs) < 2 or len(set(obs)) != len(obs) or any(x < 0 or x >= sequence for x in obs):
        raise ValueError('Observed tokens must be distinct and within sequence')
    if chart.shape != axes.shape or chart.ndim != 2 or chart.shape[1]!=2:
        raise ValueError('Chart and axes must be [hidden,2]')
    if not (torch.isfinite(chart).all() and torch.isfinite(axes).all()):
        raise ValueError('Nonfinite axes or chart')
    d = chart.shape[0]
    if any(int(idx)<0 or int(idx)>=len(model.blocks) for idx in model.module_indices):
        raise ValueError('Invalid module index')
    def optional_schedule(seq, label):
        if seq is None:
            return None
        if len(seq) != visits:
            raise ValueError(f'{label} expected one entry per visit')
        return seq
    pcs=optional_schedule(pre_centers,'pre centers')
    mcs=optional_schedule(mid_centers,'mid centers')
    pes=optional_schedule(pre_center_errors,'pre center errors')
    mes=optional_schedule(mid_center_errors,'mid center errors')
    if (pcs is None)!=(pes is None) or (mcs is None)!=(mes is None):
        raise ValueError('Center arrays require error claims')
    bound_rows=[]
    detailed=[]
    radii=[Fraction(0) for _ in range(sequence)]
    chart_norm=spectral_upper(chart)
    patch_radius=_nonnegative(amplitude)*spectral_upper(axes)
    for visit,idx in enumerate(model.module_indices):
        diagnostics={}
        if visit == injected_after_visit:
            radii[injected_token]=patch_radius
        if visit > injected_after_visit:
            block=model.blocks[idx]
            if int(block.hidden_size) != d:
                raise ValueError('Mismatch in model hidden dimension')
            step=block_token_enclosure(block,radii,
                pre_center=pcs[visit] if pcs is not None else None,
                mid_center=mcs[visit] if mcs is not None else None,
                pre_center_errors=pes[visit] if pes is not None else None,
                mid_center_errors=mes[visit] if mes is not None else None)
            radii=list(step['output_radii'])
            diagnostics={'norm1_max':str(max(step['norm1_lipschitz'])),
                         'norm2_max':str(max(step['norm2_lipschitz'])),
                         'method':step['method']}
        if visit < injected_after_visit:
            row={f'{i},{j}':'0' for i,j in combinations(range(len(obs)),2)}
        else:
            row={f'{i},{j}':str(ceiling_decimal(chart_norm*(radii[obs[i]]+radii[obs[j]]),digits=12))
                 for i,j in combinations(range(len(obs)),2)}
        bound_rows.append(row)
        detailed.append({'visit':visit+1,'module_index':int(idx),
                         'token_radii':tuple(str(z) for z in radii),
                         **diagnostics})
    use_centers=pcs is not None or mcs is not None
    return {'schema':'scm07_tokenwise_real_rmp_v1',
            'bound_scope': 'conditional_on_external_ideal_center_enclosures' if use_centers else 'source_global_unconditional',
            'continuous_real_model_only':True, 'floating_execution_certified':False,
            'center_enclosures_verified_by_this_program':False,
            'source_architecture':'RMP_ExplicitDecoderBlock',
            'amplitude':str(_nonnegative(amplitude)),
            'per_visit_pair_bounds':bound_rows,
            'per_visit_tokens':detailed,
            'chart_operator_upper':str(chart_norm),
            'axes_operator_upper':str(spectral_upper(axes))}

@torch.no_grad()
def capture_float_clean_centers(model: nn.Module, initial: torch.Tensor, *,
                                fixed_post_visit: torch.Tensor | None = None) -> dict:
    """Diagnostic-only pre/mid clean centers, acquired from PyTorch forward hooks.

    The returned tensors are *not certified enclosures* of exact real states.
    For RMP FF-U, `fixed_post_visit=model.embed(tokens)` mirrors its fixed
    per-visit input reinjection when `initial=model.embed(tokens)`.
    """
    if model.training or initial.ndim!=3 or initial.shape[0]!=1:
        raise ValueError('Require eval model and one [1,T,H] input')
    if fixed_post_visit is not None and fixed_post_visit.shape != initial.shape:
        raise ValueError('Input reinjection must match initial shape')
    state=initial
    pre=[]; mid=[]; post=[]
    for idx in model.module_indices:
        block=model.blocks[idx]
        found=[]
        def tap(_mod,args):
            found.append(args[0][0].detach().cpu().clone())
        hook=block.norm2.register_forward_pre_hook(tap)
        try:
            pre.append(state[0].detach().cpu().clone())
            state=block(state)
        finally:
            hook.remove()
        if len(found)!=1:
            raise RuntimeError('Norm2 was not invoked exactly once in the block')
        mid.append(found[0])
        if fixed_post_visit is not None:
            state=state+fixed_post_visit
        post.append(state[0].detach().cpu().clone())
    return {'pre':pre, 'mid':mid, 'post':post, 'source':'uncertified_float_PyTorch_forward',
            'certified_center_errors':None}
