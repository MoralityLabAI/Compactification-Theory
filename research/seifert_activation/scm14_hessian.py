"""SCM-14: source-level, tokenwise second-derivative bounds for RMP pre-LN.

The domain is a CLOSED two-dimensional intervention disk parametrized by
u in R^2, ||u|| <= 1, injected as `amplitude * axes @ u` after one visit.
Each J_i bounds ||D h_i(u)[v]||, and each H_i bounds
||D^2 h_i(u)[v,w]|| for arbitrary unit vectors v,w and every u in the disk.

The source graph is the exact-real interpretation of RMP's frozen binary
weights with pre-LayerNorm, causal softmax attention and exact GELU. Numeric
center-ball premises may come only from a verified SCM-10 BallTubes object;
its floating/BLAS verification assumptions remain in force. The program
never certifies physical CUDA/FP32 execution or trained-model observations.

No finite-difference estimate is used to establish a derivative supremum.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import torch
from torch import nn

from ball_centered import BallTubes, _block_norms, _certified_qkv_reference
from rmp_lipschitz import (rational, sqrt_upper, sqrt_lower, ceiling_decimal,
                           spectral_upper, _gamma_beta, block_bound)


def _finite_nonnegative(value) -> Fraction:
    r = rational(value)
    if r < 0:
        raise ValueError('Expected nonnegative source-certified magnitude')
    return r


def layernorm_jet_bounds(ln: nn.LayerNorm, center: torch.Tensor,
                         full_disk_radius) -> tuple[Fraction, Fraction]:
    """Bound ||D LN|| and ||D² LN|| on an entire Euclidean center ball.

    For a=Px and s²=epsilon+||a||²/d:
       ||D LN|| <= gamma_inf / s,
       ||D² LN|| <= gamma_inf * (3||a||/(d s³)
                                  +3||a||³/(d² s⁵)).
    The global second derivative inequality
       ||D² LN|| <= 4 gamma_inf / (sqrt(3d) epsilon)
    is also valid and is used if tighter. Every number is conservatively
    rounded up using rational arithmetic and exact stored float values.
    """
    if center.ndim != 1 or not center.is_floating_point() or not torch.isfinite(center).all():
        raise ValueError('Expected finite one-token floating point center')
    gamma, _, d, eps = _gamma_beta(ln)
    if d != center.numel():
        raise ValueError('LayerNorm and source-center dimension mismatch')
    radius = _finite_nonnegative(full_disk_radius)
    values = [rational(float(v)) for v in center.detach().cpu().tolist()]
    mean = sum(values, Fraction(0)) / d
    norm2 = sum(((v - mean) ** 2 for v in values), Fraction(0))
    nlo, nhi = sqrt_lower(norm2, digits=20), sqrt_upper(norm2, digits=20)
    A_lo = max(Fraction(0), nlo - radius)
    A_hi = nhi + radius
    s_lo = sqrt_lower(eps + A_lo * A_lo / d, digits=20)
    if s_lo <= 0:
        raise ArithmeticError('Missing positive LayerNorm denominator lower bound')
    first = ceiling_decimal(gamma / s_lo, digits=18)
    local_second = gamma * (3 * A_hi / (d * s_lo**3)
                           + 3 * A_hi**3 / (d*d*s_lo**5))
    global_second = 4 * gamma * sqrt_upper(Fraction(1, 3*d), digits=20) / eps
    second = ceiling_decimal(min(local_second, global_second), digits=18)
    return first, second


@dataclass(frozen=True)
class ScalarJet:
    """Uniform spectral/Jacobian and bilinear/Hessian operator radii."""
    j: Fraction
    h: Fraction

    def __post_init__(self):
        if self.j < 0 or self.h < 0:
            raise ValueError('Negative derivative bound')


def softmax_jet_bound(score_jet: ScalarJet) -> ScalarJet:
    """Scores measured in ||.||_infty; probabilities in ||.||_1.

    ||D softmax(s) a||_1 <= ||a||_infty,
    ||D² softmax(s)[a,b]||_1 <= 2||a||_infty||b||_infty.
    The chain rule adds the score Hessian to the second bound.
    """
    return ScalarJet(score_jet.j, score_jet.h + 2 * score_jet.j**2)


def head_jet_bound(*, qnorm, knorm, vnorm, normalized_max,
                   query: ScalarJet, keys: ScalarJet, values: ScalarJet,
                   head_dim: int, q_value_upper=None, k_value_upper=None,
                   v_value_upper=None) -> ScalarJet:
    """Uniform causal attention head J/H for one query and its key prefix.

    `query`, `keys`, `values` bound the normalized-then-projected Q/K/V
    derivative jets; `qnorm`, etc. are verified spectral norms of their
    weight matrices. Baseline Q/K/V norms are bounded by W_norm*M.
    """
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError('Head dimension must be positive')
    qn, kn, vn = map(_finite_nonnegative, (qnorm, knorm, vnorm))
    M = _finite_nonnegative(normalized_max)
    # Inputs `query`, `keys`, `values` are derivatives of LayerNorm *states*;
    # multiply by W norms here for each head.
    dq, dk, dv = qn*query.j, kn*keys.j, vn*values.j
    hq, hk, hv = qn*query.h, kn*keys.h, vn*values.h
    root_inv = sqrt_upper(Fraction(1, head_dim), digits=18)
    Q = min(qn*M, _finite_nonnegative(q_value_upper)) if q_value_upper is not None else qn*M
    K = min(kn*M, _finite_nonnegative(k_value_upper)) if k_value_upper is not None else kn*M
    R = min(vn*M, _finite_nonnegative(v_value_upper)) if v_value_upper is not None else vn*M
    score_j = root_inv * (dq * K + Q * dk)
    score_h = root_inv * (hq * K + 2 * dq * dk + Q * hk)
    probs = softmax_jet_bound(ScalarJet(score_j, score_h))
    return ScalarJet(
        dv + R * probs.j,
        hv + 2 * probs.j * dv + R * probs.h,
    )


def block_jet_bounds(block: nn.Module, norms, pre: torch.Tensor,
                     pre_center_errors: Sequence, incoming: Sequence[ScalarJet],
                     mid: torch.Tensor, mid_center_errors: Sequence,
                     *, pre_response: Sequence | None = None,
                     min_round_digits: int = 14,
                     source_local_values: bool = True,
                     ) -> tuple[ScalarJet, ...]:
    """Full-disk J/H at output of a supported RMP explicit pre-LN block.

    The incoming-state disk diameter is bounded by J_i (center = clean
    u=0 state) via the mean value theorem. An independently source-certified
    tighter full-disk response bound may be supplied from SCM-11. Both feed
    the local LayerNorm denominator, with the smaller proven enclosure used.
    """
    n = len(incoming)
    if n < 1 or len(pre_center_errors) != n or len(mid_center_errors) != n:
        raise ValueError('Invalid source-center lengths')
    if pre.shape != mid.shape or pre.ndim != 2 or pre.shape[0] != n:
        raise ValueError('Expected matching [sequence, hidden] source centers')
    if pre_response is not None and len(pre_response) != n:
        raise ValueError('Invalid SCM11 response length')
    if min_round_digits < 0 or min_round_digits > 18:
        raise ValueError('Unsupported precision')
    pnorm = []
    first_enclosures = []
    first_lipschitz = []
    for i in range(n):
        response = incoming[i].j
        if pre_response is not None:
            response = min(response, _finite_nonnegative(pre_response[i]))
        l1, l2 = layernorm_jet_bounds(block.norm1, pre[i],
                                     _finite_nonnegative(pre_center_errors[i]) + response)
        first_enclosures.append(_finite_nonnegative(pre_center_errors[i]) + response)
        first_lipschitz.append(l1)
        pnorm.append(ScalarJet(l1 * incoming[i].j,
                               l1 * incoming[i].h + l2 * incoming[i].j**2))
    a = abs(norms.alpha)
    b = abs(norms.beta)
    qkv = None
    if source_local_values:
        from fast_centered import _weights
        qkv = _certified_qkv_reference(block, pre, _weights(block))
    source_caps = []
    if qkv is not None:
        # Cache per-head Q values and causal prefix K/V suprema. Computing
        # exact rational expressions inside every prefix would be quadratic
        # and needlessly expensive at the original 64-token RMP shape.
        for h,(qn,kn,vn) in enumerate(norms.qkv_heads):
            q_bound = []
            k_bound = []
            v_bound = []
            cumulative_k = Fraction(0)
            cumulative_v = Fraction(0)
            for j in range(n):
                normalized_error = first_lipschitz[j]*first_enclosures[j]
                q_bound.append(min(qn*norms.max_normalized,
                                   qkv[h][0][j]+qn*normalized_error))
                cumulative_k=max(cumulative_k,min(kn*norms.max_normalized,
                                   qkv[h][1][j]+kn*normalized_error))
                cumulative_v=max(cumulative_v,min(vn*norms.max_normalized,
                                   qkv[h][2][j]+vn*normalized_error))
                k_bound.append(cumulative_k)
                v_bound.append(cumulative_v)
            source_caps.append((q_bound,k_bound,v_bound))
    inv_cache = []
    for i in range(n):
        prefix = pnorm[:i+1]
        keys = ScalarJet(max(x.j for x in prefix),max(x.h for x in prefix))
        attn = []
        for h,(qn,kn,vn) in enumerate(norms.qkv_heads):
            extras = {}
            if source_caps:
                qs,ks,vs=source_caps[h]
                extras={'q_value_upper':qs[i],
                        'k_value_upper':ks[i],
                        'v_value_upper':vs[i]}
            attn.append(head_jet_bound(qnorm=qn,knorm=kn,vnorm=vn,
                         normalized_max=norms.max_normalized,query=pnorm[i],
                         keys=keys,values=keys,head_dim=norms.head_dim,
                         **extras))
        attn_j = norms.output * sqrt_upper(sum(x.j**2 for x in attn), digits=min_round_digits)
        attn_h = norms.output * sqrt_upper(sum(x.h**2 for x in attn), digits=min_round_digits)
        mj = a * incoming[i].j + b * attn_j
        mh = a * incoming[i].h + b * attn_h
        # J_mid bounds the ENTIRE difference of mid(u)-mid(0) on unit disk.
        l1, l2 = layernorm_jet_bounds(block.norm2,mid[i],
                         _finite_nonnegative(mid_center_errors[i])+mj)
        zn_j = l1*mj
        zn_h = l1*mh+l2*mj*mj
        up, down = norms.up, norms.down
        mlp_j = down * Fraction(5,4) * up * zn_j
        mlp_h = down * (Fraction(5,4)*up*zn_h + up*up*zn_j*zn_j)
        oj = ceiling_decimal(a*mj+b*mlp_j,digits=min_round_digits)
        oh = ceiling_decimal(a*mh+b*mlp_h,digits=min_round_digits)
        inv_cache.append(ScalarJet(oj,oh))
    return tuple(inv_cache)


def hessian_disk_bounds(model: nn.Module, tubes: BallTubes, *,
                        axes: torch.Tensor, chart: torch.Tensor, amplitude,
                        injected_token: int, observed_tokens: Sequence[int],
                        inject_after_visit: int = 0,
                        use_scm11_response: bool = False,
                        source_local_values: bool = True,
                        ) -> dict:
    """End-to-end Hessian bounds on the normalized two-coordinate disk.

    Full intervention is `amplitude * axes @ u`, ||u||<=1, at the output
    of `inject_after_visit`. Thus the source J/H are derivatives with respect
    to normalized u, and a per-token chart remainder is H_chart/2.

    When `use_scm11_response`, the prior independently derived bound is
    recomputed from the SAME source model and verified clean tubes; it is
    never accepted as an untrusted JSON numeric assertion.
    """
    if model.training:
        raise ValueError('Require eval-mode source decoder')
    if (tubes.method != 'scm10_ieee_cpu_roundoff_plus_causal_L2_balls_exact_real'
        or tubes.input_scope != 'ideal_real_with_exact_dyadic_frequency_table'):
        raise ValueError('Source-validated SCM-10 clean tubes required')
    if len(model.module_indices) != len(tubes.pre_centers) or len(model.module_indices) > 8:
        raise ValueError('Source visit-count mismatch')
    n,d = tubes.pre_centers[0].shape
    if n > 64 or d > 128 or model.hidden_size != d:
        raise ValueError('Unsupported source model dimensions')
    if axes.shape != chart.shape or axes.shape != (d,2):
        raise ValueError('Expected two intervention axes and planar [hidden,2] chart')
    if not torch.isfinite(axes).all() or not torch.isfinite(chart).all():
        raise ValueError('Nonfinite measurement or intervention axes')
    if not 0 <= injected_token < n or not 0 <= inject_after_visit < len(model.module_indices):
        raise ValueError('Invalid injection token or visit')
    obs = tuple(int(i) for i in observed_tokens)
    if len(obs) < 2 or len(obs) != len(set(obs)) or any(i<0 or i>=n for i in obs):
        raise ValueError('Invalid observed tokens')
    amp = _finite_nonnegative(amplitude)
    if amp > 1:
        raise ValueError('Amplitude exceeds registered normalized-disk source budget')
    for idx in set(model.module_indices):
        block_bound(model.blocks[idx],sequence=n)
    tighter = None
    if use_scm11_response:
        from refined_response import refined_disk_bounds
        report = refined_disk_bounds(model,tubes,chart=chart,axes=axes,
                  amplitude=amp,injected_token=injected_token,
                  observed_tokens=obs,inject_after_visit=inject_after_visit)
        tighter = tuple(tuple(_finite_nonnegative(x) for x in row)
                        for row in report['visit_response_radii'])
        if len(tighter) != len(model.module_indices):
            raise ValueError('Source-response visit mismatch')
    scale = amp * spectral_upper(axes)
    chart_norm = spectral_upper(chart)
    jets = [ScalarJet(Fraction(0),Fraction(0)) for _ in range(n)]
    norm_cache = {idx:_block_norms(model.blocks[idx]) for idx in set(model.module_indices)}
    records=[]
    for visit,idx in enumerate(model.module_indices):
        # The input patch occurs at the completed output of one visit.
        if visit > inject_after_visit:
            pre_resp = tighter[visit-1] if tighter is not None else None
            jets = list(block_jet_bounds(model.blocks[idx], norm_cache[idx],
                         tubes.pre_centers[visit],tubes.pre_center_errors[visit],jets,
                         tubes.mid_centers[visit],tubes.mid_center_errors[visit],
                         pre_response=pre_resp,
                         source_local_values=source_local_values))
        if visit == inject_after_visit:
            jets[injected_token]=ScalarJet(jets[injected_token].j+scale,
                                          jets[injected_token].h)
        report={str(i):{
            'first_derivative_upper':str(ceiling_decimal(chart_norm*jets[i].j,digits=14)),
            'second_derivative_upper':str(ceiling_decimal(chart_norm*jets[i].h,digits=14)),
            'taylor_remainder_on_unit_disk_upper':str(ceiling_decimal(chart_norm*jets[i].h/2,digits=14))
          } for i in obs}
        records.append({'visit':visit+1,'observed':report,
                        'causal_prefix_zero':all(jets[k].j==jets[k].h==0 for k in range(injected_token))})
    return {'schema':'scm14/two_coordinate_source_hessian/v1',
            'scope':'ideal_real_RMP_weights_conditional_CPU_binary64_BLAS_center_verifier',
            'intervention_coordinate':'u_in_unit_disk; patch=amplitude*axes@u',
            'amplitude':str(amp),
            'chart_norm_upper':str(chart_norm),
            'source_response_bound':'recomputed_SCM11' if tighter is not None else 'integrated_J_bound',
            'qkv_value_bounds':'validated_QKV_centers_plus_state_balls' if source_local_values else 'global_LN_image',
            'observed_token_positions':list(obs),
            'per_visit':records,
            'hessian_upper_certified_under_source_contract':True,
            'exact_real_center_and_jacobian_exported':False,
            'float32_cuda_execution_certified':False,
            'trained_rmp_model_measured':False,
            'four_dimensional_surface_invariant':False}


def verify_hessian_from_source(model: nn.Module, source, *,
                               fixed_input_reinjection: bool,
                               axes: torch.Tensor, chart: torch.Tensor,
                               amplitude, injected_token: int,
                               observed_tokens: Sequence[int],
                               inject_after_visit: int = 0,
                               source_local_values: bool = True) -> dict:
    """High-assurance entry: recompute clean state bounds from original source.

    This entry is for local research runs. It obtains BallTubes from the
    checkpoint weights + exact input enclosure rather than accepting a
    manually constructed BallTubes carrying only a self-declared method tag.
    The fixed input-reinjection policy is explicit and part of provenance.
    """
    from ball_centered import verified_ball_tubes
    tubes = verified_ball_tubes(model, source,
                fixed_input_reinjection=fixed_input_reinjection,
                max_seq=64, max_dim=128, max_visits=8)
    result = hessian_disk_bounds(model,tubes,axes=axes,chart=chart,
            amplitude=amplitude,injected_token=injected_token,
            observed_tokens=observed_tokens,
            inject_after_visit=inject_after_visit,
            source_local_values=source_local_values)
    result['source_preimage'] = 'recomputed_from_checkpoint_weights_and_source_input'
    result['fixed_input_reinjection'] = bool(fixed_input_reinjection)
    result['frequency_sha256'] = tubes.frequency_sha256
    return result
