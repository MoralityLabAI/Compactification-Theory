"""SCM-06: sound global Lipschitz upper bounds for RMP's real-valued decoder.

The certificate treats stored floating-point weights as exact real constants.
It does NOT certify the discontinuous floating-point implementation, finite
precision activation capture, checkpoint loading, or the input corpus.

Only RMP's supported pre-norm, explicit attention + exact GELU architecture
is accepted. Every matrix norm upper bound uses exact integer arithmetic on
the binary floats underlying a fixed tensor. Transcendental inequalities are
replaced by conservative rational bounds.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isfinite, isqrt
from typing import Any

import torch
from torch import nn


def rational(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a valid numeric upper bound')
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError('Nonfinite parameter')
    return Fraction(value)


def sqrt_upper(value: Fraction, *, digits: int = 10) -> Fraction:
    """Exact outward decimal-grid ceiling of a nonnegative rational square root."""
    value = rational(value)
    if value < 0 or not (0 <= digits <= 20):
        raise ValueError('Invalid square-root argument')
    base = 10 ** digits
    floor = isqrt((value.numerator * base * base) // value.denominator)
    is_exact = floor * floor * value.denominator == value.numerator * base * base
    return Fraction(floor if is_exact else floor + 1, base)


def sqrt_lower(value: Fraction, *, digits: int = 10) -> Fraction:
    value = rational(value)
    if value < 0 or not (0 <= digits <= 20):
        raise ValueError('Invalid square-root argument')
    base = 10 ** digits
    return Fraction(isqrt((value.numerator * base * base) // value.denominator), base)


def ceiling_decimal(value: Fraction, *, digits: int = 10) -> Fraction:
    """Discard insignificant fractions only in the conservative direction."""
    value = rational(value)
    base = 10 ** digits
    return Fraction(-(-(value.numerator * base) // value.denominator), base)


def spectral_upper(matrix: torch.Tensor) -> Fraction:
    """Exact bound min(Frobenius, sqrt(induced-1 * induced-infinity)).

    The tensor's stored binary floats are interpreted as rational numbers.
    Sums of squares and absolute row/column sums are integers after scaling
    by a shared dyadic denominator; sqrt uses an integer ceiling. Thus the
    result is an exact rational, valid even with subnormal coefficients.
    """
    if matrix.ndim != 2 or matrix.numel() == 0 or not matrix.is_floating_point():
        raise ValueError('Expected nonempty real floating-point matrix')
    rows = matrix.detach().cpu().tolist()
    ratios = []
    exponent = 0
    for row in rows:
        items = []
        for value in row:
            if not isfinite(value):
                raise ValueError('Nonfinite checkpoint matrix')
            numerator, denominator = float(value).as_integer_ratio()
            shift = denominator.bit_length() - 1
            if denominator != 1 << shift:
                raise ArithmeticError('Floating-point coefficient is not dyadic')
            exponent = max(exponent, shift)
            items.append((numerator, shift))
        ratios.append(items)
    scaled_rows = []
    col_sums = [0] * len(rows[0])
    sum_squares = 0
    for row in ratios:
        absolute = [abs(num) << (exponent - shift) for num, shift in row]
        scaled_rows.append(sum(absolute))
        for j, number in enumerate(absolute):
            col_sums[j] += number
            sum_squares += number * number
    norm_product = max(scaled_rows) * max(col_sums)
    sq = min(norm_product, sum_squares)
    root = isqrt(sq)
    if root * root < sq:
        root += 1
    return Fraction(root, 1 << exponent)


def vector_upper(tensor: torch.Tensor) -> Fraction:
    if tensor.ndim != 1:
        raise ValueError('Need vector')
    return spectral_upper(tensor.reshape(1, -1))


def _gamma_beta(norm: nn.Module) -> tuple[Fraction, Fraction, int, Fraction]:
    if not isinstance(norm, nn.LayerNorm):
        raise ValueError('Only nn.LayerNorm is supported')
    if len(norm.normalized_shape) != 1 or not norm.elementwise_affine:
        raise ValueError('Need per-token 1D affine LayerNorm')
    if norm.weight is None:
        raise ValueError('Missing LayerNorm gamma')
    gamma = norm.weight.detach().cpu().reshape(-1).tolist()
    if not gamma or any(not isfinite(x) for x in gamma):
        raise ValueError('Invalid LayerNorm gamma')
    max_gamma = max(abs(rational(float(x))) for x in gamma)
    beta = Fraction(0) if norm.bias is None else vector_upper(norm.bias)
    eps = rational(float(norm.eps))
    if eps <= 0:
        raise ValueError('LayerNorm epsilon must be positive')
    return max_gamma, beta, len(gamma), eps


def layernorm_bounds(norm: nn.Module) -> tuple[Fraction, Fraction]:
    """Global spectral Jacobian bound and global image row-norm bound.

    ||D LN(x)||_2 <= ||gamma||_infinity / sqrt(eps), and
    ||LN(x)||_2 <= sqrt(d) ||gamma||_infinity + ||beta||_2.
    Both statements are valid for any input x in real arithmetic.
    """
    gamma, beta, dimension, eps = _gamma_beta(norm)
    return (ceiling_decimal(gamma * sqrt_upper(1 / eps)),
            ceiling_decimal(sqrt_upper(Fraction(dimension)) * gamma + beta))


def _head_matrices(block: nn.Module) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    # Matches research_gym.neural.looped_decoder.ExplicitDecoderBlock.
    if not all(hasattr(block, key) for key in ('qkv', 'attention_out', 'num_heads', 'head_size', 'hidden_size')):
        raise ValueError('Expected RMP ExplicitDecoderBlock with qkv and attention_out')
    projection = block.qkv
    output = block.attention_out
    if not isinstance(projection, nn.Linear) or not isinstance(output, nn.Linear):
        raise ValueError('Expected linear qkv and output projections')
    if projection.bias is not None or output.bias is not None:
        raise ValueError('Biased QKV or attention output unsupported')
    d, h, head_dim = int(block.hidden_size), int(block.num_heads), int(block.head_size)
    if d != h * head_dim or projection.weight.shape != (3*d, d) or output.weight.shape != (d, d):
        raise ValueError('Unexpected explicit attention dimensions')
    weight = projection.weight.detach()
    heads = []
    for k in range(h):
        interval = slice(k * head_dim, (k + 1) * head_dim)
        heads.append((weight[interval], weight[d + k*head_dim:d+(k+1)*head_dim],
                      weight[2*d + k*head_dim:2*d+(k+1)*head_dim]))
    return heads, output.weight


def block_bound(block: nn.Module, *, sequence: int) -> dict[str, Fraction]:
    """Conservative global Frobenius-to-Frobenius Lipschitz factor.

    Each causal attention row is a probability simplex, so ||P||_2<=sqrt(T);
    its softmax Jacobian satisfies ||DP||_2<=1/2. LayerNorm's output is
    globally bounded, which permits a GLOBAL attention Lipschitz bound.
    """
    if not isinstance(sequence, int) or sequence < 1:
        raise ValueError('Sequence length must be positive')
    if not all(hasattr(block, v) for v in ('norm1', 'norm2', 'mlp', 'alpha', 'beta')):
        raise ValueError('Expected RMP pre-norm block')
    if getattr(block, 'normalization', 'pre') != 'pre':
        raise ValueError('Only pre-norm blocks are supported')
    if not isinstance(block.mlp, nn.Sequential) or len(block.mlp) != 3:
        raise ValueError('Unexpected RMP MLP')
    up, gelu, down = tuple(block.mlp)
    if not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
        raise ValueError('Expected linear GELU linear')
    if not isinstance(gelu, nn.GELU) or gelu.approximate != 'none':
        raise ValueError('Only exact erf GELU is supported')
    ln1, input_row_bound = layernorm_bounds(block.norm1)
    ln2, _ = layernorm_bounds(block.norm2)
    heads, out_weight = _head_matrices(block)
    root_T = sqrt_upper(Fraction(sequence))
    head_dim = int(block.head_size)
    inverse_root_d = sqrt_upper(Fraction(1, head_dim))
    heads_squared = Fraction(0)
    for wq, wk, wv in heads:
        qnorm, knorm, vnorm = spectral_upper(wq), spectral_upper(wk), spectral_upper(wv)
        # The 1/2 softmax-Jacobian bound cancels the factor 2 in
        # D(Q K^T)=DQ K^T + Q DK^T.
        head = root_T*vnorm + sequence * input_row_bound**2 * qnorm*knorm*vnorm * inverse_root_d
        heads_squared += head**2
    L_attention = ceiling_decimal(ln1 * spectral_upper(out_weight) * sqrt_upper(heads_squared))
    # Exact erf GELU has |GELU'(x)| <= 1 + 1/sqrt(2*pi*e) < 5/4.
    L_mlp = ceiling_decimal(ln2 * spectral_upper(down.weight) * Fraction(5,4)
                            * spectral_upper(up.weight))
    alpha, beta = abs(rational(float(block.alpha))), abs(rational(float(block.beta)))
    L_full = ceiling_decimal((alpha + beta*L_attention) * (alpha + beta*L_mlp))
    return {'attention': L_attention, 'mlp': L_mlp, 'full': L_full,
            'norm1': ln1, 'norm2': ln2, 'normalized_row': input_row_bound}


def _real_finite(value: Any) -> Fraction:
    val = rational(value)
    if val < 0:
        raise ValueError('Negative magnitude/radius')
    return val


def visit_bounds(model: nn.Module, *, sequence: int, chart: torch.Tensor,
                 axes: torch.Tensor, amplitude: Any, observed_tokens,
                 injected_token: int, injected_after_visit: int = 0) -> dict:
    """Return exact upper pairwise angular derivative bounds for all visits.

    `amplitude` is fixed for a task: intervention radius * baseline token RMS.
    Re-injecting a constant embedding each visit leaves derivatives unchanged.
    `model` is the RMP AnalysisModel.model (LoopedDecoderLM), in eval mode.
    """
    if model.training:
        raise ValueError('Model must be eval() to exclude train-time randomness')
    if not hasattr(model, 'blocks') or not hasattr(model, 'module_indices'):
        raise ValueError('Expected RMP LoopedDecoderLM model')
    if chart.ndim != 2 or axes.shape != chart.shape or chart.shape[1] != 2:
        raise ValueError('Chart and axes must both have [hidden,2]')
    if not (torch.isfinite(chart).all() and torch.isfinite(axes).all()):
        raise ValueError('Nonfinite chart or intervention axes')
    n = len(tuple(observed_tokens))
    observed = tuple(int(x) for x in observed_tokens)
    if len(set(observed)) != n or n < 2 or any(x < 0 or x >= sequence for x in observed):
        raise ValueError('Need distinct observed positions within the sequence')
    if injected_token < 0 or injected_token >= sequence or injected_after_visit < 0:
        raise ValueError('Invalid injection position')
    if any(index >= len(model.blocks) or index < 0 for index in model.module_indices):
        raise ValueError('Invalid model module indices')
    nvisits = len(model.module_indices)
    if injected_after_visit >= nvisits:
        raise ValueError('Injection after last visit')
    input_perturbation = ceiling_decimal(_real_finite(amplitude) * spectral_upper(axes))
    chart_pair = sqrt_upper(Fraction(2)) * spectral_upper(chart)
    pair_bound = ceiling_decimal(chart_pair * input_perturbation)
    factors: list[dict[str, str]] = []
    bounds: list[dict[str, str]] = []
    pairs = list(combinations(range(n),2))
    pairnames = [f'{i},{j}' for i,j in pairs]
    for visit, idx in enumerate(model.module_indices):
        if visit > injected_after_visit:
            calculation = block_bound(model.blocks[idx], sequence=sequence)
            factors.append({'visit': visit+1, 'module_index':idx,
                            **{k: str(v) for k,v in calculation.items()}})
            pair_bound = ceiling_decimal(pair_bound * calculation['full'])
        current = {}
        for i,j in pairs:
            if visit < injected_after_visit or \
                    (visit == injected_after_visit and injected_token not in observed) or \
                    (observed[i] < injected_token and observed[j] < injected_token):
                # RMP's attention is causal and all other operations are
                # token-local. Earlier positions cannot see an injection in
                # the future. This is an exact structural zero independent
                # of the norms of the checkpoint matrices.
                upper = Fraction(0)
            else:
                upper = pair_bound
            current[f'{i},{j}'] = str(upper)
        bounds.append(current)
    return {'schema':'scm06_real_rmp_lipschitz_v1',
            'arithmetic_scope':'exact_stored_weight_coefficients_as_real_numbers',
            'floating_execution_certified':False,
            'method':'exact_dyadic_matrix_norm_and_global_smooth_layer_bounds',
            'amplitude':str(rational(amplitude)),
            'axes_operator_upper':str(spectral_upper(axes)),
            'chart_operator_upper':str(spectral_upper(chart)),
            'per_visit_pair_bounds':bounds,
            'per_block_factors':factors}


def topology_zero_disk_certificate(clean_by_visit, *, pair_bounds_result: dict, margin='0') -> list[dict]:
    """Give a sufficient disk criterion conditional on exact clean chart centers.

    The angular bound is also a uniform disk perturbation bound because
    ||A v|| <= ||A|| for every radius coordinate ||v||<=1, followed by the
    same global Lipschitz products. A positive certificate proves the
    boundary pure braid is null-homotopic in the *real-valued* decoder.
    """
    observed_frames = clean_by_visit
    bounds = pair_bounds_result['per_visit_pair_bounds']
    if len(observed_frames) != len(bounds):
        raise ValueError('Clean baseline and model visit count mismatch')
    eps = _real_finite(margin)
    result=[]
    for visit, (points, per_pair) in enumerate(zip(observed_frames,bounds),1):
        n=len(points)
        checks=[]
        for i,j in combinations(range(n),2):
            if len(points[i])!=2 or len(points[j])!=2:
                raise ValueError('Bad 2D clean chart')
            u=rational(points[i][0])-rational(points[j][0])
            v=rational(points[i][1])-rational(points[j][1])
            lower=sqrt_lower(u*u+v*v)
            excursion=rational(per_pair[f'{i},{j}'])
            checks.append({'pair':f'{i},{j}', 'baseline_distance_lower':str(lower),
                           'max_disk_excursion_upper':str(excursion),
                           'certified':bool(lower>excursion+eps)})
        result.append({'visit':visit,'conditional_null_if_clean_baseline_exact':all(k['certified'] for k in checks),
                       'assumes_exact_clean_chart_values':True,'pair_checks':checks})
    return result


# 2*pi < 44/7. It is a rigorous, deliberately simple bound on exact angular
# intervals 2*pi/N, unlike differences of approximate binary-float angles.
def angular_gap_upper(n_angles: int) -> Fraction:
    if not isinstance(n_angles, int) or n_angles < 1:
        raise ValueError('Positive number of angular intervals required')
    return Fraction(44,7*n_angles)
