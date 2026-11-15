"""OLM-08: dependency-free exact rational global derivative tube bounds.

Weight arrays should consist of exact binary floating-point *values* read
from a hash-verified frozen checkpoint; all norm inequalities use Fraction.
The returned bounds describe the exact-real mathematical network, excluding
finite-precision kernel arithmetic.
"""
from fractions import Fraction
from math import isqrt


def rational(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        return Fraction.from_float(x)
    return Fraction(x)


def sqrt_upper(q, bits=32):
    """Dyadic upper bound sqrt(q), proved by exact integer comparison."""
    q = rational(q)
    if q < 0 or bits < 1:
        raise ValueError("sqrt requires nonnegative rational and positive precision")
    scaled = q.numerator << (2 * bits)
    root = isqrt(scaled // q.denominator)
    if root * root * q.denominator < scaled:
        root += 1
    upper = Fraction(root, 1 << bits)
    assert upper * upper >= q
    assert upper <= q + 2 if q >= 1 else upper <= 2
    return upper


def frobenius_upper(rows, bits=32):
    rows = tuple(tuple(rational(v) for v in row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("nonempty rectangular matrix required")
    squares = sum((v * v for row in rows for v in row), Fraction(0))
    return sqrt_upper(squares, bits)


def linear_mlp_bounds(w1, w2, bits=32):
    """Bounds for x -> x + W2 GELU(W1 x + b1) + b2.

    Uses |GELU'| <= 5/4, |GELU''| <= 4/5 (exact erf GELU).
    """
    norm1 = frobenius_upper(w1, bits)
    norm2 = frobenius_upper(w2, bits)
    return (1 + Fraction(5, 4) * norm2 * norm1,
            Fraction(4, 5) * norm2 * norm1 * norm1)


def layernorm_bounds(gamma, eps, d=48, bits=32):
    """Global derivatives for population LayerNorm on d channels.

    ||DN|| <= ||gamma||inf / sqrt(eps)
    ||D2N|| <= 2||gamma||inf / (sqrt(3*d) * eps)
    """
    if d < 3:
        raise ValueError("d>=3 required for sharp LN Hessian formula")
    eps = rational(eps)
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    gain = max((abs(rational(v)) for v in gamma), default=Fraction(0))
    return (gain * sqrt_upper(1 / eps, bits),
            2 * gain * sqrt_upper(Fraction(1, 3 * d), bits) / eps)


def composition_bounds(layer_bounds, output_linear_norm):
    """Rigorous global norm bounds L=||DF||, M=||D2F||.

    For f_i with a_i>=||Df_i||, b_i>=||D2f_i||:
      A_i = a_i A_(i-1)
      B_i = b_i A_(i-1)^2 + a_i B_(i-1)
    """
    A = Fraction(1)
    B = Fraction(0)
    trace = []
    for a, b in layer_bounds:
        a, b = rational(a), rational(b)
        if a < 0 or b < 0:
            raise ValueError("derivative bounds must be nonnegative")
        B, A = b * A * A + a * B, a * A
        trace.append((A, B))
    w = rational(output_linear_norm)
    if w < 0:
        raise ValueError("linear readout norm must be nonnegative")
    return {"jacobian": w * A, "hessian": w * B, "trace": tuple(trace)}


def directional_interval(first_derivative_norm, r, M):
    """Guaranteed [lower,upper] of ||F(h+r u)-F(h)|| for ||u||=1."""
    a, r, M = rational(first_derivative_norm), rational(r), rational(M)
    if min(a, r, M) < 0:
        raise ValueError("nonnegative direction/radius/curvature required")
    linear = r * a
    remainder = M * r * r / 2
    return (max(Fraction(0), linear - remainder), linear + remainder)


def dominance_certificate(active_slope, control_slope, r, M):
    """Sufficient (not necessary) finite-radius active > control theorem."""
    a, b, r, M = map(rational, (active_slope, control_slope, r, M))
    if min(a, b, r, M) < 0:
        raise ValueError("negative parameters")
    # Both finite-radius effects are within M*r^2/2 of their linear slopes.
    return r * (a - b) > M * r * r


def source_model_global_certificate(weights, bits=32):
    """Weights: dict with W1/W2 for token and channel MLP, LN gamma, head.

    Names follow PyTorch state_dict fields of the locked 44K TRM-inspired
    reference model. Returns global *exact real-arithmetic* norm ceilings.
    """
    required = (
        'block.tok_mix.0.weight', 'block.tok_mix.2.weight',
        'block.chan_mix.0.weight', 'block.chan_mix.2.weight',
        'block.ln1.weight', 'block.ln2.weight', 'head.weight'
    )
    if any(k not in weights for k in required):
        raise ValueError("missing frozen tensor in source model")
    token = linear_mlp_bounds(weights[required[0]], weights[required[1]], bits)
    channel = linear_mlp_bounds(weights[required[2]], weights[required[3]], bits)
    ln1 = layernorm_bounds(weights[required[4]], Fraction.from_float(1e-5), 48, bits)
    ln2 = layernorm_bounds(weights[required[5]], Fraction.from_float(1e-5), 48, bits)
    readout = frobenius_upper(weights['head.weight'], bits)
    layers = (token, ln1, channel, ln2) * 3
    bounds = composition_bounds(layers, readout)
    return {
        "source_model": "44K frozen TRM-inspired 3-step continuation after stage1",
        "num_elementary_stages": 12,
        "weight_frobenius_ceilings": {
            "token_linear_a": str(token[0]), "token_hessian_b": str(token[1]),
            "channel_linear_a": str(channel[0]), "channel_hessian_b": str(channel[1]),
            "ln1_linear_a": str(ln1[0]), "ln1_hessian_b": str(ln1[1]),
            "ln2_linear_a": str(ln2[0]), "ln2_hessian_b": str(ln2[1]),
            "head_frobenius": str(readout),
        },
        "global_jacobian_upper": bounds["jacobian"],
        "global_hessian_upper": bounds["hessian"],
        "claim_scope": "Exact-real norm upper bound over every hidden state; numerical torch rounding excluded"
    }


if __name__ == "__main__":
    import json
    q = (
        (Fraction(7, 4), Fraction(1, 2)),
        (Fraction(3, 2), Fraction(1, 3))
    )
    s = composition_bounds(q, Fraction(2))
    print(json.dumps({
        "sample_L": str(s["jacobian"]), "sample_M": str(s["hessian"]),
        "sqrt_2_upper": str(sqrt_upper(Fraction(2), 24)),
        "example_interval": list(map(str, directional_interval(1, Fraction(1, 10), 3))),
        "active_dominates": dominance_certificate(3, 1, Fraction(1, 10), Fraction(2)),
    }, indent=2))
