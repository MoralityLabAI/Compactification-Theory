"""Source-only recurrence error bounds for Inkling-style looped computation.

This is an analytical instrument, not a claim about measured model Jacobians.
All Lipschitz constants and one-step numerical error bounds must be supplied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def error_envelope(
    local_lipschitz: Sequence[float],
    additive_error: Sequence[float],
    *,
    initial_error: float = 0.0,
) -> tuple[float, ...]:
    """Upper bound e[t+1] <= L[t] e[t] + eps[t] for a shared-state recurrence."""
    if len(local_lipschitz) != len(additive_error):
        raise ValueError("Lipschitz and error sequences require the same length")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x < 0
        for x in (initial_error, *local_lipschitz, *additive_error)
    ):
        raise ValueError("bounds must be nonnegative finite numbers")
    errors = [float(initial_error)]
    for local_gain, error in zip(local_lipschitz, additive_error, strict=True):
        upper = local_gain * errors[-1] + error
        if not math.isfinite(upper):
            raise ValueError("bound overflow")
        errors.append(upper)
    return tuple(errors)


def fixed_gate_gain(
    gain_coordinates: Sequence[float],
    message_lipschitz: float,
) -> float:
    """Infinity-norm bound for diag(gamma) M when gamma is held fixed.

    Input-dependent softmax readers and gates require a bound on their full
    Jacobian, including normalization, gating and attention effects.
    """
    if not gain_coordinates:
        raise ValueError("at least one gain coordinate is required")
    values = (*gain_coordinates, message_lipschitz)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float))
        or not math.isfinite(v)
        for v in values
    ) or message_lipschitz < 0:
        raise ValueError("finite gate coefficients and a nonnegative Lipschitz bound required")
    return max(abs(v) for v in gain_coordinates) * message_lipschitz


def applied_layer_proxy(
    *,
    fixed_layers: int,
    looped_layers: int,
    loop_passes: int,
    total_generated_tokens: int,
) -> int:
    """Analytical token-layer applications, not measured FLOPs, bytes or joules."""
    args = (fixed_layers, looped_layers, loop_passes, total_generated_tokens)
    if any(type(v) is not int or v < 0 for v in args):
        raise ValueError("integer nonnegative layer/token counts required")
    if loop_passes < 1 or looped_layers < 1 or total_generated_tokens < 1:
        raise ValueError("positive loop passes, middle layers and token count required")
    return (fixed_layers + looped_layers * loop_passes) * total_generated_tokens


def first_certified_depth(
    error_bounds: Sequence[float],
    tolerance: float,
) -> int | None:
    """First finite depth satisfying the supplied STATE error threshold only."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a real number")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative finite")
    if any(not math.isfinite(v) or v < 0 for v in error_bounds):
        raise ValueError("invalid error bounds")
    return next((i for i, value in enumerate(error_bounds) if value <= tolerance), None)
