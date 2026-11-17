"""SCM-12: a rank-one response-funnel sufficient null-braid certificate.

If every projected strand is z_i(theta)=a_i+b_i*f(theta)+e_i(theta),
where f:S^1->[lo,hi] is continuous and ||e_i|| <= eta_i uniformly,
then a positive pair clearance greater than eta_i+eta_j gives a homotopy
through collision-free configurations to a loop factoring through an interval.
The latter loop is contractible; thus the labeled braid is trivial.

All numerical premises are EXACT rational values. Supremum residual claims must
be separately proven from the source model before interpreting a neural trace.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Sequence

from continuous import rat, min_segment_distance_sq, _sqrt_bounds


def rank_one_null_certificate(
    base: Sequence[Sequence],
    directions: Sequence[Sequence],
    scalar_range: Sequence,
    errors: Sequence,
    *,
    margin="0",
    sqrt_digits: int = 12,
) -> dict:
    if len(base) < 2 or len(base) != len(directions) or len(base) != len(errors):
        raise ValueError('At least two strands and matching base, directions and errors required')
    if len(scalar_range) != 2:
        raise ValueError('Expected two scalar interval endpoints')
    lo, hi = (rat(x) for x in scalar_range)
    if lo > hi:
        raise ValueError('Reversed scalar range')
    extra = rat(margin)
    if extra < 0:
        raise ValueError('Margin must be nonnegative')
    a = []
    b = []
    for center, derivative in zip(base, directions):
        if len(center) != 2 or len(derivative) != 2:
            raise ValueError('Each strand base and direction must be planar')
        a.append(tuple(rat(x) for x in center))
        b.append(tuple(rat(x) for x in derivative))
    eta = [rat(x) for x in errors]
    if any(x < 0 for x in eta):
        raise ValueError('Uniform errors must be nonnegative')
    pairs = {}
    passes = True
    for i, j in combinations(range(len(a)), 2):
        def relative(t: Fraction):
            return (
                a[i][0] - a[j][0] + (b[i][0] - b[j][0]) * t,
                a[i][1] - a[j][1] + (b[i][1] - b[j][1]) * t,
            )
        distance_sq = min_segment_distance_sq(relative(lo), relative(hi))
        lower = _sqrt_bounds(distance_sq, sqrt_digits)[0]
        required = eta[i] + eta[j] + extra
        ok = lower > required
        passes = passes and ok
        pairs[f'{i},{j}'] = {
            'lower_clearance': str(lower),
            'max_relative_error_plus_margin': str(required),
            'pass': ok,
        }
    return {
        'schema': 'scm12/rank-one-null/v1',
        'result': 'conditional_null_braid' if passes else 'inconclusive',
        'valid_for': 'continuous_exact_real_chart_loops_under_proven_uniform_error_claims',
        'scalar_interval': [str(lo), str(hi)],
        'per_pair': pairs,
        'error_bounds_source_verified': False,
        'observed_neural_topology_certified': False,
    }
