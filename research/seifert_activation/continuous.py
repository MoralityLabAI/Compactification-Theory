"""SCM-05: conservative, rational-geometry braid sampling/compression checks.

All geometric comparisons use Fraction arithmetic on the exact numeric inputs,
with outward square-root rounding at a registered fixed decimal precision.
The *continuous* conclusions are conditional on a trusted, independently
established pairwise Lipschitz bound, including its interval scope.
Finite differences, automatic differentiation at isolated points, and double
sampling are NOT proofs of these global upper bounds.

A positive certificate preserves the labeled loop's free homotopy class in
Conf_n(R^2), hence the closed braid/link and its oriented link invariants.
The based braid is preserved if the two loops have the same base configuration.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isfinite, isqrt
from typing import Mapping, Sequence

Pair = tuple[int, int]
Point = tuple[Fraction, Fraction]


def rat(value) -> Fraction:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a scalar')
    if isinstance(value, float) and not isfinite(value):
        raise ValueError('Nonfinite scalar')
    ans = Fraction(value)
    if ans.denominator == 0:
        raise ValueError('Invalid rational denominator')
    return ans


def _data(frames) -> list[list[Point]]:
    if len(frames) < 2 or len(frames[0]) < 2:
        raise ValueError('Need at least two samples and at least two strands')
    n = len(frames[0])
    out = []
    for row in frames:
        if len(row) != n:
            raise ValueError('Strand count changes between frames')
        if any(len(point) != 2 for point in row):
            raise ValueError('Every projected point must have two coordinates')
        out.append([(rat(point[0]), rat(point[1])) for point in row])
    return out


def _closed(data) -> bool:
    # Exact agreement is the conservative default. A deterministic neural run
    # on identical endpoint perturbations should satisfy this on CPU.
    return data[0] == data[-1]


def _relative(row, i: int, j: int) -> Point:
    return row[i][0] - row[j][0], row[i][1] - row[j][1]


def norm_squared(point: Point) -> Fraction:
    return point[0] ** 2 + point[1] ** 2


def min_segment_distance_sq(a: Point, b: Point) -> Fraction:
    """Squared clearance of a PL relative strand from the collision origin."""
    v = (b[0] - a[0], b[1] - a[1])
    v2 = norm_squared(v)
    if v2 == 0:
        return norm_squared(a)
    dot = a[0] * v[0] + a[1] * v[1]
    s = max(Fraction(0), min(Fraction(1), -dot / v2))
    return norm_squared((a[0] + s * v[0], a[1] + s * v[1]))


def _sqrt_bounds(value: Fraction, digits: int) -> tuple[Fraction, Fraction]:
    """Exact outward rational bounds for sqrt(value), decimal grid 10^-digits."""
    if value < 0 or not (0 <= digits <= 18):
        raise ValueError('Invalid square or precision')
    scale = 10 ** digits
    integer_part = isqrt((value.numerator * scale * scale) // value.denominator)
    lower = Fraction(integer_part, scale)
    upper = lower if integer_part * integer_part * value.denominator == value.numerator * scale * scale \
        else Fraction(integer_part + 1, scale)
    return lower, upper


def _bounds(value: Mapping, n: int, gaps: int) -> dict[Pair, list[Fraction]]:
    """Accept exact pair keys 'i,j'; each value uniform or a per-gap list."""
    if value is None:
        raise ValueError('Missing externally established Lipschitz upper bounds')
    if not isinstance(value, Mapping):
        raise ValueError('Lipschitz bounds must be a mapping')
    result: dict[Pair, list[Fraction]] = {}
    expected = [f'{i},{j}' for i, j in combinations(range(n), 2)]
    if set(value) != set(expected):
        raise ValueError(f'Bounds must contain exactly {expected}')
    for key in expected:
        i, j = map(int, key.split(','))
        raw = value[key]
        entries = raw if isinstance(raw, (tuple, list)) else [raw] * gaps
        if len(entries) != gaps:
            raise ValueError(f'Wrong number of per-interval bounds for {key}')
        vv = [rat(x) for x in entries]
        if any(x < 0 for x in vv):
            raise ValueError('Lipschitz bound must be nonnegative')
        result[i, j] = vv
    return result


def _gaps(lengths: Sequence, n_intervals: int) -> list[Fraction]:
    if len(lengths) != n_intervals:
        raise ValueError('Interval length count does not match samples')
    gaps = [rat(x) for x in lengths]
    if any(x <= 0 for x in gaps):
        raise ValueError('Angular intervals must have positive lengths')
    return gaps


def _summary_rows(a, gaps, original_bounds, reconstructed=None,
                  reconstructed_bounds=None, *, extra_margin=Fraction(0), digits=12):
    n = len(a[0])
    per_pair = {}
    for i, j in combinations(range(n), 2):
        tested = []
        for k, gap in enumerate(gaps):
            p0, p1 = _relative(a[k], i, j), _relative(a[k+1], i, j)
            m_sq = min_segment_distance_sq(p0, p1)
            m_lower = _sqrt_bounds(m_sq, digits)[0]
            source_error = original_bounds[i, j][k] * gap / 2
            reconstruct_displacement = Fraction(0)
            reconstruction_alias = Fraction(0)
            if reconstructed is not None:
                q0, q1 = _relative(reconstructed[k], i, j), _relative(reconstructed[k+1], i, j)
                errors_sq = [norm_squared((p0[0]-q0[0],p0[1]-q0[1])),
                             norm_squared((p1[0]-q1[0],p1[1]-q1[1]))]
                reconstruct_displacement = _sqrt_bounds(max(errors_sq), digits)[1]
                reconstruction_alias = reconstructed_bounds[i, j][k] * gap / 2
            source_margin = m_lower - source_error - extra_margin
            target_margin = m_lower - reconstruct_displacement - reconstruction_alias - extra_margin
            passed = source_margin > 0 and target_margin > 0
            tested.append({
                'interval': k,
                'certified_condition': bool(passed),
                'minimum_PL_clearance_lower': float(m_lower),
                'original_alias_upper': float(source_error),
                'sampled_reconstruction_error_upper': float(reconstruct_displacement),
                'reconstruction_alias_upper': float(reconstruction_alias),
                'source_margin_lower': float(source_margin),
                'reconstruction_margin_lower': float(target_margin),
            })
        per_pair[f'{i},{j}'] = {
            'passes': all(x['certified_condition'] for x in tested),
            'worst_source_margin_lower': min(x['source_margin_lower'] for x in tested),
            'worst_reconstruction_margin_lower': min(x['reconstruction_margin_lower'] for x in tested),
            'intervals': tested,
        }
    return per_pair



def allowable_pair_lipschitz_ceiling(frames, *, interval_lengths, extra_margin='0', digits=12) -> dict:
    """Maximum *sufficient* per-pair uniform L permitted by sampled clearance.

    Independent of any supplied derivative claim. This is a feasibility
    diagnostic, never evidence that the actual path satisfies that bound.
    Each interval requires L < 2*(m_lower - margin)/delta_theta.
    """
    a=_data(frames)
    if not _closed(a):
        return {'valid':False,'reason':'sampled_endpoints_not_exactly_equal'}
    gaps=_gaps(interval_lengths,len(a)-1)
    margin=rat(extra_margin)
    if margin<0:
        raise ValueError('Extra margin must be nonnegative')
    limits={}
    for i,j in combinations(range(len(a[0])),2):
        ceilings=[]
        for k,gap in enumerate(gaps):
            p0,p1=_relative(a[k],i,j),_relative(a[k+1],i,j)
            m=_sqrt_bounds(min_segment_distance_sq(p0,p1),digits)[0]
            ceilings.append(2*(m-margin)/gap)
        limits[f'{i},{j}']=float(min(ceilings))
    return {'valid':True,'scope':'diagnostic_only_no_derivative_proof',
            'max_sufficient_uniform_L_by_pair':limits,
            'any_segment_collisions':any(x<=0 for x in limits.values())}


def continuous_sampling_certificate(frames, *, interval_lengths,
                                    pair_lipschitz_bounds,
                                    extra_margin='0', digits=12) -> dict:
    """Conditional theorem: the unknown continuous braid equals the PL braid.

    For each pair q=z_i-z_j with verified interval Lipschitz bound L,
    ||q(theta) - PL(q)(theta)|| <= L * delta_theta / 2.
    If min_clearance(PL(q)) exceeds this plus margin on every interval,
    straight interpolation preserves pairwise noncollisions everywhere.
    Exact labeled endpoint closure is required and checked for sampled values.
    An independent source must prove exact closure and the true L bounds.
    """
    a = _data(frames)
    if not _closed(a):
        return {'passes': False, 'reason': 'sampled_endpoints_not_exactly_equal',
                'scope': 'uncertified', 'pairs': {}}
    gaps = _gaps(interval_lengths, len(a)-1)
    bound = _bounds(pair_lipschitz_bounds, len(a[0]), len(gaps))
    margin = rat(extra_margin)
    if margin < 0:
        raise ValueError('Extra safety margin must be nonnegative')
    pairs = _summary_rows(a, gaps, bound, extra_margin=margin, digits=digits)
    passes = all(x['passes'] for x in pairs.values())
    return {'passes': passes, 'scope': 'conditional_on_external_interval_lipschitz_proofs',
            'kind': 'continuous_vs_sampled_braid', 'rational_sqrt_digits': digits,
            'pairs': pairs, 'reason': 'sufficient_margin' if passes else 'insufficient_margin'}


def continuous_compression_certificate(original, reconstructed, *, interval_lengths,
                                       original_pair_lipschitz_bounds,
                                       reconstructed_pair_lipschitz_bounds,
                                       extra_margin='0', digits=12) -> dict:
    """Conditional theorem that two CONTINUOUS loops represent same braid conjugacy class.

    For every pair relative path, let m be the PL original minimum clearance,
    E the maximum sampled relative reconstruction error on the interval,
    B=L_orig * dtheta/2, C=L_recon * dtheta/2.
    If m > max(B,E+C)+margin, original and reconstructed continuous loops
    are joined via the original PL loop by two collision-free linear homotopies.
    Their oriented, ordered link invariants are identical, including Milnor.
    Requires independent proofs of the interval pair bounds and exact closure.
    """
    a, b = _data(original), _data(reconstructed)
    if len(a)!=len(b) or len(a[0])!=len(b[0]):
        raise ValueError('Original and reconstruction must use identical sample/strand dimensions')
    if not _closed(a) or not _closed(b):
        return {'passes': False, 'reason': 'sampled_endpoints_not_exactly_equal',
                'scope': 'uncertified', 'pairs': {}}
    gaps = _gaps(interval_lengths, len(a)-1)
    original_bounds = _bounds(original_pair_lipschitz_bounds, len(a[0]), len(gaps))
    recon_bounds = _bounds(reconstructed_pair_lipschitz_bounds, len(a[0]), len(gaps))
    margin = rat(extra_margin)
    if margin < 0:
        raise ValueError('Extra safety margin must be nonnegative')
    pairs = _summary_rows(a,gaps,original_bounds,reconstructed=b,
                          reconstructed_bounds=recon_bounds,extra_margin=margin,digits=digits)
    passes = all(x['passes'] for x in pairs.values())
    based = a[0] == b[0]
    return {'passes': passes, 'scope': 'conditional_on_external_interval_lipschitz_proofs',
            'kind': 'continuous_original_vs_compressed', 'based_braid_preserved':bool(passes and based),
            'closed_braid_and_link_preserved':bool(passes),
            'rational_sqrt_digits':digits,'pairs':pairs,
            'reason':'sufficient_margin' if passes else 'insufficient_margin'}
