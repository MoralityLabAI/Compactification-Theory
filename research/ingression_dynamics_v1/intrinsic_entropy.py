"""QO-05: constructive metric-entropy bounds for latent low-dimensional hit rates.

All values are rational. A global L∞-L∞ Lipschitz certificate L for a
latent rate map phi:[0,1]^m -> [0,1]^d implies a rate-space cover with
at most ceil(L/(2*tau))^m non-goal macrostates, for tau>0.
The certificate and phi's description cost are **outside** this label-bit bound.
"""

from fractions import Fraction as Q
from itertools import combinations


def _fraction_ceiling(q):
    q = Q(q)
    return -(-q.numerator // q.denominator)


def grid_axis_count(lipschitz, tolerance):
    L, tau = Q(lipschitz), Q(tolerance)
    if L < 0 or tau <= 0:
        raise ValueError("nonnegative global Lipschitz constant and positive tolerance required")
    return max(1, _fraction_ceiling(L / (2*tau)))


def latent_grid_partition(latents, lipschitz, tolerance):
    """Construct an explicit quotient via regular cells in latent [0,1]^m."""
    latents = tuple(tuple(Q(v) for v in u) for u in latents)
    if not latents or not latents[0] or any(len(u) != len(latents[0]) for u in latents):
        raise ValueError("nonempty latent coordinates of equal dimension required")
    if any(not 0 <= x <= 1 for u in latents for x in u):
        raise ValueError("latent parameters must lie in the unit cube")
    cells = grid_axis_count(lipschitz, tolerance)
    signatures = [
        tuple(min(cells-1, (x*cells).numerator // (x*cells).denominator) for x in u)
        for u in latents
    ]
    indices = {}
    classes = tuple(indices.setdefault(v, len(indices)) for v in signatures)
    return classes, cells, tuple(signatures)


def bit_capacity_upper(latent_dim, lipschitz, tolerance):
    """Fixed-length macrostate bits incl. separately observable absorbing goal."""
    if type(latent_dim) is not int or latent_dim < 1:
        raise ValueError("positive intrinsic dimension required")
    m = grid_axis_count(lipschitz, tolerance)
    transient_classes = m**latent_dim
    return transient_classes.bit_length()


def rate_radius(points, labels):
    """Exact uniform hit/hold TV defect for the supplied transient partition."""
    pts = tuple(tuple(Q(x) for x in p) for p in points)
    if not pts or len(pts) != len(labels) or not pts[0] or any(
            len(p) != len(pts[0]) for p in pts):
        raise ValueError("rate vectors and labels must have matching lengths")
    classes = {}
    for p, label in zip(pts, labels):
        if any(not 0 <= x <= 1 for x in p):
            raise ValueError("rates must be in [0,1]")
        classes.setdefault(label, []).append(p)
    return max(((max(p[j] for p in block) - min(p[j] for p in block)) / 2
                for block in classes.values() for j in range(len(pts[0]))), default=Q(0))


def linear_rates(latents, coefficient_rows, intercepts=None):
    """Evaluate an affine map with exact arithmetic; certify L∞ Lipschitz by row L1."""
    latents = tuple(tuple(Q(x) for x in u) for u in latents)
    rows = tuple(tuple(Q(c) for c in row) for row in coefficient_rows)
    if not rows or not latents or not latents[0]:
        raise ValueError("nonempty latent points and rate outputs required")
    dim = len(latents[0])
    if any(len(row) != dim for row in rows) or any(len(u) != dim for u in latents):
        raise ValueError("coefficient/latent dimension mismatch")
    if intercepts is None:
        intercepts = (Q(0),) * len(rows)
    intercepts = tuple(Q(b) for b in intercepts)
    if len(intercepts) != len(rows):
        raise ValueError("intercept dimension mismatch")
    outputs = tuple(tuple(intercepts[a] +
                          sum((rows[a][j]*u[j] for j in range(dim)), Q(0))
                          for a in range(len(rows))) for u in latents)
    if any(not 0 <= v <= 1 for p in outputs for v in p):
        raise ValueError("sampled affine rates must be in [0,1]")
    L = max(sum((abs(c) for c in row), Q(0)) for row in rows)
    return outputs, L


def finite_lipschitz_check(latents, outputs, lipschitz):
    """Finite-point necessary check; global Lipschitz requires an independent proof."""
    latents = tuple(tuple(Q(x) for x in u) for u in latents)
    outputs = tuple(tuple(Q(x) for x in p) for p in outputs)
    L = Q(lipschitz)
    if len(latents) != len(outputs) or L < 0:
        raise ValueError("invalid sample pairing or Lipschitz constant")
    return all(
        max((abs(x-y) for x,y in zip(outputs[i], outputs[j])),default=Q(0))
        <= L * max((abs(x-y) for x,y in zip(latents[i], latents[j])),default=Q(0))
        for i,j in combinations(range(len(latents)),2)
    )


def exact_packing_lower_bound(rate_points, tolerance):
    """Max pairwise >2tau separated subset, exponential small-n oracle."""
    pts = tuple(tuple(Q(x) for x in p) for p in rate_points)
    if not pts or not pts[0] or any(len(p)!=len(pts[0]) for p in pts):
        raise ValueError("nonempty equal-dimensional rate points required")
    tau = Q(tolerance)
    if tau < 0:
        raise ValueError("nonnegative error tolerance required")
    if len(pts) > 16:
        raise ValueError("exhaustive packing oracle limited to 16 points")
    n = len(pts)
    for size in range(n, 0, -1):
        for chosen in combinations(range(n), size):
            if all(any(abs(x-y) > 2*tau for x,y in zip(pts[i],pts[j]))
                   for i,j in combinations(chosen,2)):
                return chosen
    raise AssertionError("nonempty set has a singleton packing")


def packing_bit_lower_bound(rate_points, tolerance):
    """Goal label consumes one additional class, as in QO-04."""
    required = len(exact_packing_lower_bound(rate_points, tolerance))
    return required.bit_length()


def metric_entropy_certificate(latents, coefficient_rows, tolerance, intercepts=None):
    """Exact finite sample demonstration plus analytic global affine L bound."""
    rate_points, L = linear_rates(latents, coefficient_rows, intercepts)
    labels, axis_cells, grid_signatures = latent_grid_partition(latents, L, tolerance)
    actual = rate_radius(rate_points, labels)
    upper = bit_capacity_upper(len(tuple(latents)[0]), L, tolerance)
    if actual > Q(tolerance) or len(set(labels)).bit_length() > upper:
        raise AssertionError("Lipschitz/cover certificate violated")
    return {
        "intrinsic_dim": len(tuple(latents)[0]),
        "actions": len(coefficient_rows),
        "global_lipschitz_L": str(L),
        "tolerance": str(Q(tolerance)),
        "grid_cells_per_axis": axis_cells,
        "used_transient_classes": len(set(labels)),
        "macro_label_bits_upper": upper,
        "observed_one_step_TV_defect": str(actual),
        "goal_label": len(set(labels)),
        "labels": labels,
        "grid_signatures": grid_signatures,
    }
