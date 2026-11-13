"""Independent pairwise winding checks for *pure* observed activation braids.

These integers belong to a specified 2D measurement chart, not to the neural
representation independently of that chart. The analytic claims apply to a
continuous disk map; this module computes integers for polygonal sample paths.
"""
from __future__ import annotations

import math
from typing import Sequence

from knot import Extraction, InvalidBraidTrace


Point = tuple[float, float]


def _point(p: Sequence[float]) -> Point:
    if len(p) != 2 or not all(math.isfinite(float(x)) for x in p):
        raise InvalidBraidTrace("Non-finite or malformed projected point")
    return float(p[0]), float(p[1])


def _segment_clearance(a: Point, b: Point) -> float:
    """Exact Euclidean minimum separation of a straight segment from the origin."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    scale = vx * vx + vy * vy
    if scale == 0:
        return math.hypot(*a)
    alpha = max(0., min(1., -(a[0] * vx + a[1] * vy) / scale))
    return math.hypot(a[0] + alpha * vx, a[1] + alpha * vy)


def pairwise_windings(frames: Sequence[Sequence[Sequence[float]]], *,
                      closure_tolerance: float = 1e-6,
                      clearance: float = 1e-7,
                      integer_tolerance: float = 1e-5) -> dict[tuple[int, int], int]:
    """Winding of each labeled difference z_i-z_j along a closed PL cycle.

    All *individual* strand endpoints must close. Sign uses (x,y) orientation;
    SCM-01's Artin crossing convention yields braid linking = -winding.
    Rejects any segment of a relative path passing near the origin.
    """
    if len(frames) < 2 or len(frames[0]) < 2:
        raise InvalidBraidTrace("At least two frames and two strands required")
    n = len(frames[0])
    points: list[list[Point]] = []
    for frame in frames:
        if len(frame) != n:
            raise InvalidBraidTrace("Strand count changes")
        points.append([_point(p) for p in frame])
    for i in range(n):
        if math.dist(points[0][i], points[-1][i]) > closure_tolerance:
            raise InvalidBraidTrace("Named strand endpoints do not close: not a pure braid")
    result: dict[tuple[int,int],int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            relative = [(f[i][0] - f[j][0], f[i][1] - f[j][1]) for f in points]
            angle = 0.0
            for a, b in zip(relative, relative[1:]):
                if _segment_clearance(a, b) <= clearance:
                    raise InvalidBraidTrace("Projected pair collision along a PL segment")
                det = a[0] * b[1] - a[1] * b[0]
                dot = a[0] * b[0] + a[1] * b[1]
                angle += math.atan2(det, dot)
            turns = angle / (2 * math.pi)
            nearest = round(turns)
            if abs(turns - nearest) > integer_tolerance:
                raise InvalidBraidTrace("Non-integral winding from unclosed relative trajectory")
            result[i, j] = int(nearest)
    return result


def braid_pairwise_linking(extraction: Extraction) -> dict[tuple[int, int], int]:
    """Linking numbers for a pure braid, tracked by crossing strand identities.

    The event sign is the Artin generator sign as defined by SCM-01.
    The relation -winding = linking provides an independent consistency gate.
    """
    n = extraction.n_strands
    order = list(range(n))
    signed = {(i, j): 0 for i in range(n) for j in range(i+1, n)}
    for _time, i, j, generator in extraction.crossings:
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise InvalidBraidTrace("Malformed crossing identities")
        a, b = order.index(i), order.index(j)
        if abs(a - b) != 1 or min(a, b) + 1 != abs(generator):
            raise InvalidBraidTrace("Crossing inconsistent with adjacent strand ordering")
        key = (min(i, j), max(i, j))
        signed[key] += 1 if generator > 0 else -1
        order[a], order[b] = order[b], order[a]
    if order != list(range(n)):
        raise InvalidBraidTrace("Braid induces a nonidentity permutation")
    if any(value % 2 for value in signed.values()):
        raise InvalidBraidTrace("Pure braid pair signed crossings must be even")
    return {key: value // 2 for key, value in signed.items()}


def verify_pure_braid(frames: Sequence[Sequence[Sequence[float]]],
                      extraction: Extraction, *,
                      clearance: float = 1e-7) -> dict[tuple[int, int], int]:
    """Require two independent algorithms to give the same pairwise link numbers."""
    winding = pairwise_windings(frames, clearance=clearance)
    linking = braid_pairwise_linking(extraction)
    mismatches = {key: (val, linking[key]) for key, val in winding.items()
                  if -val != linking[key]}
    if mismatches:
        raise InvalidBraidTrace(f"Braid/winding consistency failure: {mismatches}")
    return winding


def verified_sampling_bound(frames: Sequence[Sequence[Sequence[float]]],
                            *, interval_lengths: Sequence[float],
                            pair_derivative_bounds: dict[tuple[int, int], float],
                            clearance: float = 1e-7) -> bool:
    """Sufficient condition for interpolation preserving a *continuous* braid.

    Requires EXTERNALLY VERIFIED upper bounds on |d(z_i-z_j)/dtheta|
    over each interval, supplied here as uniform bounds per pair. Observed
    finite differences or endpoint autodiff derivatives do not establish them.

    For every pair and interval, |difference(theta_k)| > L_ij * dtheta_k
    + clearance ensures the true pair path and its linear interpolation lie
    in the same origin-avoiding convex disk. Straight-line homotopy therefore
    preserves all pairwise noncollisions across the whole configuration.
    """
    if len(frames) < 2 or len(interval_lengths) != len(frames)-1:
        raise ValueError("Interval lengths must match frame gaps")
    n = len(frames[0])
    if n < 2 or any(len(frame) != n for frame in frames):
        raise ValueError("All frames need the same number of at least two strands")
    points = [[_point(p) for p in frame] for frame in frames]
    if any(math.dist(points[0][i], points[-1][i]) > clearance for i in range(n)):
        raise InvalidBraidTrace("Unclosed labeled cycle cannot be certified")
    all_pairs = {(i, j) for i in range(n) for j in range(i+1, n)}
    if set(pair_derivative_bounds) != all_pairs:
        raise ValueError("One verified derivative bound required per pair")
    for k, dt in enumerate(interval_lengths):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Intervals must be positive and finite")
        for i, j in all_pairs:
            bound = pair_derivative_bounds[i, j]
            if not math.isfinite(bound) or bound < 0:
                raise ValueError("Derivative bounds must be finite and nonnegative")
            dx = points[k][i][0] - points[k][j][0]
            dy = points[k][i][1] - points[k][j][1]
            if math.hypot(dx, dy) <= bound * dt + clearance:
                return False
    return True
