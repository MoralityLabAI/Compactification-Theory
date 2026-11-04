"""QO-04: exact geometric quotient optimization for a FIXED action dimension.

Input: rational points in [0,1]^d representing per-action hit probabilities.
Each transient macrostate class must fit in an axis-aligned side-2*tau box.
Goal is a separate absorbing macrostate, charged one additional codeword.

The bounded-k solver enumerates at most n^d anchored candidate boxes and
does a k-depth search. Polynomial for separately fixed dimension d and
transient class budget k, with no claim of efficient variable-k complexity.
"""

from fractions import Fraction as Q
from itertools import combinations, product
from functools import lru_cache


def _as_points(points, require_unit=True):
    points = tuple(tuple(Q(v) for v in point) for point in points)
    if not points or not points[0] or any(len(p) != len(points[0]) for p in points):
        raise ValueError("nonempty equal-dimension rate vectors are required")
    if require_unit and any(v < 0 or v > 1 for p in points for v in p):
        raise ValueError("hit rates must be in [0,1]")
    return points


def candidates(points, side):
    """Canonical boxes (anchor, covered bit mask), with anchors from coordinate values.

    Every feasible block has a covering box whose lower corner is comprised
    of per-coordinate minima of that block's points; these coordinate values
    need not come from the SAME input point.
    """
    points = _as_points(points)
    side = Q(side)
    if side < 0:
        raise ValueError("side length must be nonnegative")
    choices = tuple(tuple(sorted({p[j] for p in points}))
                    for j in range(len(points[0])))
    by_mask = {}
    for corner in product(*choices):
        mask = sum(1 << i for i, p in enumerate(points)
                   if all(corner[j] <= p[j] <= corner[j] + side
                          for j in range(len(corner))))
        if mask and (mask not in by_mask or corner < by_mask[mask]):
            by_mask[mask] = corner
    return tuple((corner, mask) for mask, corner in
                 sorted(by_mask.items(), key=lambda item: (item[1], item[0])))


def _solve(points, side, budget=None):
    points = _as_points(points)
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("positive integer class budget is required")
    boxes = candidates(points, side)
    full = (1 << len(points)) - 1
    relevant = tuple(tuple(i for i, (_, mask) in enumerate(boxes)
                           if mask & (1 << s))
                     for s in range(len(points)))
    @lru_cache(None)
    def solve(uncovered, remaining):
        if not uncovered:
            return ()
        if remaining == 0:
            return None
        least = (uncovered & -uncovered).bit_length() - 1
        best = None
        for i in relevant[least]:
            next_missing = uncovered & ~boxes[i][1]
            suffix = solve(next_missing, remaining - 1)
            if suffix is None:
                continue
            candidate = (i,) + suffix
            if best is None or (len(candidate), candidate) < (len(best), best):
                best = candidate
        return best
    max_classes = min(len(points), budget if budget is not None else len(points))
    result = solve(full, max_classes)
    return None if result is None else tuple(boxes[i] for i in result)


def cover_with_k(points, side, k):
    """Exact bounded-k witness using at most (n^d)^k branches."""
    return _solve(points, side, k)


def minimum_cover(points, side):
    """Exact minimum number of boxes, using 2^n-uncovered-state memoization."""
    return _solve(points, side)


def class_assignment(points, witness):
    """Assign each transient state to the first covering candidate, goal separately."""
    points = _as_points(points)
    if not witness:
        raise ValueError("at least one box is required")
    labels = []
    for i in range(len(points)):
        choices = [j for j, (_, mask) in enumerate(witness) if mask & (1 << i)]
        if not choices:
            raise ValueError("box witness leaves a state uncovered")
        labels.append(choices[0])
    return tuple(labels) + (len(witness),)


def optimum_for_k(points, k):
    """Global optimum radius for fixed d,k; finite pairwise-distance breakpoints."""
    points = _as_points(points)
    if type(k) is not int or k < 1:
        raise ValueError("positive number of transient classes required")
    n = len(points)
    levels = {Q(0)}
    for i, j in combinations(range(n), 2):
        levels.add(max(abs(a-b) for a,b in zip(points[i], points[j])) / 2)
    for error in sorted(levels):
        boxes = cover_with_k(points, 2*error, k)
        if boxes is not None:
            return error, boxes, class_assignment(points, boxes)
    raise AssertionError("singleton boxes always provide a solution for k>=n")


def one_dimensional_greedy(points, side):
    """O(n log n) after sorting for d=1; valid for variable class budget."""
    points = _as_points(points)
    if len(points[0]) != 1:
        raise ValueError("single-action greedy solver requires d=1")
    side = Q(side)
    if side < 0:
        raise ValueError("negative side")
    order = sorted(range(len(points)), key=lambda i: (points[i][0], i))
    boxes = []
    while order:
        anchor = points[order[0]][0]
        covered = tuple(i for i in order if points[i][0] <= anchor+side)
        mask = sum(1 << i for i in covered)
        boxes.append(((anchor,), mask))
        order = [i for i in order if not (mask & (1 << i))]
    return tuple(boxes)


def minimum_label_bits(points, side):
    """Exactly minimum fixed-length macro label bits; absorbing goal costs +1."""
    needed = len(minimum_cover(points, side))
    return needed.bit_length()


def normalize_cover_instance(points, side):
    """Polynomial-size rational affine scaling to probabilities in [0,1]^d."""
    pts = _as_points(points, require_unit=False)
    side = Q(side)
    if side <= 0:
        raise ValueError("positive geometric square side required")
    low = tuple(min(p[j] for p in pts) for j in range(len(pts[0])))
    high = tuple(max(p[j] for p in pts) for j in range(len(pts[0])))
    scale = max((Q(1), side, *(high[j] - low[j] for j in range(len(low)))))
    normalized = tuple(tuple((p[j] - low[j])/scale
                             for j in range(len(low))) for p in pts)
    return normalized, side/scale


def pad_budget_to_binary(points, side, requested_classes):
    """Add isolated points so 2^b-1 allowed transient labels means exactly k original.

    Input may have arbitrary rational 2D coordinates. Added states are forced
    singleton blocks by separation strictly greater than the box side.
    """
    pts = _as_points(points, require_unit=False)
    if len(pts[0]) != 2:
        raise ValueError("binary-budget padding is two-dimensional")
    side = Q(side)
    k = requested_classes
    if type(k) is not int or not 1 <= k <= len(pts) or side <= 0:
        raise ValueError("require side>0 and 1<=k<=number of points")
    bits = k.bit_length()            # ceil(log2(k+1))
    slots = (1 << bits) - 1
    extras = slots - k
    highest_x = max(p[0] for p in pts)
    bottom_y = min(p[1] for p in pts)
    jump = side + 1
    padded = pts + tuple((highest_x + jump*(i+1), bottom_y)
                         for i in range(extras))
    normalized, normalized_side = normalize_cover_instance(padded, side)
    return normalized, normalized_side, bits, extras


def points_as_action_rates(points):
    """One hit/hold action per coordinate; goal is a distinct absorbing state."""
    pts = _as_points(points)
    return {f"action_{j}": tuple(p[j] for p in pts)
            for j in range(len(pts[0]))}
