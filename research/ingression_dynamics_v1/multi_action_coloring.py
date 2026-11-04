"""QO-03: exact finite action-aware goal aggregation / graph coloring reduction.

For an absorbing-goal hit/hold system, a class B has one-step minimax
total-variation radius (1/2)*max_a (max_{s in B} p[a,s] -
min_{s in B} p[a,s]). This module supplies explicit graph encodings,
witnesses, and independent small-instance decision oracles.
"""
from fractions import Fraction
from itertools import product

HALF = Fraction(1, 2)
QUARTER = Fraction(1, 4)


def normalize_edges(n, edges):
    """Simple graph on vertices 0..n-1; reject loops, canonicalize edges."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("positive number of transient vertices required")
    canonical = set()
    for u, v in edges:
        if (not isinstance(u, int) or not isinstance(v, int) or
                not (0 <= u < n and 0 <= v < n) or u == v):
            raise ValueError("invalid edge")
        canonical.add(tuple(sorted((u, v))))
    return tuple(sorted(canonical))


def edge_hit_rates(n, edges):
    """One hit/hold action per graph edge; edge endpoints have rates 0 and 1."""
    edges = normalize_edges(n, edges)
    rates = {"idle": (Fraction(0),) * n}
    for u, v in edges:
        rates[f"edge_{u}_{v}"] = tuple(
            Fraction(0) if s == u else
            Fraction(1) if s == v else HALF
            for s in range(n))
    return rates


def maximum_rate_radius(rates, assignment):
    """Minimax TV radius with a separate absorbing goal; exact rationals."""
    n = len(assignment)
    if not n:
        raise ValueError("at least one transient state required")
    if not rates or any(len(row) != n for row in rates.values()):
        raise ValueError("each action needs a probability for each transient")
    if any(not (0 <= p <= 1) for row in rates.values() for p in row):
        raise ValueError("rates must be probabilities")
    result = Fraction(0)
    for row in rates.values():
        buckets = {}
        for s, z in enumerate(assignment):
            buckets.setdefault(z, []).append(row[s])
        radius = max(((max(v) - min(v)) / 2 for v in buckets.values()),
                     default=Fraction(0))
        result = max(result, radius)
    return result


def exact_three_colorings(n, edges):
    """Independent decision oracle using direct proper vertex colorings."""
    edges = normalize_edges(n, edges)
    for colors in product(range(3), repeat=n):
        if all(colors[u] != colors[v] for u, v in edges):
            yield colors


def best_radius_with_three_transient_classes(n, edges):
    """Exhaustive 3-label aggregation oracle, exponential in n."""
    rates = edge_hit_rates(n, edges)
    return min(maximum_rate_radius(rates, labels)
               for labels in product(range(3), repeat=n))


def witness_from_coloring(n, edges):
    """At most three transient classes plus a distinct goal label (2 bits)."""
    coloring = next(exact_three_colorings(n, edges), None)
    if coloring is None:
        return None
    assert maximum_rate_radius(edge_hit_rates(n, edges), coloring) <= QUARTER
    return coloring + (3,)


def actual_macro_kernels_from_rates(rates, coloring):
    """Construct exact midpoint kernels for each action and macro class."""
    colors = sorted(set(coloring))
    goal = max(colors) + 1
    macro = {}
    for action, row in rates.items():
        macro[action] = {}
        for z in colors:
            values = [p for i, p in enumerate(row) if coloring[i] == z]
            center = (min(values) + max(values)) / 2
            macro[action][z] = {z: 1 - center, goal: center}
        macro[action][goal] = {goal: Fraction(1)}
    return macro


def joint_action_obstruction():
    """A 2-D square: each single action needs 2 classes, both need 4."""
    return {
        "x": tuple(Fraction(p) for p in (0, 0, 1, 1)),
        "y": tuple(Fraction(p) for p in (0, 1, 0, 1)),
    }
