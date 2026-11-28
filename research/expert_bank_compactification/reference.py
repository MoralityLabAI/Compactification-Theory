"""EBC-01: exact finite support compactification (small deterministic fixtures)."""
from fractions import Fraction
from itertools import combinations


def mask(ids):
    out = 0
    for e in ids:
        if type(e) is not int or e < 0:
            raise ValueError("invalid expert ID")
        out |= 1 << e
    return out


def validate(demand, successors, cap):
    n = len(demand)
    if not 1 <= n <= 10 or len(successors) != n or type(cap) is not int or cap < 1:
        raise ValueError("expected 1..10 states and positive capacity")
    if any(type(d) is not int or d <= 0 or d.bit_count() > cap for d in demand):
        raise ValueError("infeasible demand")
    a = len(successors[0])
    if any(len(row) != a or any(type(i) is not int or not 0 <= i < n for i in row)
           for row in successors):
        raise ValueError("transition domain is not closed")


def quotient(demand, successors, cap):
    """Minimize class count subject to support cap and successor congruence.

    Exact exponential search; only for finite development fixtures. The full
    native state is still supplied to the inner router, which resolves choices
    within each class's allowed support. No native trajectory is simulated.
    """
    validate(demand, successors, cap)
    n = len(demand)
    best = None
    explored = 0

    def visit(labels, supports):
        nonlocal best, explored
        if best is not None and len(supports) >= len(best[1]):
            return
        i = len(labels)
        if i == n:
            explored += 1
            transition = {}
            for s, c in enumerate(labels):
                row = tuple(labels[t] for t in successors[s])
                if c in transition and transition[c] != row:
                    return
                transition[c] = row
            best = (tuple(labels), tuple(supports), tuple(transition[c] for c in range(len(supports))))
            return
        for c, old in enumerate(supports):
            new = old | demand[i]
            if new.bit_count() <= cap:
                visit(labels + [c], supports[:c] + [new] + supports[c+1:])
        visit(labels + [len(supports)], supports + [demand[i]])

    visit([], [])
    labels, supports, transition = best
    return {"labels": labels, "supports": supports, "successors": transition,
            "classes": len(supports), "bits": (len(supports)-1).bit_length(),
            "completed_candidates": explored}


def full_trace_classes(demand, successors):
    """Ordinary exact route-set observation quotient via partition refinement."""
    validate(demand, successors, max(d.bit_count() for d in demand))
    def compress(items):
        table = {}
        return tuple(table.setdefault(x, len(table)) for x in items)
    classes = compress(demand)
    while True:
        updated = compress((demand[s], tuple(classes[t] for t in row))
                           for s, row in enumerate(successors))
        if updated == classes:
            return classes
        classes = updated


def exact_static(demand, weights, universe, cap):
    """Maximize weighted *complete route* coverage; exhaustive small oracle."""
    if not 1 <= universe <= 20 or not 1 <= cap <= universe or len(demand) != len(weights):
        raise ValueError("invalid bounded static instance")
    if any(type(d) is not int or not 0 < d < 1 << universe for d in demand):
        raise ValueError("invalid demand")
    if any(w < 0 for w in weights):
        raise ValueError("negative weight")
    best = None
    for ids in combinations(range(universe), cap):
        m = mask(ids)
        covered = sum(w for d, w in zip(demand, weights) if d & m == d)
        if best is None or covered > best[0]:
            best = (covered, ids)
    return best


def normalized_tv(common, removed, added):
    """Exact TV of mixtures with unchanged scores on the common experts."""
    c, a, b = map(Fraction, (common, removed, added))
    if min(c, a, b) < 0 or min(c+a, c+b) <= 0:
        raise ValueError("invalid mixture mass")
    return 1 - c / max(c+a, c+b)


def fixture():
    demand = [1, 1, 2, 4]
    successors = [(2,), (3,), (2,), (3,)]
    return {"future_trace_classes": full_trace_classes(demand, successors),
            "capacity_frontier": {str(k): quotient(demand, successors, k) for k in (1, 2, 3)},
            "static_counterexample": exact_static([mask((0,1)), mask((0,2)), mask((3,4))], [4,3,5], 5, 3)}


if __name__ == "__main__":
    import json
    print(json.dumps(fixture(), indent=2))
