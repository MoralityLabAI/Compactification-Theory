"""COE-02: exact finite binary observer design with an ell_1 Lipschitz budget.

Each binary coordinate induces a cut of the latent sample. Its Hamming distance
is the sum of the cut semimetrics. A breadth-first search over all feasible
integer pair-distance vectors finds a minimum-width cut multiset, or proves
infeasibility for a finite metric. This is deliberately an exact small-n oracle.
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import ceil
from typing import Iterable, Sequence


Pair = tuple[int, int]


def metric_matrix(distance: Sequence[Sequence[object]]) -> tuple[tuple[Fraction, ...], ...]:
    """Check and return an exact rational finite metric."""
    n = len(distance)
    if n < 1:
        raise ValueError("at least one state is required")
    d = tuple(tuple(Fraction(v) for v in row) for row in distance)
    if any(len(row) != n for row in d):
        raise ValueError("metric must be square")
    for i in range(n):
        for j in range(n):
            if d[i][j] < 0 or d[i][j] != d[j][i]:
                raise ValueError("metric must be symmetric and nonnegative")
            if (i == j) != (d[i][j] == 0):
                raise ValueError("metric must separate distinct states")
            if any(d[i][j] > d[i][k] + d[k][j] for k in range(n)):
                raise ValueError("triangle inequality violated")
    return d


def graph_edges(n: int, edges: Iterable[Pair]) -> tuple[Pair, ...]:
    result = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("evaluator edges must join distinct valid vertices")
        result.add((min(u, v), max(u, v)))
    return tuple(sorted(result))


def crossing(mask: int, u: int, v: int) -> int:
    return ((mask >> u) ^ (mask >> v)) & 1


def canonical_cuts(n: int) -> tuple[int, ...]:
    """One orientation of each nontrivial cut, anchored at vertex zero."""
    if n < 1:
        raise ValueError("n must be positive")
    return tuple(mask for mask in range(2, 1 << n, 2))


@dataclass(frozen=True)
class CutObserver:
    words: tuple[int, ...]
    cuts: tuple[int, ...]
    loads: tuple[int, ...]
    capacities: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.cuts)


def words_from_cuts(n: int, cuts: Sequence[int]) -> tuple[int, ...]:
    return tuple(sum(crossing(cut, 0, u) << j for j, cut in enumerate(cuts))
                 for u in range(n))


def verify_observer(distance: Sequence[Sequence[object]], edges: Iterable[Pair],
                    L: object, margin: int, observer: CutObserver) -> bool:
    d = metric_matrix(distance)
    ee = graph_edges(len(d), edges)
    lipschitz = Fraction(L)
    if lipschitz < 0 or margin < 1 or len(observer.words) != len(d):
        return False
    if any(cut not in canonical_cuts(len(d)) for cut in observer.cuts):
        return False
    if observer.words != words_from_cuts(len(d), observer.cuts):
        return False
    pairs = tuple(combinations(range(len(d)), 2))
    distances = tuple((observer.words[u] ^ observer.words[v]).bit_count()
                      for u, v in pairs)
    caps = tuple(int(lipschitz * d[u][v]) for u, v in pairs)
    return (observer.loads == distances and observer.capacities == caps
            and all(value <= cap for value, cap in zip(distances, caps))
            and all((observer.words[u] ^ observer.words[v]).bit_count() >= margin
                    for u, v in ee))


def minimum_l1_observer(distance: Sequence[Sequence[object]],
                        edges: Iterable[Pair], L: object, margin: int = 1,
                        max_states: int = 250_000) -> CutObserver | None:
    """Exact minimum-width observer, or None when infeasible.

    BFS states record the total cut distance for *every* pair. Every nonconstant
    coordinate crosses at least one pair, so capacities bound the finite search
    depth by sum(capacities). We deduplicate equal distance vectors, retaining a
    witness path. max_states is a resource guard, never an infeasibility proof.
    """
    d = metric_matrix(distance)
    n = len(d)
    ee = graph_edges(n, edges)
    lipschitz = Fraction(L)
    if lipschitz < 0 or margin < 1 or max_states < 1:
        raise ValueError("nonnegative L, positive margin and state cap required")
    pairs = tuple(combinations(range(n), 2))
    caps = tuple(int(lipschitz * d[u][v]) for u, v in pairs)
    target = {pair: i for i, pair in enumerate(pairs)}
    target_indices = tuple(target[e] for e in ee)
    initial = (0,) * len(pairs)
    if not target_indices:
        return CutObserver((0,) * n, (), initial, caps)
    if any(caps[i] < margin for i in target_indices):
        return None
    patterns = tuple((cut, tuple(crossing(cut, u, v) for u, v in pairs))
                     for cut in canonical_cuts(n))
    # State paths represent the ordered list of cut masks; equivalent load
    # vectors are merged because every future move depends only on loads.
    queue = deque([initial])
    path = {initial: ()}
    while queue:
        current = queue.popleft()
        if all(current[i] >= margin for i in target_indices):
            cuts = path[current]
            result = CutObserver(words_from_cuts(n, cuts), cuts, current, caps)
            assert verify_observer(d, ee, lipschitz, margin, result)
            return result
        for mask, incidence in patterns:
            new = tuple(current[i] + incidence[i] for i in range(len(pairs)))
            if new in path or any(a > b for a, b in zip(new, caps)):
                continue
            if len(path) >= max_states:
                raise RuntimeError("exact finite search exceeded max_states; result unknown")
            path[new] = path[current] + (mask,)
            queue.append(new)
    return None


def equilateral(n: int, side: object = 1) -> tuple[tuple[Fraction, ...], ...]:
    if n < 1 or Fraction(side) <= 0:
        raise ValueError("nonempty equilateral metric with positive side required")
    return tuple(tuple(Fraction(0) if i == j else Fraction(side)
                       for j in range(n)) for i in range(n))


def dual_width_lower_bound(distance: Sequence[Sequence[object]],
                           edges: Iterable[Pair], L: object, margin: int,
                           alpha: dict[Pair, object],
                           beta: dict[Pair, object]) -> int:
    """Check a rational cut dual and return its rigorous integer width bound.

    For every canonical cut S, require
       sum_e alpha[e] delta_S(e) <= 1 + sum_p beta[p] delta_S(p).
    Then any valid observer has width at least
       ceil(margin*sum_e alpha[e] - sum_p beta[p]*floor(L*d[p])).
    """
    d = metric_matrix(distance)
    n = len(d)
    ee = graph_edges(n, edges)
    pairs = tuple(combinations(range(n), 2))
    L = Fraction(L)
    if L < 0 or margin < 1:
        raise ValueError("invalid L or margin")
    if any(p not in ee for p in alpha) or any(p not in pairs for p in beta):
        raise ValueError("unknown dual edge")
    a = {e: Fraction(alpha.get(e, 0)) for e in ee}
    b = {p: Fraction(beta.get(p, 0)) for p in pairs}
    if any(v < 0 for v in a.values()) or any(v < 0 for v in b.values()):
        raise ValueError("dual weights must be nonnegative")
    for cut in canonical_cuts(n):
        lhs = sum(a[e] * crossing(cut, *e) for e in ee)
        rhs = 1 + sum(b[p] * crossing(cut, *p) for p in pairs)
        if lhs > rhs:
            raise ValueError("cut dual inequality violated")
    lower = margin * sum(a.values()) - sum(b[p] * int(L * d[p[0]][p[1]])
                                               for p in pairs)
    return max(0, ceil(lower))



def balanced_fractional_certificate(n: int) -> tuple[tuple[int, ...], Fraction, Fraction]:
    """Exact symmetric fractional cut-cover witness for equilateral K_n.

    The cut-relaxation optimum is binom(n,2) / floor(n*n/4).
    All balanced cuts receive one rational weight, and every pair
    receives fractional separation exactly one. This is not a binary
    observer when the cut weight is nonintegral.
    """
    if n < 2:
        raise ValueError("at least two vertices are required")
    cuts = tuple(c for c in canonical_cuts(n)
                 if c.bit_count() in {n // 2, (n + 1) // 2})
    pairs = tuple(combinations(range(n), 2))
    incidences = [sum(crossing(c, u, v) for c in cuts) for u, v in pairs]
    assert len(set(incidences)) == 1 and incidences[0] > 0
    weight = Fraction(1, incidences[0])
    value = weight * len(cuts)
    assert all(weight * c == 1 for c in incidences)
    assert value == Fraction(len(pairs), (n * n) // 4)
    return cuts, weight, value


def run_reference() -> None:
    from itertools import combinations
    for label, n, edges, L in (
        ("K5/equilateral/L2", 5, list(combinations(range(5), 2)), 2),
        ("K5/equilateral/L3", 5, list(combinations(range(5), 2)), 3),
        ("C5/equilateral/L1", 5, [(i, (i + 1) % 5) for i in range(5)], 1),
        ("C5/equilateral/L2", 5, [(i, (i + 1) % 5) for i in range(5)], 2),
    ):
        result = minimum_l1_observer(equilateral(n), edges, L)
        print(f"{label}: {'infeasible' if result is None else f'k={result.width}, words={result.words}'}")


if __name__ == '__main__':
    run_reference()
