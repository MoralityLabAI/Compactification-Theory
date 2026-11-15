"""COE-01: finite exact chromatic observers with Hamming-noise certificates.

Standard-library reference. A vertex represents a unit vector e_v in R^n.
An edge marks a distinction an evaluator must preserve. Codewords are
binary probe outputs, with Hamming margin controlling adversarial flips.
"""

from itertools import combinations
from math import dist
from typing import Iterable


def check_graph(n: int, edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if n < 1:
        raise ValueError("require at least one latent state")
    clean = set()
    for a, b in edges:
        if not 0 <= a < n or not 0 <= b < n or a == b:
            raise ValueError("undirected simple graph with vertices 0..n-1 required")
        clean.add(tuple(sorted((a, b))))
    return tuple(sorted(clean))


def _adjacency(n: int, edges: tuple[tuple[int, int], ...]) -> list[set[int]]:
    a = [set() for _ in range(n)]
    for u, v in edges:
        a[u].add(v)
        a[v].add(u)
    return a


def optimal_coloring(n: int, edges: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    """Exact chromatic coloring via ordered branch and bound, small n only."""
    ee = check_graph(n, edges)
    adj = _adjacency(n, ee)
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    partial = [-1] * n
    best = list(range(n))
    best_count = n

    def walk(i: int, used: int) -> None:
        nonlocal best, best_count
        if i == n:
            if used < best_count:
                best, best_count = partial.copy(), used
            return
        v = order[i]
        blocked = {partial[u] for u in adj[v] if partial[u] >= 0}
        for color in range(min(used + 1, best_count)):
            if color in blocked:
                continue
            partial[v] = color
            walk(i + 1, max(used, color + 1))
        partial[v] = -1

    walk(0, 0)
    return tuple(best)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _search_margin(n: int, edges: tuple[tuple[int, int], ...], k: int,
                   margin: int) -> tuple[int, ...] | None:
    adj = _adjacency(n, edges)
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    assignment = [-1] * n
    candidates = range(1 << k)

    def walk(i: int) -> bool:
        if i == n:
            return True
        v = order[i]
        # XOR with the first word is a global isometry of Hamming space.
        words = (0,) if i == 0 else candidates
        for word in words:
            if all(assignment[u] < 0 or hamming(word, assignment[u]) >= margin
                   for u in adj[v]):
                assignment[v] = word
                if walk(i + 1):
                    return True
        assignment[v] = -1
        return False

    return tuple(assignment) if walk(0) else None


def minimum_observer(n: int, edges: Iterable[tuple[int, int]],
                     margin: int = 1) -> tuple[int, tuple[int, ...]]:
    """Return exact minimum binary width and an edge-margin witness.

    This exponential search is intended for n<=8, margin<=3. The ordinary
    coloring construction gives a certified finite upper bound.
    """
    ee = check_graph(n, edges)
    if margin < 1:
        raise ValueError("margin must be positive")
    coloring = optimal_coloring(n, ee)
    num_colors = max(coloring) + 1
    base_k = (num_colors - 1).bit_length()
    if margin == 1:
        return base_k, coloring
    for k in range(base_k, margin * base_k + 1):
        witness = _search_margin(n, ee, k, margin)
        if witness is not None:
            return k, witness
    raise AssertionError("repeated color bits must yield a witness")


def validate_code(n: int, edges: Iterable[tuple[int, int]],
                  k: int, words: tuple[int, ...], margin: int = 1) -> bool:
    ee = check_graph(n, edges)
    return (len(words) == n and k >= 0 and margin >= 1
            and all(0 <= w < 1 << k for w in words)
            and all(hamming(words[u], words[v]) >= margin for u, v in ee))


def hamming_ball_volume(k: int, radius: int) -> int:
    from math import comb
    return sum(comb(k, j) for j in range(min(k, radius) + 1))


def clique_bound_holds(clique_size: int, k: int, flips: int) -> bool:
    """Sphere-packing necessary condition for complete-graph t-flip codes."""
    return clique_size * hamming_ball_volume(k, flips) <= (1 << k)


def vertex_point(vertex: int, n: int) -> tuple[float, ...]:
    if vertex < 0 or vertex >= n:
        raise ValueError("invalid vertex")
    return tuple(1.0 if i == vertex else 0.0 for i in range(n))


def extend_probe(x: tuple[float, ...], n: int, k: int,
                 words: tuple[int, ...]) -> tuple[float, ...]:
    """Explicit coordinatewise McShane extension into [0,1]^k.

    On {0,e_0,...,e_(n-1)} the origin gets a half-vector and each
    spoke its bitword. The finite map has ell_infinity Lipschitz <= 1.
    Clipping an inf-of-cones preserves the 1-Lipschitz bound.
    """
    if len(x) != n or len(words) != n:
        raise ValueError("dimension mismatch")
    radius = sum(v * v for v in x) ** 0.5
    leaf_radii = tuple(dist(x, vertex_point(v, n)) for v in range(n))
    output = []
    for j in range(k):
        cones = [0.5 + radius]
        cones.extend(float((words[v] >> j) & 1) + leaf_radii[v]
                     for v in range(n))
        output.append(min(1.0, max(0.0, min(cones))))
    return tuple(output)


def complete_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(n), 2))


if __name__ == "__main__":
    for label, n, edges in (
        ("full-six", 6, complete_edges(6)),
        ("path-eight", 8, tuple((i, i + 1) for i in range(7))),
        ("cycle-five", 5, tuple((i, (i + 1) % 5) for i in range(5))),
    ):
        for margin in (1, 3):
            k, code = minimum_observer(n, edges, margin)
            assert validate_code(n, edges, k, code, margin)
            print(f"{label}: margin={margin}, k={k}, words={code}")
