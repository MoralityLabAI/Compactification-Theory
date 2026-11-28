"""Exact finite-state tissue-territory proxy inspired by Levinverse talks.

Pure Python, no biology simulation and no third-party dependencies.
Each vertex is a binary spin, with background clamped to -1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Iterable

Q = Fraction


@dataclass(frozen=True)
class Tissue:
    n: int
    edges: tuple[tuple[int, int], ...]
    ground: tuple[int, ...]
    program: frozenset[int]
    coupling: Q = Q(1)
    anchor: Q = Q(1)

    def __post_init__(self) -> None:
        if self.n < 1 or len(self.ground) != self.n:
            raise ValueError("n and ground lengths must agree")
        if not self.program or not self.program.issubset(range(self.n)):
            raise ValueError("program must be a nonempty vertex subset")
        if self.coupling <= 0 or self.anchor < 0:
            raise ValueError("coupling must be positive; anchor nonnegative")
        if any(b < 0 for b in self.ground):
            raise ValueError("ground edge counts must be nonnegative")
        if len(set(self.edges)) != len(self.edges) or any(
            not (0 <= u < v < self.n) for u, v in self.edges
        ):
            raise ValueError("edges must be unique, ordered, and valid")

    def cut(self, edges: Iterable[tuple[int, int]]) -> Tissue:
        removed = {tuple(sorted(edge)) for edge in edges}
        if not removed.issubset(self.edges):
            raise ValueError("can only cut present edges")
        return replace(self, edges=tuple(e for e in self.edges if e not in removed))


def square_grid(side: int = 3, *, coupling: Q = Q(1), anchor: Q = Q(1)) -> Tissue:
    """Ground-clamped square lattice, program on its unique center (odd side)."""
    if side < 3 or side % 2 != 1:
        raise ValueError("side must be odd and >=3")
    edges = []
    for r in range(side):
        for c in range(side):
            u = r * side + c
            if r + 1 < side:
                edges.append((u, u + side))
            if c + 1 < side:
                edges.append((u, u + 1))
    ground = tuple(
        4 - sum(u in e for e in edges) for u in range(side * side)
    )
    return Tissue(side * side, tuple(sorted(edges)), ground,
                  frozenset({(side * side) // 2}), Q(coupling), Q(anchor))


def boundary(t: Tissue, mask: int) -> int:
    """Number of disagreeing internal edges and positive-to-ground edges."""
    if not 0 <= mask < (1 << t.n):
        raise ValueError("invalid spin mask")
    return sum(((mask >> u) ^ (mask >> v)) & 1 for u, v in t.edges) + sum(
        b for u, b in enumerate(t.ground) if mask & (1 << u)
    )


def counts(t: Tissue, mask: int) -> tuple[int, int]:
    p = sum(bool(mask & (1 << u)) for u in t.program)
    return p, mask.bit_count() - p


def energy_gap(t: Tissue, mask: int, field: Q) -> Q:
    """E(mask) - E(all -1); field h acts only on the program region."""
    p, outside = counts(t, mask)
    return 2 * (t.coupling * boundary(t, mask) + t.anchor * outside - Q(field) * p)


def critical_field(t: Tissue, *, max_vertices: int = 20) -> tuple[Q, int]:
    """Exhaustive exact h_crit and one minimising coalition; small graphs only."""
    if t.n > max_vertices:
        raise ValueError("exhaustive search exceeds max_vertices")
    best = None
    for mask in range(1, 1 << t.n):
        p, outside = counts(t, mask)
        if not p:
            continue
        ratio = (t.coupling * boundary(t, mask) + t.anchor * outside) / p
        if best is None or ratio < best[0]:
            best = (ratio, mask)
    assert best is not None
    return best


def enumerate_ground_states(t: Tissue, field: Q, *, max_vertices: int = 20) -> tuple[Q, tuple[int, ...]]:
    if t.n > max_vertices:
        raise ValueError("exhaustive search exceeds max_vertices")
    values = [(energy_gap(t, mask, field), mask) for mask in range(1 << t.n)]
    optimal = min(gap for gap, _ in values)
    return optimal, tuple(mask for gap, mask in values if gap == optimal)


def mincut_ground_state(t: Tissue, field: Q) -> tuple[Q, int, Q]:
    """Exact rational s-t mincut; return gap, minimizing positive-mask, flow.

    For w_i = J*b_i + a*1[outside] - h*1[program],
    Delta E(S)/2 = J*cut(S) + sum_{i in S} w_i.
    Add source->i capacity -w_i for negative w_i; i->sink w_i for
    positive w_i; each graph edge gives a symmetric pair of J arcs.
    Then mincut = sum(max(0,-w_i)) + min(Delta E/2).
    """
    n = t.n
    source, sink = n, n + 1
    residual: list[dict[int, Q]] = [dict() for _ in range(n + 2)]

    def add_arc(u: int, v: int, cap: Q) -> None:
        if cap < 0:
            raise ValueError("negative capacity")
        residual[u][v] = residual[u].get(v, Q(0)) + cap
        residual[v].setdefault(u, Q(0))

    weights = tuple(
        t.coupling * t.ground[i] + (Q(0) if i in t.program else t.anchor)
        - (Q(field) if i in t.program else Q(0))
        for i in range(n)
    )
    offset = sum((-w for w in weights if w < 0), Q(0))
    for i, w in enumerate(weights):
        if w > 0:
            add_arc(i, sink, w)
        elif w < 0:
            add_arc(source, i, -w)
    for u, v in t.edges:
        add_arc(u, v, t.coupling)
        add_arc(v, u, t.coupling)

    flow = Q(0)
    while True:
        parent = {source: -1}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v, capacity in residual[u].items():
                if capacity > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path_capacity = None
        v = sink
        while v != source:
            u = parent[v]
            path_capacity = residual[u][v] if path_capacity is None else min(path_capacity, residual[u][v])
            v = u
        assert path_capacity is not None and path_capacity > 0
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_capacity
            residual[v][u] = residual[v].get(u, Q(0)) + path_capacity
            v = u
        flow += path_capacity

    reachable = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v, cap in residual[u].items():
            if cap > 0 and v not in reachable:
                reachable.add(v)
                stack.append(v)
    mask = sum(1 << i for i in range(n) if i in reachable)
    gap = 2 * (flow - offset)
    assert energy_gap(t, mask, field) == gap, "flow/cut primal mismatch"
    return gap, mask, flow


def influence_cone(t: Tissue, origin: int, steps: int) -> frozenset[int]:
    """Local message passing after <=steps edges; any rule of radius 1."""
    if not 0 <= origin < t.n or steps < 0:
        raise ValueError("invalid origin or steps")
    reached = {origin}
    frontier = {origin}
    for _ in range(steps):
        next_frontier = set()
        for u, v in t.edges:
            if u in frontier and v not in reached:
                next_frontier.add(v)
            if v in frontier and u not in reached:
                next_frontier.add(u)
        frontier = next_frontier - reached
        reached.update(frontier)
    return frozenset(reached)


def experiment() -> dict:
    intact = square_grid()
    center = 4
    lesion = intact.cut([(1, center), (3, center)])
    sham = intact.cut([(0, 1), (7, 8)])
    field = Q(3)
    out = {}
    for name, tissue in (("intact", intact), ("two_center_edges_cut", lesion), ("sham_distant_edges_cut", sham)):
        threshold, witness = critical_field(tissue)
        gap, minimizers = enumerate_ground_states(tissue, field)
        flow_gap, flow_mask, flow = mincut_ground_state(tissue, field)
        assert flow_gap == gap and flow_mask in minimizers
        off_gap, off_minimizers = enumerate_ground_states(tissue, Q(0))
        out[name] = {
            "critical_field": str(threshold),
            "critical_coalition_mask": witness,
            "field": str(field),
            "min_energy_gap": str(gap),
            "ground_state_masks": list(minimizers),
            "mincut_mask": flow_mask,
            "mincut_flow": str(flow),
            "memory_off_min_gap": str(off_gap),
            "memory_off_masks": list(off_minimizers),
            "influence_cone_sizes": [len(influence_cone(tissue, center, k)) for k in range(4)],
        }
    return {
        "model": "3x3 grounded binary tissue, 4-neighbor coupling, J=1, a=1, P={center}, h=3",
        "baseline_mask": 0,
        "programmed_mask": 1 << center,
        "source_sample_seed": "levinverse|2026-11-07|talk-math-3",
        "results": out,
        "evidence": "exact finite enumeration and independent rational s-t maxflow/mincut; no biological calibration",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(experiment(), indent=2, sort_keys=True))
