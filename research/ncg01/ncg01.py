"""NCG-01: exact finite-field boundary constraints and unnormalized gluing.

This is a finite abelian toy model. It does not implement a gravitational
phase space, area operator, dynamics, or continuum limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


@dataclass(frozen=True)
class Graph:
    """Oriented multigraph; loops and isolated vertices are allowed."""

    vertices: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.vertices < 0:
            raise ValueError("negative vertex count")
        if any(not (0 <= u < self.vertices and 0 <= v < self.vertices)
               for u, v in self.edges):
            raise ValueError("edge endpoint outside graph")

    def incidence(self, p: int) -> list[list[int]]:
        if not is_prime(p):
            raise ValueError("p must be prime")
        rows = [[0] * len(self.edges) for _ in range(self.vertices)]
        for e, (u, v) in enumerate(self.edges):
            rows[u][e] = (rows[u][e] - 1) % p
            rows[v][e] = (rows[v][e] + 1) % p
        return rows

    def components(self) -> tuple[tuple[int, ...], ...]:
        adj = [set() for _ in range(self.vertices)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        unseen = set(range(self.vertices))
        result: list[tuple[int, ...]] = []
        while unseen:
            pending = [min(unseen)]
            visited: set[int] = set()
            while pending:
                v = pending.pop()
                if v in visited:
                    continue
                visited.add(v)
                pending.extend(sorted(adj[v] - visited, reverse=True))
            unseen -= visited
            result.append(tuple(sorted(visited)))
        return tuple(result)

    def divergence(self, flow: Sequence[int], p: int) -> tuple[int, ...]:
        if len(flow) != len(self.edges):
            raise ValueError("wrong flow length")
        return tuple(sum(a * x for a, x in zip(row, flow)) % p
                     for row in self.incidence(p))


@dataclass(frozen=True)
class AffineCode:
    """Canonical free-coordinate codec for one compatible charge sector."""

    graph: Graph
    p: int
    charges: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    pivots: tuple[int, ...]
    free: tuple[int, ...]
    compatible: bool

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert span of distinct admissible basis labels."""
        return self.p ** len(self.free) if self.compatible else 0

    @property
    def rank(self) -> int:
        return len(self.pivots)

    def decode(self, coordinates: Sequence[int]) -> tuple[int, ...]:
        if not self.compatible:
            raise ValueError("incompatible charge sector")
        if len(coordinates) != len(self.free):
            raise ValueError("wrong coordinate length")
        if any(not isinstance(z, int) or not 0 <= z < self.p
               for z in coordinates):
            raise ValueError("coordinates must be canonical field elements")
        flow = [0] * len(self.graph.edges)
        for j, z in zip(self.free, coordinates):
            flow[j] = z
        for row, pivot in zip(self.rows, self.pivots):
            flow[pivot] = (row[-1] - sum(row[j] * flow[j]
                                           for j in self.free)) % self.p
        return tuple(flow)

    def encode(self, flow: Sequence[int]) -> tuple[int, ...]:
        if len(flow) != len(self.graph.edges):
            raise ValueError("wrong flow length")
        if any(not isinstance(x, int) or not 0 <= x < self.p for x in flow):
            raise ValueError("flow must use canonical field elements")
        if not self.compatible or self.graph.divergence(flow, self.p) != self.charges:
            raise ValueError("flow violates the requested charges")
        return tuple(flow[j] for j in self.free)

    def flows(self) -> Iterable[tuple[int, ...]]:
        if self.compatible:
            for coordinates in product(range(self.p), repeat=len(self.free)):
                yield self.decode(coordinates)


def solve(graph: Graph, p: int, charges: Sequence[int]) -> AffineCode:
    """Gauss-Jordan elimination, without floating point or a symbolic library."""
    if len(charges) != graph.vertices:
        raise ValueError("wrong charge length")
    if any(not isinstance(b, int) or not 0 <= b < p for b in charges):
        raise ValueError("charges must be canonical field elements")
    matrix = graph.incidence(p)
    augmented = [row + [b] for row, b in zip(matrix, charges)]
    pivot_cols: list[int] = []
    cursor = 0
    for col in range(len(graph.edges)):
        pivot_row = next((i for i in range(cursor, graph.vertices)
                          if augmented[i][col] % p), None)
        if pivot_row is None:
            continue
        augmented[cursor], augmented[pivot_row] = augmented[pivot_row], augmented[cursor]
        inverse = pow(augmented[cursor][col], -1, p)
        augmented[cursor] = [(a * inverse) % p for a in augmented[cursor]]
        for i, row in enumerate(augmented):
            if i != cursor and row[col]:
                factor = row[col]
                augmented[i] = [(a - factor * b) % p
                                for a, b in zip(row, augmented[cursor])]
        pivot_cols.append(col)
        cursor += 1
        if cursor == graph.vertices:
            break
    compatible = all(any(row[:-1]) or row[-1] == 0 for row in augmented)
    free_cols = tuple(i for i in range(len(graph.edges)) if i not in pivot_cols)
    return AffineCode(graph, p, tuple(charges),
                      tuple(tuple(row) for row in augmented[:len(pivot_cols)]),
                      tuple(pivot_cols), free_cols, compatible)


def predicted_dimension(graph: Graph, p: int, charges: Sequence[int]) -> int:
    if len(charges) != graph.vertices:
        raise ValueError("wrong charge length")
    if not is_prime(p):
        raise ValueError("p must be prime")
    components = graph.components()
    if any(sum(charges[v] for v in component) % p for component in components):
        return 0
    return p ** (len(graph.edges) - graph.vertices + len(components))


def amplitude(graph: Graph, p: int, charges: Sequence[int],
              weights: Sequence[Sequence[int]]) -> int:
    """Sum signed integer products, retaining the absolute normalization."""
    if len(weights) != len(graph.edges) or any(len(w) != p for w in weights):
        raise ValueError("one length-p weight vector is required for every edge")
    total = 0
    for flow in solve(graph, p, charges).flows():
        weight = 1
        for w, x in zip(weights, flow):
            weight *= w[x]
        total += weight
    return total


def amplitude_table(graph: Graph, p: int, weights: Sequence[Sequence[int]]) -> dict[tuple[int, ...], int]:
    """All nonzero boundary amplitudes, evaluated through the affine codec."""
    table: dict[tuple[int, ...], int] = {}
    for charges in product(range(p), repeat=graph.vertices):
        value = amplitude(graph, p, charges, weights)
        if value:
            table[charges] = value
    return table


def convolve(left: dict[tuple[int, ...], int],
             right: dict[tuple[int, ...], int], p: int) -> dict[tuple[int, ...], int]:
    """Glue charge amplitudes by q_left + q_right = q_total in F_p."""
    if not is_prime(p):
        raise ValueError("p must be prime")
    result: dict[tuple[int, ...], int] = {}
    for a, av in left.items():
        for b, bv in right.items():
            if len(a) != len(b):
                raise ValueError("boundary charge widths disagree")
            key = tuple((x + y) % p for x, y in zip(a, b))
            result[key] = result.get(key, 0) + av * bv
    return {key: value for key, value in result.items() if value}


def glue_shared_boundary(left: Graph, right: Graph, p: int,
                         charges: Sequence[int], boundary: Sequence[int],
                         left_weights: Sequence[Sequence[int]],
                         right_weights: Sequence[Sequence[int]]) -> int:
    """Glue subgraphs embedded in a common vertex set along a supplied cut.

    Vertices with incident edges on both sides must be in `boundary`.
    Isolated shared vertices are allowed and contribute only zero local charge.
    """
    if left.vertices != right.vertices or len(charges) != left.vertices:
        raise ValueError("vertex counts disagree")
    n = left.vertices
    if len(set(boundary)) != len(boundary) or any(not 0 <= v < n for v in boundary):
        raise ValueError("invalid boundary vertices")
    left_incident = set(v for edge in left.edges for v in edge)
    right_incident = set(v for edge in right.edges for v in edge)
    if (left_incident & right_incident) - set(boundary):
        raise ValueError("boundary omits a shared incident vertex")
    total = 0
    for q in product(range(p), repeat=len(boundary)):
        a = [0] * n
        b = [0] * n
        for v in range(n):
            if v not in boundary:
                if v in left_incident:
                    a[v] = charges[v]
                else:
                    b[v] = charges[v]
        for v, x in zip(boundary, q):
            a[v] = x
            b[v] = (charges[v] - x) % p
        total += amplitude(left, p, a, left_weights) * amplitude(right, p, b, right_weights)
    return total
