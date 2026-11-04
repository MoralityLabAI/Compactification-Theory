"""Finite-field cellular sheaf of a cyclic directed machine.

C^0 = direct sum of vertex vector spaces; C^1 = direct sum of edge spaces.
(delta s)_i = s_(i+1 mod m) - A_i s_i over F_p.
A 1-cochain b has a global affine solution iff b belongs to im(delta).
For a single cycle ker(delta) and coker(delta) have the dimensions of
ker(I-M) and coker(I-M), M=A_(m-1)...A_0.
"""
from __future__ import annotations
from typing import Sequence

Matrix = list[list[int]]
Vector = list[int]


def eye(n: int) -> Matrix:
    if n < 1:
        raise ValueError("dimension must be positive")
    return [[int(i == j) for j in range(n)] for i in range(n)]


def matmul(a: Matrix, b: Matrix, p: int) -> Matrix:
    if not a or not b or len(a[0]) != len(b) or any(len(row) != len(a[0]) for row in a) or any(len(row) != len(b[0]) for row in b):
        raise ValueError("incompatible matrix shapes")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) % p
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a: Matrix, x: Vector, p: int) -> Vector:
    if not a or any(len(row) != len(x) for row in a):
        raise ValueError("incompatible dimensions")
    return [sum(v * q for v, q in zip(row, x)) % p for row in a]


def rref(a: Matrix, p: int):
    """Exact Gauss-Jordan elimination for a prime field. Returns RREF and pivot columns."""
    if type(p) is not int or not 2 <= p <= 1009 or any(p % d == 0 for d in range(2, int(p ** .5) + 1)):
        raise ValueError("reference solver requires a prime p in [2, 1009]")
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("nonempty rectangular matrix required")
    t = [[int(x) % p for x in row] for row in a]
    row = 0
    pivots = []
    for col in range(len(t[0])):
        nz = next((r for r in range(row, len(t)) if t[r][col]), None)
        if nz is None:
            continue
        t[row], t[nz] = t[nz], t[row]
        inv = pow(t[row][col], -1, p)
        t[row] = [v * inv % p for v in t[row]]
        for r in range(len(t)):
            if r != row and t[r][col]:
                factor = t[r][col]
                t[r] = [(u - factor * v) % p for u, v in zip(t[r], t[row])]
        pivots.append(col)
        row += 1
        if row >= len(t):
            break
    return t, pivots


def rank(a: Matrix, p: int) -> int:
    return len(rref(a, p)[1])


def solve(a: Matrix, b: Vector, p: int) -> Vector | None:
    """Find any solution of Ax=b, with free variables set to zero."""
    if len(a) != len(b) or not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("bad system size")
    n = len(a[0])
    rr, pivots = rref([row + [b[i]] for i, row in enumerate(a)], p)
    for row in rr:
        if all(v == 0 for v in row[:n]) and row[n] != 0:
            return None
    x = [0] * n
    for i, c in enumerate(pivots):
        if c < n:
            x[c] = rr[i][n]
    return x


def _validate_edges(edges: Sequence[Matrix]):
    if not edges or not edges[0] or not edges[0][0]:
        raise ValueError("edges must be nonempty square matrices")
    d = len(edges[0])
    if any(len(a) != d or any(len(row) != d for row in a) for a in edges):
        raise ValueError("all edge maps must be d by d")
    return d


def coboundary(edges: Sequence[Matrix], p: int) -> Matrix:
    """Matrix of delta: C^0 -> C^1, for the oriented m-edge cycle."""
    d = _validate_edges(edges)
    m = len(edges)
    mat = [[0] * (m*d) for _ in range(m*d)]
    for e, a in enumerate(edges):
        for i in range(d):
            mat[e*d+i][((e+1) % m)*d+i] = (mat[e*d+i][((e+1) % m)*d+i] + 1) % p
            for j in range(d):
                mat[e*d+i][e*d+j] = (mat[e*d+i][e*d+j] - a[i][j]) % p
    return mat


def monodromy(edges: Sequence[Matrix], p: int) -> Matrix:
    d = _validate_edges(edges)
    m = eye(d)
    for a in edges:
        m = matmul(a, m, p)
    return m


def cycle_cohomology(edges: Sequence[Matrix], p: int):
    d = _validate_edges(edges)
    delta = coboundary(edges, p)
    r = rank(delta, p)
    hol = monodromy(edges, p)
    minus = [[(int(i == j) - hol[i][j]) % p for j in range(d)] for i in range(d)]
    small_rank = rank(minus, p)
    result = {
        "p": p, "vertices": len(edges), "fiber_dim": d,
        "rank_delta": r,
        "H0_dimension": len(delta[0]) - r,
        "H1_dimension": len(delta) - r,
        "rank_identity_minus_monodromy": small_rank,
        "ker_identity_minus_monodromy": d - small_rank,
        "coker_identity_minus_monodromy": d - small_rank,
    }
    assert result["H0_dimension"] == result["ker_identity_minus_monodromy"]
    assert result["H1_dimension"] == result["coker_identity_minus_monodromy"]
    return result


def affine_cocycle(edges: Sequence[Matrix], b: Sequence[Vector], p: int):
    """Solve local constraints s_(i+1) - A_i s_i=b_i around a full cycle."""
    d = _validate_edges(edges)
    if len(b) != len(edges) or any(len(v) != d for v in b):
        raise ValueError("bad cocycle length")
    mon = monodromy(edges, p)
    c = [0] * d
    for a, v in zip(edges, b):
        c = [(u+v[i]) % p for i,u in enumerate(matvec(a,c,p))]
    matrix = [[(int(i == j)-mon[i][j]) % p for j in range(d)] for i in range(d)]
    start = solve(matrix, c, p)
    if start is None:
        return {"solvable": False, "obstruction": c, "states": None}
    states = [start]
    for a, v in zip(edges, b):
        states.append([(u+v[i]) % p for i,u in enumerate(matvec(a,states[-1],p))])
    assert states[-1] == states[0]
    return {"solvable": True, "obstruction": c, "states": states[:-1]}
