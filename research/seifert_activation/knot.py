"""Closed-braid observables for cyclic feature interventions.

Algebra is exact (SymPy). Braid extraction is piecewise-linear and reports
only topology of a specified two-dimensional observable chart, never that of
the original high-dimensional neural representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import sympy as sp

t = sp.Symbol('t')


class InvalidBraidTrace(ValueError):
    """Observed trace fails the explicitly checked closed-braid contract."""


def reduced_generator(n: int, index: int) -> sp.Matrix:
    """Reduced Burau matrix for Artin generator sigma_index, 1-based."""
    if n < 2 or not (1 <= index < n):
        raise ValueError('Require n >= 2 and 1 <= index < n')
    if n == 2:
        return sp.Matrix([[-t]])
    m = sp.eye(n - 1)
    if index == 1:
        m[0, 0], m[0, 1] = -t, 1
    elif index == n - 1:
        m[n - 3, n - 3] = 1
        m[n - 3, n - 2] = 0
        m[n - 2, n - 3] = t
        m[n - 2, n - 2] = -t
    else:
        j = index - 2
        m[j + 1, j] = t
        m[j + 1, j + 1] = -t
        m[j + 1, j + 2] = 1
    return m


def burau(n: int, word: Sequence[int]) -> sp.Matrix:
    """Word in Artin generators; negative values denote inverses."""
    M = sp.eye(n - 1)
    for g in word:
        if g == 0 or abs(g) >= n:
            raise ValueError(f'Invalid generator {g} for B_{n}')
        G = reduced_generator(n, abs(g))
        M = M * (G if g > 0 else G.inv())
    return M.applyfunc(sp.cancel)


def normalized_alexander(n: int, word: Sequence[int]) -> sp.Expr:
    """One-variable Alexander polynomial, up to units ±t^k.

    Canonical output has integer coefficients, minimal exponent 0, positive
    leading coefficient. For multi-component links, zero is possible.
    This cannot distinguish distinct spanning surfaces of the same link.
    """
    M = burau(n, word)
    delta = sp.cancel((1 - t) * (sp.eye(n - 1) - M).det() / (1 - t**n))
    if delta == 0:
        return sp.Integer(0)
    terms = sp.Add.make_args(sp.expand(delta))
    powers = []
    for term in terms:
        coeff, exponent = term.as_coeff_exponent(t)
        if not exponent.is_Integer or not coeff.is_Integer:
            raise ArithmeticError(f'Alexander expression not in Z[t,t^-1]: {delta}')
        powers.append((int(exponent), int(coeff)))
    lo = min(k for k, _ in powers)
    polynomial = sp.expand(sum(c * t**(k - lo) for k, c in powers))
    lead = sp.Poly(polynomial, t).LC()
    return sp.expand(polynomial if lead > 0 else -polynomial)


@dataclass(frozen=True)
class Extraction:
    n_strands: int
    word: tuple[int, ...]
    crossings: tuple[tuple[float, int, int, int], ...]
    # crossings: (fractional sample index, strand_a, strand_b, signed generator)


def extract_closed_braid(frames: Sequence[Sequence[Sequence[float]]],
                         *, clearance: float = 1e-7,
                         simultaneity: float = 1e-7) -> Extraction:
    """Recover a braid from ordered (sample, strand-id, x/y) observations.

    Contract: frame zero and last have the same *unordered* point set;
    strands are labeled consistently across frames; x-order crossings are
    isolated, transverse and separated in y. Piecewise linear interpolation
    is assumed between frames, and no claim is made about unobserved curves.
    """
    if len(frames) < 2:
        raise InvalidBraidTrace('At least two frames required')
    n = len(frames[0])
    if n < 2:
        raise InvalidBraidTrace('At least two strands required')
    ps = []
    for row in frames:
        if len(row) != n:
            raise InvalidBraidTrace('Strand count changes')
        ps.append([])
        for p in row:
            if len(p) != 2 or not all(math.isfinite(float(q)) for q in p):
                raise InvalidBraidTrace('Non-finite or malformed feature point')
            ps[-1].append((float(p[0]), float(p[1])))
        for i in range(n):
            for j in range(i+1, n):
                if math.dist(ps[-1][i], ps[-1][j]) <= clearance:
                    raise InvalidBraidTrace('Strand collision in sampled chart')
                if abs(ps[-1][i][0]-ps[-1][j][0]) <= clearance:
                    raise InvalidBraidTrace('x-coordinate tie on sampled slice')
    start, end = sorted(ps[0]), sorted(ps[-1])
    if any(math.dist(a,b) > clearance for a,b in zip(start,end)):
        raise InvalidBraidTrace('Open trace: no canonical endpoint closure')
    candidates = []
    for k in range(len(ps) - 1):
        a, b = ps[k], ps[k+1]
        for i in range(n):
            for j in range(i+1, n):
                d0, d1 = a[i][0]-a[j][0], b[i][0]-b[j][0]
                if d0*d1 >= 0:
                    continue
                u = d0/(d0-d1)
                ydiff0, ydiff1 = a[i][1]-a[j][1], b[i][1]-b[j][1]
                dy = ydiff0 + u*(ydiff1-ydiff0)
                if abs(dy) <= clearance:
                    raise InvalidBraidTrace('Unresolved over/under crossing')
                x_cross = a[i][0] + u*(b[i][0]-a[i][0])
                for other in range(n):
                    if other not in (i,j):
                        xother = a[other][0]+u*(b[other][0]-a[other][0])
                        if abs(xother-x_cross) <= clearance:
                            raise InvalidBraidTrace('Triple x crossing')
                # Rank just before crossing; all x values are distinct there.
                v = max(0., u - min(1e-4, u/2))
                before = [a[s][0]+v*(b[s][0]-a[s][0]) for s in range(n)]
                order = sorted(range(n), key=lambda s: before[s])
                rank_i, rank_j = order.index(i), order.index(j)
                if abs(rank_i-rank_j) != 1:
                    raise InvalidBraidTrace('Non-adjacent projected crossing')
                g = min(rank_i,rank_j)+1
                crossing_orientation = (d1-d0)*dy
                if crossing_orientation < 0:
                    g = -g
                candidates.append((k+u, i, j, g))
    candidates.sort()
    for left, right in zip(candidates, candidates[1:]):
        if abs(left[0] - right[0]) <= simultaneity:
            raise InvalidBraidTrace('Simultaneous crossings; refine sampling')
    return Extraction(n, tuple(item[3] for item in candidates), tuple(candidates))


def synthetic_braid(n: int, word: Sequence[int]) -> list[list[tuple[float,float]]]:
    """Piecewise-linear sample fixture realizing Artin generators as 3D strands.

    The second observable is over/under depth. This is a mathematical fixture,
    not a trace from a neural model.
    """
    order = list(range(n))
    xs = [2*i-(n-1) for i in range(n)]
    frames: list[list[tuple[float,float]]] = [[]]
    def row_for(swap_slot=None, phase=0., sign=1):
        points = [(0.,0.) for _ in range(n)]
        for slot, identity in enumerate(order):
            x,y = xs[slot],0.
            if swap_slot == slot:
                x += 2*phase
                y = sign*math.sin(math.pi*phase)
            elif swap_slot is not None and swap_slot+1 == slot:
                x -= 2*phase
                y = -sign*math.sin(math.pi*phase)
            points[identity] = (x,y)
        return points
    frames[0] = row_for()
    for g in word:
        if g == 0 or abs(g) >= n:
            raise ValueError('Invalid braid word')
        slot, sign = abs(g)-1, (1 if g>0 else -1)
        for phase in (0.25, 0.75):
            frames.append(row_for(slot, phase, sign))
        order[slot],order[slot+1] = order[slot+1],order[slot]
        frames.append(row_for())
    if not word:
        frames.append(row_for())
    return frames


@dataclass(frozen=True)
class SeifertSurfaceStats:
    """Combinatorics of the oriented Seifert-algorithm surface of a braid."""
    disks: int
    bands: int
    surface_components: int
    boundary_components: int
    euler_characteristic: int
    total_genus: int


def braid_seifert_surface(n: int, word: Sequence[int]) -> SeifertSurfaceStats:
    """The canonical Seifert surface: n disks and one band per crossing.

    In general this is *a* spanning surface, not a minimal-genus surface.
    Its genus is presentation-dependent. Conjugation and stabilization can
    change the presentation; crossing cancellation can increase the surface
    genus without changing the link boundary.
    """
    if n < 2:
        raise ValueError('n must be >=2')
    parents = list(range(n))
    slots = list(range(n))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for g in word:
        if g == 0 or abs(g) >= n:
            raise ValueError(f'Invalid generator {g} for B_{n}')
        i = abs(g) - 1
        parents[root(i+1)] = root(i)
        slots[i], slots[i+1] = slots[i+1], slots[i]
    C = len({root(i) for i in range(n)})
    # Each boundary component follows a cycle of the braid permutation.
    visited = set()
    mu = 0
    for i in range(n):
        if i in visited:
            continue
        mu += 1
        j = i
        while j not in visited:
            visited.add(j)
            j = slots[j]
    chi = n - len(word)
    two_genus = 2*C - mu - chi
    if two_genus < 0 or two_genus % 2:
        raise ArithmeticError('Inconsistent oriented braid surface genus')
    return SeifertSurfaceStats(n, len(word), C, mu, chi, two_genus//2)

