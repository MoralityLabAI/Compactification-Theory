"""Exact finite Dirac-block prototype for marked completion spaces.

For each pair i<j use D_ij = [[0, 1/d_ij], [1/d_ij, 0]] and
pi_ij(a) = diag(a_i, a_j). Thus ||[D,pi(a)]|| = max |a_i-a_j|/d_ij.
This is a finite construction; no infinite compact-resolvent claim is made.
"""
from fractions import Fraction
from itertools import combinations
from typing import Sequence

Matrix = tuple[tuple[Fraction, ...], ...]


def validate_metric(d: Matrix) -> None:
    n = len(d)
    if n < 2 or any(len(row) != n for row in d):
        raise ValueError('A square matrix with at least two points is required.')
    for i in range(n):
        for j in range(n):
            if d[i][j] != d[j][i] or (i == j and d[i][j] != 0):
                raise ValueError('Metric symmetry or zero diagonal failed.')
            if i != j and d[i][j] <= 0:
                raise ValueError('Distinct points must have positive distance.')
            for k in range(n):
                if d[i][j] > d[i][k] + d[k][j]:
                    raise ValueError('Triangle inequality failed.')


def ultrametric(
    points: Sequence[Sequence[int]],
    order: Sequence[int] | None = None,
    weights: Sequence[Fraction] | None = None,
) -> Matrix:
    p = tuple(tuple(x) for x in points)
    if len(p) < 2 or len(set(p)) != len(p):
        raise ValueError('At least two distinct completion samples are required.')
    width = len(p[0])
    if width == 0 or any(len(x) != width or any(b not in (0, 1) for b in x) for x in p):
        raise ValueError('Points must be equally sized, nonempty binary tuples.')
    order = tuple(range(width)) if order is None else tuple(order)
    if sorted(order) != list(range(width)):
        raise ValueError('Sentence order must be a permutation of all coordinates.')
    w = tuple(Fraction(1, 2 ** (i + 1)) for i in range(width)) if weights is None else tuple(map(Fraction, weights))
    if len(w) != width or any(x <= 0 for x in w) or any(a <= b for a, b in zip(w, w[1:])):
        raise ValueError('Weights must be positive, strictly decreasing, and match the width.')

    def distance(x, y):
        if x == y:
            return Fraction(0)
        return next(w[k] for k, i in enumerate(order) if x[i] != y[i])

    d = tuple(tuple(distance(x, y) for y in p) for x in p)
    validate_metric(d)
    return d


def dirac_spectrum(d: Matrix) -> tuple[Fraction, ...]:
    """Eigenvalues, with multiplicity, of the direct sum of two-point blocks."""
    validate_metric(d)
    return tuple(sorted(value for i, j in combinations(range(len(d)), 2)
                        for value in (-1 / d[i][j], 1 / d[i][j])))


def commutator_norm(d: Matrix, a: Sequence[Fraction]) -> Fraction:
    validate_metric(d)
    if len(a) != len(d):
        raise ValueError('One observable value per point is required.')
    a = tuple(map(Fraction, a))
    return max(abs(a[i] - a[j]) / d[i][j] for i, j in combinations(range(len(d)), 2))


def recovery_certificates(d: Matrix) -> tuple[tuple[int, int, Fraction, Fraction], ...]:
    """A distance-to-target observable saturates each Connes-distance bound.

    The upper bound is the corresponding pair constraint in the seminorm.
    A returned certificate contains (source, target, achieved, seminorm).
    """
    validate_metric(d)
    certificates = []
    for i, j in combinations(range(len(d)), 2):
        a = tuple(d[k][j] for k in range(len(d)))
        seminorm = commutator_norm(d, a)
        achieved = abs(a[i] - a[j])
        if seminorm > 1 or achieved != d[i][j]:
            raise AssertionError('Exact metric-recovery certificate failed.')
        certificates.append((i, j, achieved, seminorm))
    return tuple(certificates)


if __name__ == '__main__':
    points = ((0, 0, 0), (0, 0, 1), (0, 1, 1))
    for order in ((0, 1, 2), (2, 1, 0)):
        d = ultrametric(points, order=order)
        print('sentence_order:', order)
        print('pair_distances:', [str(d[i][j]) for i, j in combinations(range(len(d)), 2)])
        print('Dirac_spectrum:', [str(x) for x in dirac_spectrum(d)])
        print('exact_recovery_certificates:', len(recovery_certificates(d)))
