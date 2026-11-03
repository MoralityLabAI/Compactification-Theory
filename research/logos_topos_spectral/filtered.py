"""Exact finite models of filtered logical interpretations and Dirac blocks.

All metric and matrix computations use fractions.Fraction. A finite sample
is an explicit finite theory model, not a sample-based proof about an
infinite arithmetic completion space.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations, product
from typing import Sequence

Bits = tuple[int, ...]
Matrix2 = tuple[tuple[Q, Q], tuple[Q, Q]]


@dataclass(frozen=True)
class MarkedSpace:
    """Finite, nonempty space with an injective binary marking."""
    points: tuple[Bits, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("A marked space must be nonempty.")
        width = len(self.points[0])
        if any(len(p) != width or any(b not in (0, 1) for b in p) for p in self.points):
            raise ValueError("Marks must be equally sized binary tuples.")
        if len(set(self.points)) != len(self.points):
            raise ValueError("The marking must separate states.")

    @property
    def width(self) -> int:
        return len(self.points[0])

    def dist(self, i: int, j: int) -> Q:
        """2^(-k), where k starts at zero. Equal states have distance zero."""
        if i == j:
            return Q(0)
        return next(Q(1, 2**k) for k, (a, b) in enumerate(zip(self.points[i], self.points[j])) if a != b)

    def agree(self, i: int, j: int, depth: int) -> bool:
        if depth < 0:
            raise ValueError("Depth must be nonnegative.")
        return self.points[i][:depth] == self.points[j][:depth]


@dataclass(frozen=True)
class Interpretation:
    """A map on completions, contravariant to translation of sentences."""
    source: MarkedSpace
    target: MarkedSpace
    image: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.image) != len(self.source.points):
            raise ValueError("The image must name every source state.")
        if any(type(i) is not int or i < 0 or i >= len(self.target.points) for i in self.image):
            raise ValueError("Image index outside the target space.")

    def controls(self, target_depth: int, source_depth: int) -> bool:
        if min(target_depth, source_depth) < 0:
            raise ValueError("Depths must be nonnegative.")
        return all(not self.source.agree(i, j, source_depth)
                   or self.target.agree(self.image[i], self.image[j], target_depth)
                   for i, j in combinations(range(len(self.image)), 2))

    def depth_profile(self) -> tuple[int, ...]:
        """Least h(n) for n=0,...,target.width, found by exhaustive semantics."""
        return tuple(next(m for m in range(self.source.width + 1) if self.controls(n, m))
                     for n in range(self.target.width + 1))

    def least_delay(self) -> int:
        return max(0, *(m - n for n, m in enumerate(self.depth_profile())))

    def has_delay(self, c: int) -> bool:
        if c < 0:
            raise ValueError("Delay must be nonnegative.")
        return all(self.controls(n, n + c) for n in range(self.target.width + 1))

    def lipschitz_constant(self) -> Q:
        return max((self.target.dist(self.image[i], self.image[j]) / self.source.dist(i, j)
                    for i, j in combinations(range(len(self.image)), 2)), default=Q(0))

    def profile_constant(self) -> Q:
        """Exact identity L(f)=sup_{h(n)>0} 2^(h(n)-n)."""
        def dyadic(exponent: int) -> Q:
            return Q(2**exponent) if exponent >= 0 else Q(1, 2**(-exponent))
        return max((dyadic(m-n) for n, m in enumerate(self.depth_profile()) if m > 0), default=Q(0))

    def conservative(self) -> bool:
        return set(self.image) == set(range(len(self.target.points)))

    def then(self, other: Interpretation) -> Interpretation:
        if self.target != other.source:
            raise ValueError("Interpretations have incompatible middle spaces.")
        return Interpretation(self.source, other.target, tuple(other.image[i] for i in self.image))

    def pullback(self, observable: Sequence[Q]) -> tuple[Q, ...]:
        if len(observable) != len(self.target.points):
            raise ValueError("One observable value per target state is required.")
        return tuple(Q(observable[i]) for i in self.image)

    def optimal_pullback_witness(self) -> tuple[tuple[Q, ...], Q]:
        """Distance to a target point attains the optimal unit-ball bound."""
        if self.lipschitz_constant() == 0:
            return (Q(0),) * len(self.target.points), Q(0)
        i, j = max(combinations(range(len(self.image)), 2),
                   key=lambda ij: self.target.dist(self.image[ij[0]], self.image[ij[1]]) / self.source.dist(*ij))
        a = tuple(self.target.dist(k, self.image[j]) for k in range(len(self.target.points)))
        assert seminorm(self.target, a) == 1
        achieved = seminorm(self.source, self.pullback(a))
        assert achieved == self.lipschitz_constant()
        return a, achieved


def seminorm(space: MarkedSpace, a: Sequence[Q]) -> Q:
    if len(a) != len(space.points):
        raise ValueError("One observable value per state is required.")
    return max((abs(Q(a[i])-Q(a[j]))/space.dist(i, j)
                for i, j in combinations(range(len(space.points)), 2)), default=Q(0))


def multiply(a: Matrix2, b: Matrix2) -> Matrix2:
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(2)), Q(0)) for j in range(2)) for i in range(2))


def subtract(a: Matrix2, b: Matrix2) -> Matrix2:
    return tuple(tuple(a[i][j]-b[i][j] for j in range(2)) for i in range(2))


def mass_block(distance: Q, mass: Q) -> Matrix2:
    distance, mass = Q(distance), Q(mass)
    if distance <= 0:
        raise ValueError("A two-point block needs positive distance.")
    return ((mass, 1/distance), (1/distance, -mass))


def block_commutator(distance: Q, mass: Q, a: Q, b: Q) -> Matrix2:
    d = mass_block(distance, mass)
    p = ((Q(a), Q(0)), (Q(0), Q(b)))
    return subtract(multiply(d, p), multiply(p, d))


def eigenvalue_square(distance: Q, mass: Q) -> Q:
    if distance <= 0:
        raise ValueError("A two-point block needs positive distance.")
    return Q(mass)**2 + 1/Q(distance)**2


def resolvent_square_bound(first_omitted_index: int) -> Q:
    """For masses 2^j: upper bound for ||tail (1+D^2)^(-1)||.

    The infinite bound is analytic. This exact formula is checked against
    finite blocks and never used as a certificate of theoremhood.
    """
    if first_omitted_index < 0:
        raise ValueError("The first omitted index must be nonnegative.")
    return Q(1, 1 + 4**first_omitted_index)


def fixture_report() -> dict:
    full = MarkedSpace(tuple(product((0, 1), repeat=5)))
    target = MarkedSpace(tuple(product((0, 1), repeat=3)))
    image = tuple(target.points.index(p[2:]) for p in full.points)
    shift = Interpretation(full, target, image)
    witness, achieved = shift.optimal_pullback_witness()
    return {
        "example": "drop first two observations",
        "source_states": len(full.points), "target_states": len(target.points),
        "depth_profile": shift.depth_profile(), "least_delay": shift.least_delay(),
        "lipschitz_constant": str(shift.lipschitz_constant()),
        "profile_constant": str(shift.profile_constant()),
        "optimal_pullback_bound": str(achieved), "surjective": shift.conservative(),
        "squared_resolvent_tail_bound_at_8": str(resolvent_square_bound(8)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fixture_report(), indent=2))
