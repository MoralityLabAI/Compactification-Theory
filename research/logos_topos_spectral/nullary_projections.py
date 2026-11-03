"""Exact finite logical projections and Dirac commutators.

This is a finite audit of the Lean LTR-06 proof obligations in
OntologicalProofs: a Boolean truth locus is an idempotent projection,
its sup norm is either zero or one, and a two-point Dirac commutator
detects whether a marked sentence separates its two states.

The finite tests do not prove infinitary completeness, quantifier
elimination, or a canonical infinite Dirac operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations, product

from filtered import MarkedSpace, block_commutator


@dataclass(frozen=True)
class TruthLocus:
    """Truth assignment for a finite set of possible completions."""
    values: tuple[bool, ...]

    def __post_init__(self):
        if not self.values:
            raise ValueError("The completion space must be nonempty.")
        if any(type(v) is not bool for v in self.values):
            raise ValueError("The values must be Boolean.")

    def refutation(self) -> "TruthLocus":
        return TruthLocus(tuple(not v for v in self.values))

    def projection(self) -> tuple[Q, ...]:
        return tuple(Q(1) if b else Q(0) for b in self.values)

    def norm(self) -> Q:
        return max(abs(v) for v in self.projection())

    def entails(self) -> bool:
        return all(self.values)

    def satisfiable(self) -> bool:
        return any(self.values)


def sentence_projection(space: MarkedSpace, sentence) -> TruthLocus:
    """Evaluate a finite Boolean formula on every listed one-element model."""
    return TruthLocus(tuple(bool(sentence(bits)) for bits in space.points))


def dirac_sentence_commutator(
    space: MarkedSpace,
    locus: TruthLocus,
    i: int,
    j: int,
    mass: Q = Q(1),
):
    """The exact real two-point Dirac commutator for a logical projection."""
    if len(space.points) != len(locus.values):
        raise ValueError("Each complete theory needs exactly one truth value.")
    if i == j:
        raise ValueError("The two selected completions must be different.")
    scale = Q(1) / space.dist(i, j)
    a, b = locus.projection()[i], locus.projection()[j]
    return block_commutator(space.dist(i, j), mass, a, b)


def dirac_sentence_seminorm(
    space: MarkedSpace,
    locus: TruthLocus,
    mass: Q = Q(1),
) -> Q:
    """Finite direct-sum commutator operator norm, exactly in rationals.

    Every real 2x2 commutator block [[0,t],[-t,0]] has operator norm
    abs(t). The norm of a finite block direct sum is the maximum norm
    of its summands.
    """
    return max((abs(dirac_sentence_commutator(space, locus, i, j, mass)[0][1])
                for i, j in combinations(range(len(space.points)), 2)),
               default=Q(0))


def refutation_norm_gap(locus: TruthLocus) -> tuple[Q, bool]:
    """Norm zero exactly when the underlying sentence is entailed."""
    return locus.refutation().norm(), locus.entails()


if __name__ == "__main__":
    space = MarkedSpace(tuple(product((0, 1), repeat=3)))
    p0 = sentence_projection(space, lambda x: x[0] == 1)
    tautology = sentence_projection(space, lambda x: x[0] or not x[0])
    for name, locus in (("P_0", p0), ("P_0 OR NOT P_0", tautology)):
        norm, entails = refutation_norm_gap(locus)
        print(f"{name}: refutation norm={norm}, semantic consequence={entails}")
    print("P_0 pair commutator:", dirac_sentence_commutator(space, p0, 0, 4))
