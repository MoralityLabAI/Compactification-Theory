"""Information price of a finite summary, under a *synthetic uniform prior*.

No theological model, empirical corpus, or learned policy is represented here.
"""
from __future__ import annotations

from collections import Counter
from math import log2, prod
from typing import Hashable, Sequence


def max_exact_recovery_probability(summary: Sequence[Hashable]) -> float:
    """Optimal exact inverse success for uniform preimages of a given summary.

    Each summary class can yield one correct source label at most. Thus the
    probability is number of nonempty classes / number of source states.
    """
    if not summary:
        raise ValueError("summary must have at least one state")
    return len(set(summary)) / len(summary)


def conditional_entropy_bits(summary: Sequence[Hashable]) -> float:
    """H(X | f(X)) for X uniform over distinct input states indexed by summary."""
    if not summary:
        raise ValueError("summary must have at least one state")
    counts = Counter(summary)
    n = len(summary)
    return sum((size / n) * log2(size) for size in counts.values())


def uniform_label_space(axes_cardinalities: Sequence[int]) -> tuple[int, float, float]:
    """M, success probability, and erased bits for a constant summary of M labels.

    Requires independent, uniformly distributed axes of the given finite sizes.
    """
    if not axes_cardinalities or any(type(k) is not int or k < 1 for k in axes_cardinalities):
        raise ValueError("positive integer cardinalities required")
    m = prod(axes_cardinalities)
    return m, 1 / m, log2(m)
