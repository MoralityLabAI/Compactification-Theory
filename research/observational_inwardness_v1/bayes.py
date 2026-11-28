"""OIC-04: exact Bayesian model-discrimination curve under uniform hidden phase."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product

from model import trajectory


def trace_distribution(n: int, variant: str, word: tuple[int, ...]) -> dict[tuple[int, ...], Fraction]:
    if n < 2:
        raise ValueError("n must be >= 2")
    counts = Counter(trajectory(n, variant, z, word) for z in range(n))
    return {trace: Fraction(count, n) for trace, count in counts.items()}


def total_variation(n: int, word: tuple[int, ...]) -> Fraction:
    a = trace_distribution(n, "fixed", word)
    b = trace_distribution(n, "rotating", word)
    return sum((abs(a.get(k, 0) - b.get(k, 0)) for k in a.keys() | b.keys()), Fraction()) / 2


def bayes_error(n: int, word: tuple[int, ...]) -> Fraction:
    """Optimal 0-1 classification error with equal system priors, uniform hidden phase."""
    return (1 - total_variation(n, word)) / 2


def predicted_error(n: int, horizon: int) -> Fraction:
    if n < 2 or horizon < 0:
        raise ValueError("invalid phase count or horizon")
    if horizon <= 1:
        return Fraction(1, 2)
    return Fraction(max(n - horizon, 0), 2 * n)


def optimal_fixed_schedule(n: int, horizon: int) -> tuple[Fraction, tuple[int, ...]]:
    """Independent exhaustive search over all nonadaptive schedules of this length."""
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    return min(
        (bayes_error(n, word), word)
        for word in product((0, 1), repeat=horizon)
    )


def curve(n: int) -> list[dict[str, int | str]]:
    records = []
    for t in range(n + 1):
        actual = bayes_error(n, (1,) * t)
        predicted = predicted_error(n, t)
        if actual != predicted:
            raise AssertionError(f"Bayes formula mismatch at n={n}, t={t}")
        records.append({"horizon": t, "total_variation": str(total_variation(n, (1,) * t)),
                        "bayes_error": str(actual)})
    return records
