"""SIML-01 arithmetic reference. No models, broker, training, or authorization.

Intervals require a frozen candidate/comparator and conditionally IID, fresh
paired block scores. This module cannot establish those acquisition conditions.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Iterable, Mapping


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def attempt_alpha(total_alpha: Fraction, attempt: int) -> Fraction:
    """Every attempted promotion consumes its index, including failed attempts."""
    _positive_integer(attempt, "attempt")
    if not isinstance(total_alpha, Fraction) or not 0 < total_alpha < 1:
        raise ValueError("total_alpha must be a Fraction strictly between 0 and 1")
    return total_alpha / (attempt * (attempt + 1))


def endpoint_alpha(total_alpha: Fraction, attempt: int, endpoints: int) -> Fraction:
    _positive_integer(endpoints, "endpoints")
    return attempt_alpha(total_alpha, attempt) / endpoints


def paired_interval(differences: Iterable[float], alpha: float) -> tuple[float, float]:
    """Two-sided Hoeffding interval for E[candidate score - comparator score].

    Each score lies in [0, 1], so the paired difference has range length TWO.
    One item here is an independent episode/world block, not a token or turn.
    alpha is already allocated to this endpoint and this fixed sample size.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    values = [float(x) for x in differences]
    if not values or any(not math.isfinite(x) or abs(x) > 1 for x in values):
        raise ValueError("need nonempty finite differences in [-1, 1]")
    mean = math.fsum(values) / len(values)
    radius = math.sqrt(2 * math.log(2 / alpha) / len(values))
    return max(-1.0, mean - radius), min(1.0, mean + radius)


def passes_declared_floors(
    intervals: Mapping[str, tuple[float, float]],
    floors: Mapping[str, float],
) -> bool:
    """Require the exact endpoint vocabulary and every declared lower floor.

    A False result includes insufficient evidence. It does not distinguish
    that case from a demonstrated regression. The future broker must do so.
    """
    if not floors or set(intervals) != set(floors):
        raise ValueError("intervals and nonempty floors must have identical keys")
    for name, floor in floors.items():
        lower, upper = intervals[name]
        if not all(math.isfinite(x) for x in (lower, upper, floor)):
            raise ValueError("nonfinite interval or floor")
        if not -1 <= lower <= upper <= 1 or not -1 <= floor <= 1:
            raise ValueError("intervals and floors must lie in [-1, 1]")
    return all(intervals[name][0] >= floor for name, floor in floors.items())


def finite_class_radius(description_bits: int, words: int, n: int, delta: float) -> float:
    """Two-sided bounded-loss finite-family union-bound radius, before clipping.

    The full predeclared description family has <= 2**description_bits elements.
    This function does not prove a model's actual description fits that budget.
    """
    if isinstance(description_bits, bool) or not isinstance(description_bits, int) or description_bits < 0:
        raise ValueError("description_bits must be a nonnegative integer")
    _positive_integer(words, "words")
    _positive_integer(n, "n")
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return math.sqrt((description_bits * math.log(2) + math.log(2 * words / delta)) / (2 * n))
