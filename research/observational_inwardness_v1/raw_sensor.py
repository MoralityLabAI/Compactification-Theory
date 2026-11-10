"""OIC-08/09: raw-state channel, parity latching, and fault tolerance.

A raw readout is the visible state x_t at times 0,...,T, subject to
at most e substitutions and r marked erasures of individual state samples.
Actions are fixed before the episode; raw readouts are not the separate
transition-level audit channel studied in OIC-T06.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

from model import trajectory
from robust import ambiguous_receipt, hamming


def raw_codebook(n: int, variant: str, word: Iterable[int]) -> frozenset[tuple[int, ...]]:
    """All raw visible-state traces from unknown initial hidden phase; x0=0."""
    word = tuple(word)
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if variant not in ("fixed", "rotating"):
        raise ValueError("invalid model family")
    if any(bit not in (0, 1) for bit in word):
        raise ValueError("probe inputs must be binary")
    return frozenset(trajectory(n, variant, z, word) for z in range(n))


def minimum_raw_distance(n: int, word: Iterable[int]) -> int:
    """Minimum state-sample Hamming distance between the two model families."""
    word = tuple(word)
    return min(
        hamming(a, b)
        for a in raw_codebook(n, "fixed", word)
        for b in raw_codebook(n, "rotating", word)
    )


def even_latching_word(n: int, horizon: int) -> tuple[int, ...]:
    """n active probes, then passive readouts, for n even and T>=n."""
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("even n >= 2 required")
    if not isinstance(horizon, int) or horizon < n:
        raise ValueError("horizon must be >= n")
    return (1,) * n + (0,) * (horizon - n)


def even_maximum_raw_distance(n: int, horizon: int) -> int:
    """Sharp maximum interfamily distance over fixed schedules (OIC-T08)."""
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("even n >= 2 required")
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("nonnegative integer horizon required")
    return max(0, horizon - n + 1)


def even_minimum_raw_horizon(n: int, flips: int = 0, erasures: int = 0) -> int:
    """Sharp fixed-schedule horizon with at most flips substitutions and erasures."""
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("even n >= 2 required")
    if not isinstance(flips, int) or not isinstance(erasures, int) or flips < 0 or erasures < 0:
        raise ValueError("nonnegative integer fault budgets required")
    return n + 2 * flips + erasures


def raw_collision(
    n: int, word: Iterable[int], flips: int = 0, erasures: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int | None, ...]] | None:
    """An ambiguous corrupted readout, or None when exact decoding is guaranteed."""
    if not isinstance(flips, int) or not isinstance(erasures, int) or flips < 0 or erasures < 0:
        raise ValueError("nonnegative integer fault budgets required")
    word = tuple(word)
    for a in sorted(raw_codebook(n, "fixed", word)):
        for b in sorted(raw_codebook(n, "rotating", word)):
            receipt = ambiguous_receipt(a, b, flips, erasures)
            if receipt is not None:
                return a, b, receipt
    return None


def best_fixed_word(n: int, horizon: int) -> tuple[int, tuple[int, ...]]:
    """Independent exhaustive fixed-schedule optimization for small n and T."""
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("nonnegative integer horizon required")
    if not isinstance(n, int) or n < 2:
        raise ValueError("n >= 2 required")
    return max(
        (minimum_raw_distance(n, word), word)
        for word in product((0, 1), repeat=horizon)
    )


def terminal_separated(n: int, word: Iterable[int]) -> bool:
    """Whether the final raw bit alone identifies the model family."""
    word = tuple(word)
    a = {trace[-1] for trace in raw_codebook(n, "fixed", word)}
    b = {trace[-1] for trace in raw_codebook(n, "rotating", word)}
    return a.isdisjoint(b)


def raw_to_audit(readouts: Iterable[int | None]) -> tuple[int | None, ...]:
    """Derive audit signal 1 xor x_t xor x_(t+1) from raw state readouts.

    Any missing raw sample erases both adjacent derived audit entries.
    """
    word = tuple(readouts)
    if not word or any(bit not in (0, 1, None) for bit in word):
        raise ValueError("readouts must be a nonempty binary/erasure sequence")
    return tuple(
        None if a is None or b is None else (1 ^ a ^ b)
        for a, b in zip(word, word[1:])
    )


def reference() -> dict[str, object]:
    rows = []
    for n in (2, 4, 6, 8):
        for flips, erasures in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0)):
            t = even_minimum_raw_horizon(n, flips, erasures)
            before = even_latching_word(n, t - 1) if t > n else (1,) * (n - 1)
            witness = raw_collision(n, before, flips, erasures)
            at = even_latching_word(n, t)
            rows.append({
                "hidden_phases": n,
                "max_raw_flips": flips,
                "max_raw_erasures": erasures,
                "minimum_horizon": t,
                "active_probes_at_threshold": n,
                "max_cross_distance_at_threshold_minus_one": even_maximum_raw_distance(n, t - 1),
                "cross_distance_at_threshold": minimum_raw_distance(n, at),
                "collision_before": witness is not None,
                "separated_at_threshold": raw_collision(n, at, flips, erasures) is None,
            })
    odd_counterexamples = []
    for n, flips, horizon in ((3, 1, 7), (5, 1, 7), (3, 2, 10)):
        best, word = best_fixed_word(n, horizon)
        odd_counterexamples.append({
            "hidden_phases": n, "max_raw_flips": flips,
            "first_checked_success_horizon": horizon,
            "best_distance": best,
            "witness_word": "".join(map(str, word)),
        })
    return {
        "status": "exact_finite_raw_state_readout_reference",
        "rows": rows,
        "odd_examples": odd_counterexamples,
        "claim_scope": "fixed_action_schedules_and_raw_state_sensor_faults",
    }
