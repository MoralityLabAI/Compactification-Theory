"""OIC-06: exact error/erasure-robust hidden-phase model identification.

Only the per-transition freeze audit symbol is accessible in this interface.
Bit substitutions and known-position erasures affect those audit symbols.
Raw, uncorrupted x_t readouts are unavailable in this noisy-channel protocol.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable


def signatures(n: int, variant: str, probes: Iterable[int]) -> frozenset[tuple[int, ...]]:
    """All possible audit-symbol words, allowing unknown hidden phase."""
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if variant not in ("fixed", "rotating"):
        raise ValueError("invalid model family")
    word = tuple(probes)
    if any(u not in (0, 1) for u in word):
        raise ValueError("probe bits must be 0 or 1")
    return frozenset(
        tuple(int(bool(u) and (z + (t if variant == "rotating" else 0)) % n == 0)
              for t, u in enumerate(word))
        for z in range(n)
    )


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if len(a) != len(b):
        raise ValueError("equal-length codewords required")
    return sum(x != y for x, y in zip(a, b))


def interfamily_distance(n: int, probes: Iterable[int]) -> int:
    """Cross-class minimum Hamming distance, minimized over hidden phases."""
    word = tuple(probes)
    a = signatures(n, "fixed", word)
    b = signatures(n, "rotating", word)
    return min(hamming(left, right) for left in a for right in b)


def guaranteed_distance(n: int, length: int) -> int:
    if not isinstance(n, int) or n < 2 or length < 0:
        raise ValueError("n >= 2 and length >= 0 required")
    return length // n


def minimum_robust_horizon(n: int, flips: int = 0, erasures: int = 0) -> int:
    """Sharp T for exact model identification under bounded audit corruption."""
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not isinstance(flips, int) or not isinstance(erasures, int) or flips < 0 or erasures < 0:
        raise ValueError("nonnegative integer corruption budgets required")
    return n * (2 * flips + erasures + 1)


def ambiguous_receipt(
    a: tuple[int, ...], b: tuple[int, ...], flips: int, erasures: int,
) -> tuple[int | None, ...] | None:
    """Construct a common corrupted receipt, when bounded budgets allow it.

    None means the corruption budgets cannot make these codewords coincide.
    Erasures use None and flips are counted on remaining positions.
    """
    if flips < 0 or erasures < 0:
        raise ValueError("corruption budgets must be nonnegative")
    if len(a) != len(b):
        raise ValueError("equal-length codewords required")
    indices = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(indices) > 2 * flips + erasures:
        return None
    receipt: list[int | None] = list(a)
    erase_count = min(erasures, len(indices))
    for i in indices[:erase_count]:
        receipt[i] = None
    remaining = indices[erase_count:]
    flip_a = max(0, len(remaining) - flips)
    for i in remaining[:flip_a]:
        receipt[i] = b[i]
    if flip_a > flips or len(remaining) - flip_a > flips:
        raise AssertionError("ambiguous receipt has excessive substitutions")
    return tuple(receipt)


def corruption_cost(clean: tuple[int, ...], received: tuple[int | None, ...]) -> tuple[int, int]:
    if len(clean) != len(received):
        raise ValueError("equal-length words required")
    return (
        sum(got is not None and expected != got for expected, got in zip(clean, received)),
        sum(got is None for got in received),
    )


def interfamily_ambiguity(
    n: int, probes: Iterable[int], flips: int = 0, erasures: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int | None, ...]] | None:
    word = tuple(probes)
    for left in sorted(signatures(n, "fixed", word)):
        for right in sorted(signatures(n, "rotating", word)):
            observed = ambiguous_receipt(left, right, flips, erasures)
            if observed is not None:
                return (left, right, observed)
    return None


def exhaustive_best_distance(n: int, length: int) -> tuple[int, tuple[int, ...]]:
    """Independent exhaustive search over nonadaptive binary probe schedules."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    best = -1
    word_best: tuple[int, ...] | None = None
    for word in product((0, 1), repeat=length):
        d = interfamily_distance(n, word)
        if d > best:
            best, word_best = d, word
    if word_best is None:
        raise AssertionError("no candidate action word")
    return best, word_best


def reference() -> dict[str, object]:
    rows = []
    for n in (2, 3, 4, 5, 6):
        for flips, erasures in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0)):
            t = minimum_robust_horizon(n, flips, erasures)
            rows.append({
                "n": n, "max_substitutions": flips, "max_erasures": erasures,
                "minimum_horizon": t,
                "distance_at_threshold_minus_one": interfamily_distance(n, (1,) * (t-1)),
                "distance_at_threshold": interfamily_distance(n, (1,) * t),
                "ambiguous_before": interfamily_ambiguity(n, (1,) * (t-1), flips, erasures) is not None,
                "separable_at": interfamily_ambiguity(n, (1,) * t, flips, erasures) is None,
            })
    return {"status": "exact_finite_audit_channel_reference", "rows": rows}
