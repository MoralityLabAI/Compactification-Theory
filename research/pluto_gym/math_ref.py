"""Exact and information-theoretic references for the synthetic IRC fixture."""

from __future__ import annotations

import math
import operator


def _int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} is below its minimum")
    return result


def deadline_capacity(n: int, c: int, s: int, D: int) -> int:
    """Exact command count when the first of n equal-weight fronts is selected.

    Each front needs one command of duration c; changing fronts takes s. All
    jobs are available at zero and share deadline D.
    """
    n, c, s, D = (_int(n, "n"), _int(c, "c", 1), _int(s, "s"), _int(D, "D"))
    return min(n, (D + s) // (c + s))


def gf2_rank(rows: list[int] | tuple[int, ...], *, width: int | None = None) -> int:
    """Rank of binary row vectors packed as nonnegative integers."""
    if not isinstance(rows, (tuple, list)):
        raise ValueError("rows must be a sequence")
    vectors = [_int(row, "row") for row in rows]
    if width is not None:
        width = _int(width, "width")
        if any(row.bit_length() > width for row in vectors):
            raise ValueError("row exceeds declared width")
    basis: dict[int, int] = {}
    for row in vectors:
        while row:
            pivot = row.bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    return len(basis)


def reset_required_bits(n: int, addressable) -> int:
    """Retained bits for total parity and individually addressable resets.

    The initial n-bit source is available during compression. The empty word
    and allowed singleton resets are requested afterward. Their consequence rows
    span the all-ones row together with the addressable unit rows.
    """
    n = _int(n, "n", 1)
    if not isinstance(addressable, (tuple, list, set, frozenset)):
        raise ValueError("addressable must be a finite collection of coordinates")
    indices = [_int(i, "addressable coordinate") for i in addressable]
    if any(i >= n for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("addressable coordinates must be unique and lie in [0,n)")
    return min(n, len(indices) + 1)


def binary_entropy(p: float) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability must be finite and lie in [0,1]")
    if p in (0, 1):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def query_memory_lower_bound(n: int, error: float) -> float:
    """Necessary retained information for n fair bits queried after compression.

    Return n*(1-h2(error)) before rounding a fixed-length binary representation
    up to an integer. No finite-code achievability claim accompanies this bound.
    """
    n = _int(n, "n", 1)
    entropy = binary_entropy(error)
    if error > 0.5:
        raise ValueError("query error must lie in [0,0.5]")
    return n * (1 - entropy)
