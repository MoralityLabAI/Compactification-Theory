"""COE-06: finite cyclic-gauge observer codes and independent replay.

All metric checks are exact Hamming computations. Construction uses a pinned
14-bit, constant-weight-six codebook found by integer programming; the witness
is validated independently, without requiring an optimizer.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb
from typing import Sequence

WIDTH = 14
WEIGHT = 6
# Feasible orbit-representative words from a T=14, weight=6 integer program.
# Each pair of *orbits* is separated by at least four Hamming positions.
ORBIT_WORDS = (207, 243, 311, 473, 543, 629, 697, 811, 851, 903, 1133,
               1175, 1337, 1351, 1419, 1445, 1587, 1677, 1865, 2451, 2709)


def rotate(word: int, width: int = WIDTH) -> int:
    if width < 1 or not 0 <= word < (1 << width):
        raise ValueError("word must fit in a positive-width register")
    return (word >> 1) | ((word & 1) << (width - 1))


def orbit(word: int, width: int = WIDTH) -> tuple[int, ...]:
    result = set()
    current = word
    for _ in range(width):
        result.add(current)
        current = rotate(current, width)
    return tuple(sorted(result))


def necklace(word: int, width: int = WIDTH) -> int:
    return min(orbit(word, width))


def orbit_distance(left: int, right: int, width: int = WIDTH) -> int:
    """Min Hamming distance between two rotation orbits."""
    return min((left ^ shift).bit_count() for shift in orbit(right, width))


def trace(word: int, visits: int = WIDTH, width: int = WIDTH) -> int:
    """One tied transition and a fixed one-bit head over visits."""
    state = word
    history = 0
    for t in range(visits):
        history |= (state & 1) << t
        state = rotate(state, width)
    return history


def decode_under_unknown_phase(observed: int, words: Sequence[int] = ORBIT_WORDS,
                               width: int = WIDTH, allowed_flips: int = 1) -> int | None:
    """Nearest *orbit* decoding; resolves arbitrary cyclic phase plus flips.

    A candidate gets its closest cyclic rotation. If there is no unique label
    within the allowed radius, reject rather than assign.
    """
    if not 0 <= observed < (1 << width):
        raise ValueError("out-of-range readout")
    if allowed_flips < 0:
        raise ValueError("negative radius")
    distances = [orbit_distance(observed, word, width) for word in words]
    dmin = min(distances)
    if dmin > allowed_flips or distances.count(dmin) != 1:
        return None
    return distances.index(dmin)


def certificate(words: Sequence[int] = ORBIT_WORDS, width: int = WIDTH,
                weight: int = WEIGHT) -> dict:
    """Auditable finite witness, all distinct-label orbit pairs and corruptions."""
    words = tuple(words)
    if not words or len(set(words)) != len(words):
        raise ValueError("unique codebook required")
    if not all(0 <= w < (1 << width) and w.bit_count() == weight for w in words):
        raise ValueError("constant-weight words outside range")
    orbits = tuple(orbit(w, width) for w in words)
    if len(set(necklace(w, width) for w in words)) != len(words):
        raise ValueError("clock-invariant label collision")
    pair_d = tuple(orbit_distance(a, b, width) for a, b in combinations(words, 2))
    if pair_d and min(pair_d) < 4:
        raise ValueError("single-bit correction requires cross-orbit distance >= 3")
    checked = sum(1 for i, w in enumerate(words) for shifted in orbits[i]
                  for sample in (shifted,) + tuple(shifted ^ (1 << j) for j in range(width))
                  if decode_under_unknown_phase(sample, words, width) == i)
    total = sum(len(o) * (width + 1) for o in orbits)
    assert checked == total
    return {"schema": "COE-06-clock-invariant-single-flip-v1",
            "width": width, "weight": weight, "labels": len(words),
            "codebook": list(words), "distinct_necklaces": len(orbits),
            "orbit_sizes": sorted(Counter(map(len, orbits)).items()),
            "cross_orbit_distance_min": min(pair_d) if pair_d else None,
            "pairwise_orbit_comparisons": len(pair_d),
            "noiseless_and_single_flip_decoded": checked,
            "noiseless_and_single_flip_total": total,
            "count_only_pairwise_distinctions": 0,
            "ordered_orbit_pairwise_distinctions": len(pair_d)}


if __name__ == "__main__":
    import json
    print(json.dumps(certificate(), indent=2, sort_keys=True))
