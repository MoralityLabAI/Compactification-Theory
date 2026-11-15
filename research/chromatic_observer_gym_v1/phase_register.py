"""COE-05: exact phase-sensitive coding with one tied recurrent readout.

No model training and no external dependencies. Eight physical recurrent states
are successive cyclic rotations of a binary register. The readout is always
bit zero; the initial register is one of 14 affine F_2^3 hyperplanes.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

WIDTH = 8
MASK = (1 << WIDTH) - 1


def dot_f2(a: int, x: int) -> int:
    return ((a & x).bit_count() & 1)


def affine_word(normal: int, side: int) -> int:
    """The 4-point affine hyperplane {x in F_2^3 : a.x == b}."""
    if normal not in range(1, 8) or side not in (0, 1):
        raise ValueError('normal must be nonzero in F2^3 and side must be a bit')
    return sum((1 << x) for x in range(WIDTH) if dot_f2(normal, x) == side)


def phase_codebook() -> tuple[int, ...]:
    return tuple(affine_word(a, b) for a in range(1, 8) for b in range(2))


def rotate_right(word: int, width: int = WIDTH) -> int:
    """One *time-independent* recurrent transition, no lookup by label."""
    if not 0 <= word < (1 << width) or width < 1:
        raise ValueError('word outside register alphabet')
    return ((word >> 1) | ((word & 1) << (width - 1)))


def shared_readout(word: int) -> int:
    return word & 1


def ordered_trace(initial_state: int, visits: int = WIDTH) -> int:
    state = initial_state
    transcript = 0
    for step in range(visits):
        transcript |= shared_readout(state) << step
        state = rotate_right(state)
    return transcript


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def decode_nearest(trace: int, book: tuple[int, ...], max_flips: int = 1) -> int | None:
    distances = tuple(hamming(trace, w) for w in book)
    least = min(distances)
    if least > max_flips or distances.count(least) != 1:
        return None
    return distances.index(least)


def rotation_orbit(word: int, width: int = WIDTH) -> frozenset[int]:
    seen = set()
    x = word
    for _ in range(width):
        seen.add(x)
        x = rotate_right(x, width)
    return frozenset(seen)


def orbit_id(word: int, width: int = WIDTH) -> int:
    return min(rotation_orbit(word, width))


def necklace_count(width: int, weight: int) -> int:
    """Burnside enumeration of constant-weight cyclic binary necklaces."""
    if not (0 <= weight <= width) or width < 1:
        raise ValueError('bad width/weight')
    from math import gcd
    fixed = 0
    for shift in range(width):
        cycles = gcd(width, shift)
        cycle_length = width // cycles
        if weight % cycle_length == 0:
            fixed += comb(cycles, weight // cycle_length)
    assert fixed % width == 0
    return fixed // width


def triple_packing_bound(width: int, weight: int) -> int:
    """Sharp pairwise-distance >=4 upper bound for constant weight.

    Each support (w-1)-subset belongs to at most one codeword.
    Complement symmetry gives an independent bound.
    """
    if not 0 <= weight <= width or width < 1:
        raise ValueError('invalid params')
    if weight == 0 or weight == width:
        return 1
    if weight == 1 or weight == width - 1:
        return 1
    return min(comb(width, weight - 1) // weight,
               comb(width, width - weight - 1) // (width - weight))


def evidence() -> dict:
    book = phase_codebook()
    assert len(book) == 14 and len(set(book)) == 14
    ds = [hamming(x, y) for x, y in combinations(book, 2)]
    weight_set = {w.bit_count() for w in book}
    assert weight_set == {4}
    assert min(ds) == 4 and max(ds) == 8
    assert all(ordered_trace(x) == x for x in book)
    error_hits = sum(decode_nearest(word ^ (1 << j), book) == i
                     for i, word in enumerate(book) for j in range(WIDTH))
    assert error_hits == 14 * WIDTH
    orbits = Counter(orbit_id(w) for w in book)
    # Without phase registration, uniform cyclic rotations yield identical
    # distributions for labels belonging to the same orbit.
    blind_bayes = len(orbits) / len(book)
    assert necklace_count(WIDTH, 4) == 10
    assert triple_packing_bound(8, 4) == 14
    max_width7 = max(triple_packing_bound(7, w) for w in range(8))
    assert max_width7 == 7
    return {
        'schema': 'COE-05-constant-weight-phase-code-v1',
        'dimension': 8,
        'labels': len(book),
        'codes_decimal': list(book),
        'hamming_weight': 4,
        'min_hamming_distance': min(ds),
        'max_hamming_distance': max(ds),
        'ordered_evaluator_edges': len(ds),
        'count_only_evaluator_edges': 0,
        'single_flip_correct': error_hits,
        'single_flip_total': len(book) * WIDTH,
        'packing_upper_bound_width8': triple_packing_bound(8,4),
        'max_packing_upper_bound_width7': max_width7,
        'all_weight4_necklaces_width8': necklace_count(8,4),
        'book_necklaces': len(orbits),
        'book_necklace_multiplicities': sorted(orbits.values()),
        'unknown_uniform_rotation_bayes_accuracy': blind_bayes,
    }


if __name__ == '__main__':
    import json
    print(json.dumps(evidence(), sort_keys=True, indent=2))
