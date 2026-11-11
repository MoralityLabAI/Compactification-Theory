"""OIC-32: exact arithmetic audit for the global prime-cycle fixed query theorem.

The handwritten proof in OIC32_THEORY.md is uniform over every prime p>=5.
This independent Python reference checks its counting identities, inequality,
and the attaining 3p-query block code on small prime test instances.
No MILP infeasibility verdict is used by the theorem or its tests.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations


def is_prime(value: int) -> bool:
    if type(value) is not int or value < 2:
        return False
    return all(value % k for k in range(2, int(value ** .5) + 1))


def require_prime(p: int) -> None:
    if not is_prime(p) or p < 5:
        raise ValueError('the global theorem requires prime p>=5')


def residue_occupancy(p: int, word: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if type(p) is not int or p < 2 or any(type(m) is not int or m not in range(p) for m in word):
        raise ValueError('invalid phase count / sensor address')
    counts = [[0] * p for _ in range(p)]
    for t, marker in enumerate(word):
        counts[t % p][marker] += 1
    return tuple(tuple(row) for row in counts)


def line_weights(p: int, word: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Line m=z+rd in the affine plane, with d the unknown shift."""
    occ = residue_occupancy(p, word)
    return tuple(tuple(sum(occ[r][(z + r * d) % p] for r in range(p))
                       for z in range(p)) for d in range(p))


def choose2(n: int) -> int:
    return n * (n-1) // 2


def pair_moment_identity(p: int, word: tuple[int, ...]) -> dict[str, int]:
    """Verify line-pair moments including repeated phase/residue measurements.

    Every pair of queries in different time residues belongs to one
    nonvertical affine line over a prime field. Pairs at the SAME affine
    point belong to all p lines through that point; within-row pairs at
    distinct markers belong to no such line.
    """
    require_prime(p)
    occ = residue_occupancy(p, word)
    weights = line_weights(p, word)
    rows = tuple(sum(row) for row in occ)
    different_rows = sum(rows[r] * rows[s] for r in range(p) for s in range(r + 1, p))
    same_point = sum(choose2(v) for row in occ for v in row)
    lhs = sum(choose2(w) for direction in weights for w in direction)
    rhs = different_rows + p * same_point
    if lhs != rhs:
        raise AssertionError(f'affine-plane query-pair identity fails: {lhs} != {rhs}')
    return {'line_pair_moment': lhs, 'different_residue_pairs': different_rows,
            'same_point_pairs': same_point, 'identity_rhs': rhs}


def line_report(p: int, word: tuple[int, ...], shift: int, phase: int) -> int:
    if not 0 <= shift < p or not 0 <= phase < p:
        raise ValueError('invalid hidden world')
    return sum(1 << t for t, m in enumerate(word) if (phase + t * shift) % p == m)


def minimum_cross_shift_distance(p: int, word: tuple[int, ...]) -> int:
    reports = [[line_report(p, word, d, z) for z in range(p)] for d in range(p)]
    return min((a ^ b).bit_count()
               for d in range(p) for d2 in range(d + 1, p)
               for a in reports[d] for b in reports[d2])


def prime_block_word(p: int) -> tuple[int, ...]:
    require_prime(p)
    return tuple(m for m in (0, 1, 2) for _ in range(p))


def theorem_bounds(p: int) -> dict[str, int]:
    """Audit the strict contradiction at Q=3p-1.

    Assumed min cross-shift distance >=3 forces all line weights >=2
    (see the analytical proof), with a2 weight-two lines and excess
    E=a2-p <=(p-1)//2. This gives an *upper* line-pair moment U.
    All Q=3p-1 schedules have *lower* moment L from distinct residues.
    """
    require_prime(p)
    q = 3 * p - 1
    max_light_lines = q // 2
    max_excess = max_light_lines - p
    if max_excess != (p - 1) // 2:
        raise AssertionError('excess accounting mismatch')
    upper = 3 * p * p - 2 * p + choose2(max_excess + 1)
    residue_sizes = [3] * (p - 1) + [2]
    lower = (q * q - sum(v * v for v in residue_sizes)) // 2
    strict_gap = lower - upper
    if strict_gap * 8 != 11 * p * p - 44 * p + 25 or strict_gap <= 0:
        raise AssertionError('theorem contradiction no longer strict')
    return {'prime': p, 'forbidden_query_horizon': q,
            'minimum_query_horizon': q + 1, 'max_weight_two_lines': max_light_lines,
            'max_heavy_excess': max_excess, 'line_pairs_upper_if_robust': upper,
            'line_pairs_lower_all_words': lower, 'integer_contradiction_gap': strict_gap}


def reference(primes: tuple[int, ...] = (5, 7, 11, 13)) -> dict:
    rows = []
    for p in primes:
        row = theorem_bounds(p)
        word = prime_block_word(p)
        attain = minimum_cross_shift_distance(p, word)
        if attain != 3:
            raise AssertionError('three-block construction is not robust')
        if pair_moment_identity(p, word)['line_pair_moment'] < 0:
            raise AssertionError('invalid pair-moment audit')
        row.update({'attaining_marker_count': 3, 'attaining_word_length': len(word),
                    'attaining_minimum_cross_shift_distance': attain,
                    'cross_shift_pairs_checked': choose2(p) * p * p})
        rows.append(row)
    return {'schema': 'OIC32-PRIME-GLOBAL-FIXED-FLIP-V1',
            'statement': 'For every prime p>=5, global fixed one-substitution minimum Q=3p',
            'proof_status': 'uniform_analytic_proof_with_finite_integer_checks',
            'excludes_full_marker_exception': True, 'rows': rows}


if __name__ == '__main__':
    import json
    print(json.dumps(reference(), indent=2, sort_keys=True))
