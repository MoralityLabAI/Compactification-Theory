"""OIC-32 exceptional small prime fixed one-flip exhaustive reference.

Enumerates *every* fixed address schedule at given Q modulo global phase
translation, which can normalize the first query to address zero. Distances
are recomputed directly from the true trajectories, independently of MILP.
"""
from __future__ import annotations
from itertools import product
from oic32_global import minimum_cross_shift_distance


def exhaustive_small_case(p: int, horizon: int) -> dict:
    if p not in (2, 3) or not 1 <= horizon <= 8:
        raise ValueError('registered exhaustive small-prime cases only')
    best = -1
    witness = None
    tested = 0
    for rest in product(range(p), repeat=horizon-1):
        word = (0,) + rest
        distance = minimum_cross_shift_distance(p, word)
        tested += 1
        if distance > best:
            best, witness = distance, word
    expected_count = p ** (horizon - 1)
    if tested != expected_count or witness is None:
        raise AssertionError('incomplete enumeration')
    return {'prime':p,'horizon':horizon,'normalized_schedules_checked':tested,
            'maximum_minimum_cross_shift_distance':best,'first_optimal_word':list(witness),
            'guarantees_one_substitution':best>=3}


def reference() -> dict:
    cases = [exhaustive_small_case(p,horizon) for p,horizon in
             ((2,5),(2,6),(3,7),(3,8))]
    if [(x['prime'],x['horizon'],x['maximum_minimum_cross_shift_distance'])
        for x in cases] != [(2,5,2),(2,6,3),(3,7,2),(3,8,3)]:
        raise AssertionError('registered exceptional reference changed')
    return {'schema':'OIC32-SMALL-PRIME-EXHAUSTIVE-V1',
            'statement':'fixed one-substitution minima p=2:Q6, p=3:Q8',
            'normalization':'global sensor-label translation sends first marker to 0',
            'status':'complete_small_finite_exhaustive_enumeration',
            'cases':cases}

if __name__ == '__main__':
    import json
    print(json.dumps(reference(),indent=2,sort_keys=True))
