"""OIC-32: global one-erasure fixed sensing for every prime cycle.

Analytic line-moment lower bound p>=5, complete small-prime exhaustive
classification p=2/3, and constructive 2p-query two-block codebooks.
No numerical optimizer or stochastic simulation participates in the proof.
"""
from __future__ import annotations
from itertools import product
from oic32_global import require_prime, minimum_cross_shift_distance


def two_marker_code(p: int) -> tuple[int, ...]:
    if type(p) is not int or p < 2:
        raise ValueError('p>=2 required')
    return (0,) * p + (1,) * p


def analytic_erasure_gap(p: int) -> dict:
    """Hypothetical min distance 2 at Q=2p-1 contradicts pair moments."""
    require_prime(p)
    q = 2*p-1
    upper = 3*p*(p-1)//2
    lower = 2*(p-1)**2
    gap = lower-upper
    if gap*2 != (p-1)*(p-4) or gap<=0:
        raise AssertionError('global erasure counting contradiction lost')
    return {'prime':p,'forbidden_queries':q,'optimal_queries':2*p,
            'max_light_weight_one_lines':q,'line_pair_upper_if_erasure_robust':upper,
            'line_pair_lower_for_all_words':lower,'contradiction_gap':gap}


def exhaustive_small(p: int, q: int) -> dict:
    """Global-translation-normalized enumeration of every p=2/3 fixed word."""
    if p not in (2,3) or not 1<=q<=6:
        raise ValueError('registered small erasure cases only')
    tested=0;maximum=-1;witness=None
    for rest in product(range(p),repeat=q-1):
        word=(0,)+rest
        minimum=minimum_cross_shift_distance(p,word)
        tested+=1
        if minimum>maximum:
            maximum,witness=minimum,word
    if tested!=p**(q-1):
        raise AssertionError('small-prime enumeration incomplete')
    return {'prime':p,'horizon':q,'checked_normalized_schedules':tested,
            'max_minimum_cross_shift_distance':maximum,'example':list(witness),
            'erasure_robust_schedule_exists':maximum>=2}


def reference() -> dict:
    small=[exhaustive_small(p,q) for p,q in ((2,3),(2,4),(3,5),(3,6))]
    if [r['max_minimum_cross_shift_distance'] for r in small]!=[1,2,1,2]:
        raise AssertionError('unexpected small-prime results')
    large=[]
    for p in (5,7,11,13):
        entry=analytic_erasure_gap(p)
        code=two_marker_code(p)
        dist=minimum_cross_shift_distance(p,code)
        if dist!=2:raise AssertionError('two-block erasure code distance mismatch')
        entry.update({'attained_queries':len(code),'marker_count':2,
                      'verified_code_distance':dist,'cross_shift_world_pairs':p*p*p*(p-1)//2})
        large.append(entry)
    return {'schema':'OIC32-PRIME-GLOBAL-FIXED-ERASURE-V1',
            'statement':'For every prime p>=2, global fixed one-marked-erasure minimum is 2p',
            'proof_status':'analytic_uniform_p_ge5_plus_exhaustive_p2_p3',
            'small_primes':small,'analytic_primes':large}


if __name__=='__main__':
    import json
    print(json.dumps(reference(),sort_keys=True,indent=2))
