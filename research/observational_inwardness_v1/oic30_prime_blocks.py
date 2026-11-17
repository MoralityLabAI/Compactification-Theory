"""OIC-30: exact fixed-sensor cyclic codes and two-label obstruction.

All verdicts use integer arithmetic. This module constructs and checks
fixed sensor schedules. It makes no claim about minimum query count when
all n distinct sensor labels are available.
"""
from __future__ import annotations

from collections import Counter
from itertools import product


def is_prime(p: int) -> bool:
    return type(p) is int and p >= 2 and all(p % i for i in range(2, int(p**0.5)+1))


def prime_blocks(p: int, markers: tuple[int, ...]) -> tuple[int, ...]:
    """One constant marker per p-query block, all markers distinct."""
    if not is_prime(p) or p < 3:
        raise ValueError('odd prime size >=3 required')
    if not (2 <= len(markers) <= p) or len(set(markers)) != len(markers):
        raise ValueError('need distinct block markers, 2<=k<=p')
    if any(type(m) is not int or not 0 <= m < p for m in markers):
        raise ValueError('marker outside Z_p')
    return tuple(m for m in markers for _ in range(p))


def encoded(word: tuple[int, ...], p: int, d: int, z: int) -> int:
    return sum(int((z+t*d) % p == m) << t for t,m in enumerate(word))


def cross_shift_spectrum(word: tuple[int, ...], p: int) -> dict[int, int]:
    if not word or p < 2 or any(type(m) is not int or m < 0 or m >= p for m in word):
        raise ValueError('invalid p / schedule')
    codewords={(d,z):encoded(word,p,d,z) for d,z in product(range(p),repeat=2)}
    counts=Counter()
    for d in range(p):
        for e in range(d+1,p):
            for z,zz in product(range(p),repeat=2):
                counts[(codewords[d,z]^codewords[e,zz]).bit_count()]+=1
    if sum(counts.values()) != p*p*p*(p-1)//2:
        raise AssertionError('missing pair')
    return dict(sorted(counts.items()))


def check_all_flip_paths(word: tuple[int,...], p: int) -> dict[str,int]:
    """Enumerate every world, zero fault, and every single substitution."""
    known={}
    paths=0
    for d,z in product(range(p),repeat=2):
        clean=encoded(word,p,d,z)
        for t in range(-1,len(word)):
            recv=clean if t==-1 else clean^(1<<t)
            previous=known.setdefault(recv,d)
            if previous!=d:
                raise AssertionError(f'cross-shift collision at d={d}, previous={previous}')
            paths+=1
    return {'true_world_fault_paths':paths,'distinct_received_words':len(known)}


def sensor_counts_per_residue(word: tuple[int,...], p: int, labels: tuple[int,...]) -> tuple[tuple[int,...],...]:
    if len(word) % p:
        raise ValueError('whole number of phase periods required')
    return tuple(tuple(sum(word[t] == m for t in range(r,len(word),p)) for m in labels)
                 for r in range(p))


def reconstruct_counts(p: int, counts_zero: tuple[int,...]) -> tuple[int,...]:
    """Canonical two-address word with 3 samples for each time residue."""
    if not is_prime(p) or p < 3 or len(counts_zero)!=p or any(type(c) is not int or not 0<=c<=3 for c in counts_zero):
        raise ValueError('counts must be p integers in [0,3]')
    word=[-1]*(3*p)
    for r,c in enumerate(counts_zero):
        for j in range(3):
            word[r+j*p]=0 if j<c else 1
    return tuple(word)


def two_marker_obstruction(p: int, counts_zero: tuple[int,...]) -> dict:
    """Construct a cross-shift codeword pair with Hamming distance <=2.

    Works for every odd prime p>=3, every two-marker fixed 3p word,
    after affine normalization of labels to {0,1}.
    """
    w=reconstruct_counts(p,counts_zero)
    if len(set(counts_zero))>1:
        r=counts_zero.index(min(counts_zero))
        s=counts_zero.index(max(counts_zero))
        # Positive pulse for d at sensor 0 at r and sensor 1 at s.
        d=pow((s-r)%p,-1,p)
        z=(-r*d)%p
        a=(0,2)
        b=(d,z)
        reason='unequal_phase_residue_counts'
    else:
        c=counts_zero[0]
        repeated_marker=0 if c>=2 else 1
        a=(1,repeated_marker)
        b=(2,repeated_marker)
        reason='repeated_marker_intersection'
    left=encoded(w,p,*a);right=encoded(w,p,*b)
    distance=(left^right).bit_count()
    if a[0]==b[0] or distance>2:
        raise AssertionError(f'invalid obstruction for p={p},counts={counts_zero},distance={distance}')
    return {'counts_zero':list(counts_zero),'world_a':list(a),'world_b':list(b),'hamming_distance':distance,'reason':reason}


def count_two_marker_obstructions(p: int=7) -> dict:
    if not is_prime(p) or p>7:
        raise ValueError('finite enumeration registered only for primes 3,5,7')
    histogram=Counter()
    checked=0
    for counts in product(range(4),repeat=p):
        witness=two_marker_obstruction(p,counts)
        histogram[witness['hamming_distance']]+=1
        checked+=1
    return {'prime':p,'q':3*p,'normal_form_markers':[0,1],
            'count_vectors_checked':checked,
            'all_rejected_for_one_substitution':True,
            'analytic_witness_distance_histogram':{str(k):v for k,v in sorted(histogram.items())}}


def marker_restricted_lower_bound(n:int, errors:int) -> int:
    """Sharp necessary horizon for a fixed schedule omitting >=1 of n labels."""
    if type(n) is not int or n<2 or type(errors) is not int or errors<0:
        raise ValueError('n>=2, errors>=0')
    return n*(2*errors+1)


def verify_main_witness() -> dict:
    p=7
    w=prime_blocks(p,(0,1,2))
    hist=cross_shift_spectrum(w,p)
    if min(hist)!=3:raise AssertionError('wrong distance')
    fault=check_all_flip_paths(w,p)
    if fault['true_world_fault_paths']!=p*p*(len(w)+1):raise AssertionError('fault enumeration lost path')
    width=(p-1).bit_length(); symbol_width=(len(set(w))-1).bit_length()
    return {'schema':'OIC30-N7-FIXED-PRIME-BLOCKS-V1','n':p,'query_count':len(w),
            'sensor_markers':sorted(set(w)),'schedule':list(w),
            'cross_shift_pairs':sum(hist.values()),'minimum_cross_shift_hamming_distance':min(hist),
            'distance_spectrum':{str(k):v for k,v in hist.items()},**fault,
            'response_bits':len(w),'direct_address_bits_if_online':len(w)*width,
            'dynamic_online_link_bits':len(w)*(width+1),
            'offline_marker_config_bits':len(set(w))*width,
            'offline_fixed_schedule_bits':len(w)*symbol_width,
            'first_episode_bits_with_literal_offline_preprogramming':len(w)+len(set(w))*width+len(w)*symbol_width,
            'conditional_fixed_query_lower_bound_proper_marker_subset':marker_restricted_lower_bound(p,1),
            'two_marker_q21_exhaustive':count_two_marker_obstructions(7),
            'global_fixed_minimum_query_count':'unresolved; full seven-address Q20 search inconclusive',
            'adaptive_minimum_query_count':'interval [15,20] as verified by OIC-28',
            'proof_scope':'synthetic exact finite sensor model; no learned neural or consciousness claim'}


if __name__=='__main__':
    import json
    print(json.dumps(verify_main_witness(),indent=2,sort_keys=True))
