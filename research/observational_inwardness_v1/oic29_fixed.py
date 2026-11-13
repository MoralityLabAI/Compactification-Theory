"""OIC-29: independently checked fixed-sensor codes for seven hidden phases.

One address per time step; one bit returned; unknown (d,z0) in Z7^2.
One adversarial substitution may occur in any returned report. Both
witness words are explicit, and all calculations use Python integers.

This module certifies achievable upper bounds, not minimality.
"""
from __future__ import annotations
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

N = 7
WITNESSES = {
    'three_markers_q22': (0,1,2,2,1,2,0,2,0,0,1,2,1,1,1,2,1,0,0,0,2,1),
    'two_markers_q27': (0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1),
}


def oracle_bit(d: int, z: int, t: int, address: int, n: int=N) -> int:
    return int((z + t*d) % n == address)


def encoded_report(word: tuple[int, ...], d: int, z: int, n: int=N) -> int:
    return sum(oracle_bit(d,z,t,m,n) << t for t,m in enumerate(word))


def distance_spectrum(word: tuple[int, ...], n: int=N) -> dict[int,int]:
    if not word or any(type(m) is not int or not 0<=m<n for m in word):
        raise ValueError('invalid fixed sensor schedule')
    words={(d,z):encoded_report(word,d,z,n) for d,z in product(range(n),repeat=2)}
    hist=Counter()
    for d in range(n):
        for e in range(d+1,n):
            for z,zz in product(range(n),repeat=2):
                hist[(words[d,z]^words[e,zz]).bit_count()]+=1
    if sum(hist.values())!=n*n*n*(n-1)//2:
        raise AssertionError('cross-family enumeration lost a pair')
    return dict(sorted(hist.items()))


def all_flip_receipts(word: tuple[int, ...], n: int=N) -> dict[str,int]:
    """Exhaust every world with no fault or a flip at exactly one position.

    Reject if the same received word is compatible with two different shifts.
    """
    received: dict[int,int]={}
    paths=0
    for d,z in product(range(n),repeat=2):
        clean=encoded_report(word,d,z,n)
        for pos in range(-1,len(word)):
            observed=clean if pos<0 else (clean^(1<<pos))
            prior=received.setdefault(observed,d)
            if prior!=d:
                raise AssertionError(f'conflicting shifts for report {observed}: {prior} and {d}')
            paths+=1
    return {'true_world_fault_paths':paths,'distinct_observed_receipts':len(received)}


def verify_word(name: str, word: tuple[int, ...], n: int=N) -> dict:
    spectrum=distance_spectrum(word,n)
    if min(spectrum)<3:
        raise AssertionError('witness lacks error-correction distance')
    faults=all_flip_receipts(word,n)
    q=len(word)
    markers=sorted(set(word))
    width=(n-1).bit_length()
    symbol_width=(len(markers)-1).bit_length()
    return {
        'name':name, 'n':n,'q':q,'word':list(word),
        'markers':markers,'distinct_marker_count':len(markers),
        'cross_shift_pair_count':sum(spectrum.values()),
        'minimum_cross_shift_distance':min(spectrum),
        'distance_spectrum':{str(k):v for k,v in spectrum.items()},
        **faults,
        'online_response_bits':q,
        'online_address_bits_if_transmitted_each_query':q*width,
        'online_total_bits_with_dynamic_addressing':q*(width+1),
        'offline_marker_address_config_bits':len(markers)*width,
        'offline_preprogrammed_schedule_bits':q*symbol_width,
        'first_episode_total_with_offline_preprogramming':q+len(markers)*width+q*symbol_width,
        'repeat_episode_online_bits_after_preprogramming':q,
    }


def reference() -> dict:
    return {'schema':'OIC29-FIXED-N7-V1',
      'claim_scope':'constructive upper bounds only; no 21-query UNSAT certificate',
      'schedules':[verify_word(name,word) for name,word in WITNESSES.items()],
      'unresolved':['adaptive query horizons 15-19','fixed 21-query existence'],
      'solver_status':{'q15_affine_20m':'CAP','q19_affine_12m':'CAP','q19_affine_8m_orderings':'CAP',
                       'q21_integer_solver_20s':'TIME_LIMIT'}
    }


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--write',action='store_true')
    a=p.parse_args()
    target=Path(__file__).resolve().parent/'results/oic29_fixed_reference.json'
    current=reference()
    if a.write:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(current,sort_keys=True,indent=2)+'\n')
    if a.check:
        if json.loads(target.read_text())!=current:
            raise AssertionError('OIC-29 fixed receipt replay mismatch')
        print('OIC-29 fixed codebooks: exact independent replay passed')
    if not a.check and not a.write:
        print(json.dumps(current,indent=2,sort_keys=True))


if __name__=='__main__':main()
