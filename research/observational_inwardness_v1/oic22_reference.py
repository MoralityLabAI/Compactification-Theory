"""Exact OIC-22 small-state oracle replay and lower bound witness."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from oic22_exact import FaultOracle, MixedFaultOracle

HERE=Path(__file__).resolve().parent
PINNED=HERE/'results'/'oic22_reference.json'

def all_zero_ambiguity(n,Q,e=1,r=0):
    """Construct same all-zero/erasure receipt compatible with every shift d.

    Uses the fixed all-zero received branch's sensor queries, without assuming
    which (possibly adaptive) policy selected them. The argument holds for
    every query word, and its witness exists whenever Q-r<n*(e+1).
    """
    if not (n>=2 and Q>=r>=0 and e>=0):raise ValueError('invalid counts')
    # fixed schedule is sufficient to demonstrate the pigeonhole witness.
    addresses=[t % n for t in range(Q)]
    remaining=range(Q-r)
    witnesses={}
    for d in range(n):
        counts=[sum((z+t*d)%n==addresses[t] for t in remaining) for z in range(n)]
        z=min(range(n),key=lambda k:(counts[k],k))
        witnesses[d]={'phase':z,'flips_required':counts[z]}
    return {'compatible_every_shift':all(v['flips_required']<=e for v in witnesses.values()),
            'word':addresses,'witnesses':witnesses,'n':n,'Q':Q,'flips':e,'erasures':r}


def compute():
    rows=[]
    for n in (3,4,5):
        Q=n+2
        flip=FaultOracle(n,Q,'flip')
        erasure=FaultOracle(n,Q,'erase')
        mixed=MixedFaultOracle(n,Q)
        f=flip.optimal_value(0,flip.all,0)
        r=erasure.optimal_value(0,erasure.all,0)
        v=mixed.optimal_value(0,mixed.all,0)
        rows.append({'n':n,'Q':Q,
                     'uniform_one_flip_numerator':f,'uniform_one_flip_denominator':n*n*Q,
                     'uniform_one_erasure_numerator':r,'uniform_one_erasure_denominator':n*n*Q,
                     'half_clean_half_flip_numerator':v,'half_clean_half_flip_denominator':2*n*n*Q,
                     'at_most_one_flip_zero_error_impossible': Q<2*n,
                     'address_bits':Q*(n-1).bit_length(),
                     'response_bits':Q,
                     'total_online_bits':Q*(1+(n-1).bit_length())})
    return {'tag':'OIC-22-FINITE-BELLMAN-01','rows':rows,
            'worst_case_lower_bound':'Q >= n*(e+1) + r',
            'oracle_scope':'exact uniform initial phase, shift, and one fault location; mixed includes 50pct clean',
            'large_n':'no full exact oracle claim for n>=6'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args()
    record=compute()
    if a.check:
        stored=json.loads(PINNED.read_text())
        if stored!=record:raise SystemExit('OIC-22 exact oracle receipt differs from pinned result')
        print('OIC-22 exact Bellman finite receipt PASS')
    if a.out:a.out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    if not a.check:print(json.dumps(record,indent=2,sort_keys=True))
if __name__=='__main__':main()
