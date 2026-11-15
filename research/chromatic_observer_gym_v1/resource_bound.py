"""Exact integer combinatorial upper bounds on phase-blind one-flip codes.

Two different labels whose cyclic orbits have minimum Hamming distance >=4
cannot share a rotated (w-1)-element sub-support. Each code orbit consumes
some number of (w-1)-necklace classes, which are disjoint across labels.
An integer count of the cheapest possible resource usage gives a *rigorous*
upper bound without an optimizer or SAT solver.
"""
from __future__ import annotations
from itertools import combinations
from math import comb
try:
    from .gauge_orbit_codes import necklace
except ImportError:
    from gauge_orbit_codes import necklace


def _subsets(T, w):
    for bits in combinations(range(T), w):
        yield sum(1 << b for b in bits)


def deletion_bound(T:int,w:int):
    if T < 1 or not 0 <= w <= T:raise ValueError('bad width/weight')
    if w==0:return {'T':T,'w':w,'total_resources':0,'usage_counts':{},'bound':1}
    reps=sorted(set(necklace(word,T) for word in _subsets(T,w)))
    subreps=set(necklace(word,T) for word in _subsets(T,w-1))
    resource_hist={}
    for word in reps:
        resource_count=len({necklace(word^(1<<j),T) for j in range(T)
                            if word & (1<<j)})
        resource_hist[resource_count]=resource_hist.get(resource_count,0)+1
    avail=len(subreps);used=0;count=0
    for cost,n in sorted(resource_hist.items()):
        if cost<=0:continue
        affordable=min(n,(avail-used)//cost)
        count+=affordable
        used+=affordable*cost
        if affordable<n:break
    return {'T':T,'w':w,'total_resources':avail,'usage_counts':resource_hist,'bound':count}


def all_small_bounds(max_T=13):
    full=[]
    for T in range(1,max_T+1):
        for w in range(T+1):
            if w> T//2:continue # complements yield identical bounds
            b=deletion_bound(T,w)
            # complement: superset (+1) necklace resource bound on original w
            bc=deletion_bound(T,T-w)
            b['complement_bound']=bc['bound']
            b['combined_bound']=min(b['bound'],bc['bound'])
            full.append(b)
    return full


if __name__=='__main__':
    import json
    for row in all_small_bounds(13):
        if row['T']>=8:
            print(row['T'],row['w'],row['total_resources'],row['usage_counts'],row['bound'],row['complement_bound'],row['combined_bound'])
    print('maximum 1..13:',max(r['combined_bound'] for r in all_small_bounds(13)))
