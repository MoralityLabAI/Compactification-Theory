"""OIC-31 exact conditional fixed-schedule robust sensing with mixed faults.

Extends OIC-T60 (substitution-only) to e bit substitutions plus r
marked erasures, and proves sharpness for prime p when 2<=2e+r+1<p.
All calculations are pure Python integers and explicit finite states.
"""
from __future__ import annotations
from itertools import product
from collections import Counter
from oic30_prime_blocks import prime_blocks, encoded, cross_shift_spectrum


def require_params(n:int,e:int,r:int)->None:
    if not all(type(v) is int for v in (n,e,r)) or n < 2 or e<0 or r<0:
        raise ValueError('require n >= 2 and nonnegative integer e, r')


def omission_lower_bound(n:int,e:int,r:int)->int:
    """Necessary horizon for any fixed schedule omitting one of n markers."""
    require_params(n,e,r)
    return n*(2*e+r+1)


def prime_block_schedule(p:int,e:int,r:int)->tuple[int,...]:
    """Attain the bound when k=2e+r+1 satisfies 2<=k<p prime."""
    require_params(p,e,r)
    k=2*e+r+1
    if not 2<=k<p:
        raise ValueError('sharp proper-marker construction requires 2<=2e+r+1<p')
    return prime_blocks(p,tuple(range(k)))


def cross_family_minimum(word:tuple[int,...],n:int)->int:
    return min(cross_shift_spectrum(word,n))


def count_rotating_reports(word:tuple[int,...],n:int,d:int=1)->tuple[int,...]:
    """Each query is positive in precisely one initial phase for fixed d."""
    return tuple(sum((z+t*d)%n==m for t,m in enumerate(word)) for z in range(n))


def stationary_silent_marker(word:tuple[int,...],n:int)->int:
    for m in range(n):
        if m not in word:return m
    raise ValueError('schedule uses all marker labels')


def lower_bound_witness(word:tuple[int,...],n:int,e:int,r:int):
    """Find a stationary/rotating pair whose Hamming balls can overlap.

    Returns None only when the established lower-bound hypothesis Q<n(2e+r+1)
    is violated or when the schedule uses all n markers.
    """
    require_params(n,e,r)
    if len(word)>=omission_lower_bound(n,e,r):return None
    if any(type(m) is not int or not 0<=m<n for m in word):raise ValueError('invalid word')
    try: silent=stationary_silent_marker(word,n)
    except ValueError:return None
    counts=count_rotating_reports(word,n,1)
    z=min(range(n),key=lambda zz:counts[zz])
    a=encoded(word,n,0,silent)
    b=encoded(word,n,1,z)
    dist=(a^b).bit_count()
    if a!=0 or dist!=counts[z] or dist>2*e+r or dist>len(word)//n:
        raise AssertionError('broken omission/pigeonhole witness')
    return {'stationary_phase':silent,'rotating_phase':z,'distance':dist,
            'ones_per_rotating_phase':list(counts), 'rotating_world_shift':1}


def hamming_tolerant_collision(a:int,b:int,Q:int,e:int,r:int):
    """Construct a shared report with <=e flips and <=r erasures per world.

    Represent erased positions in a tuple as None. The construction applies
    only if the two known clean codewords differ in at most 2e+r places.
    """
    if (a^b).bit_count()>2*e+r:return None
    mismatch=[t for t in range(Q) if ((a^b)>>t)&1]
    erased=mismatch[:r]
    rest=mismatch[len(erased):]
    if len(rest)>2*e:raise AssertionError('invalid corruption bound')
    obs=[(a>>t)&1 for t in range(Q)]
    for t in erased:obs[t]=None
    # flip first max(0, len(rest)-e) positions away from a, leaving <= e away from b
    for t in rest[:max(0,len(rest)-e)]:obs[t]=(b>>t)&1
    def fault_count(code):
        return (sum(ob is not None and ob!=((code>>t)&1) for t,ob in enumerate(obs)),
                sum(ob is None for ob in obs))
    if fault_count(a)[0]>e or fault_count(b)[0]>e or fault_count(a)[1]>r:
        raise AssertionError('common report exceeds error budget')
    return tuple(obs)


def check_prime_attainment(p:int,e:int,r:int):
    word=prime_block_schedule(p,e,r)
    k=2*e+r+1
    dist=cross_family_minimum(word,p)
    if dist!=k:raise AssertionError('prime-block minimum-distance mismatch')
    if len(word)!=omission_lower_bound(p,e,r):raise AssertionError('wrong length')
    if len(set(word))!=k or k>=p:raise AssertionError('missing-label assumption violated')
    if dist<=2*e+r:raise AssertionError('correctability bound violated')
    return {'p':p,'substitutions':e,'marked_erasures':r,'markers':list(range(k)),
            'query_horizon':len(word),'minimum_distance':dist,'pairs_checked':p*p*p*(p-1)//2,
            'correction_threshold':2*e+r+1}


def reference():
    configs=((5,0,1),(5,0,2),(5,1,0),(5,1,1),
             (7,1,0),(7,1,1),(7,2,0),(11,1,1))
    return {'schema':'OIC31-MIXED-FAULT-PROPER-MARKER-V1',
            'attainable_prime_cases':[check_prime_attainment(*cfg) for cfg in configs],
            'statement':'Sharp n(2e+r+1) for fixed proper-marker policies if n=p prime, 2<=2e+r+1<p',
            'unrestricted_Q20_n7_status':'Open certificate; solver-reported UNSAT only'}

if __name__=='__main__':
    import json
    print(json.dumps(reference(),sort_keys=True,indent=2))
