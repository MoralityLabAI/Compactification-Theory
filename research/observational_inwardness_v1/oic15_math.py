"""OIC-T18–T21: elementary finite-state observability and sensor-likelihood bounds.

The statements are finite and interface-conditional. The numeric reference uses
standard-library arithmetic only; no claim about consciousness or actual neural
network hidden modes follows from the synthetic construction.
"""
from __future__ import annotations
import math


def residual_phase_entropy(n:int)->float:
    """Bits of phase entropy in fixed-family, z != 0, under uniform prior."""
    if not isinstance(n,int) or n<2:
        raise ValueError('n must be integer >=2')
    return math.log2(n-1)


def rotating_pulse_likelihoods(n:int,beta:float)->tuple[float,float]:
    """Event-channel likelihood sums for an observed one-hot length-n audit word.

    Family fixed has n-1 all-zero phase candidates and one all-one candidate.
    Rotating family has one perfect match and n-1 one-hot words at distance 2.
    q=exp(-beta) is the per-mismatch likelihood ratio.
    """
    if n<2 or beta<=0:
        raise ValueError('n>=2 and positive beta required')
    q=math.exp(-beta)
    likelihood_fixed=(n-1)*q+q**(n-1)
    likelihood_rotating=1+(n-1)*q*q
    return (likelihood_fixed,likelihood_rotating)


def rotating_pulse_classified(n:int,beta:float)->bool:
    a,b=rotating_pulse_likelihoods(n,beta)
    return b>a


def first_soft_collapse_n(beta:float)->int:
    """Smallest n for which finite audit-likelihood sum loses one-hot separation."""
    if beta<=0:
        raise ValueError('positive beta required')
    lo,hi=2,2
    while rotating_pulse_classified(hi,beta):
        hi*=2
    lo=hi//2
    while lo+1<hi:
        mid=(lo+hi)//2
        if rotating_pulse_classified(mid,beta):lo=mid
        else:hi=mid
    return hi


def raw_filter_sufficient_beta(n:int,minimum_cross_distance:int,
                               flips:int=0,erasures:int=0)->float:
    """If beta > returned threshold, raw-likelihood classification is guaranteed.

    Uses the worst-case sum over at most n wrong family phase hypotheses.
    E= d_min - 2e - r must be positive; this is stricter than nearest-codeword
    identifiability due to marginalization over unknown phase.
    """
    gap=minimum_cross_distance-2*flips-erasures
    if n<2 or flips<0 or erasures<0 or gap<=0:
        raise ValueError('positive guaranteed cross-family margin required')
    return math.log(n)/gap


def raw_codebook(n:int, family:int, word:tuple[int,...])->tuple[tuple[int,...],...]:
    if n<2 or family not in (0,1) or any(u not in (0,1) for u in word):
        raise ValueError('invalid finite system')
    result=[]
    for initial_phase in range(n):
        x,z=0,initial_phase
        trace=[x]
        for action in word:
            x ^=1 ^ (action & int(z==0))
            z=(z+family)%n
            trace.append(x)
        result.append(tuple(trace))
    return tuple(result)


def codebook_min_distance(n:int,word:tuple[int,...])->int:
    a=raw_codebook(n,0,word);b=raw_codebook(n,1,word)
    return min(sum(v!=w for v,w in zip(x,y)) for x in a for y in b)


def codebook_min_error_correct(n:int,word:tuple[int,...],flips:int,erasures:int)->bool:
    return codebook_min_distance(n,word)>2*flips+erasures


def finite_exact_bayes_accuracy(n:int,t:int)->float:
    if n<2 or t<0:
        raise ValueError('n>=2, t>=0 required')
    return 1.0 if t>=n else .5 if t<=1 else 1-(n-t)/(2*n)
