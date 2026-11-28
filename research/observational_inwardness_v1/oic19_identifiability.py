"""Exact finite observable-gauge classes of constant modular shifts.

For x_{t+1}=x_t xor 1 xor (u_t and [z_t=0]), z_{t+1}=z_t+d mod n,
with z0 uniform over Z_n, distinct shifts with the same gcd(n,d)
induce identical trace distributions for every fixed action word.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Iterable


def trace(n:int, d:int, z0:int, word:Iterable[int]) -> tuple[int,...]:
    if not isinstance(n,int) or n<2 or not isinstance(d,int) or not 0<=z0<n:
        raise ValueError('n>=2, integer d, valid z0 required')
    z=z0;x=0;states=[x]
    for u in word:
        if u not in (0,1):raise ValueError('binary action required')
        x ^= 1 ^ (int(u) & int(z==0))
        z=(z+d)%n
        states.append(x)
    return tuple(states)


def distributions(n:int,d:int,word:Iterable[int]) -> dict[tuple[int,...],Fraction]:
    word=tuple(word)
    counts=Counter(trace(n,d,z,word) for z in range(n))
    return {k:Fraction(v,n) for k,v in counts.items()}


def first_n_pulse_probability(n:int,d:int)->Fraction:
    if n<2:raise ValueError('n>=2')
    return Fraction(1,gcd(n,d))


def empirical_first_n_pulse_probability(n:int,d:int)->Fraction:
    num=0
    for z in range(n):
        tr=trace(n,d,z,(1,)*n)
        num+=int(any(tr[i]==tr[i+1] for i in range(n)))
    return Fraction(num,n)


def observable_class(n:int,d:int)->int:
    """With uniform z0, the unique invariant of constant-shift trace laws."""
    if n<2:raise ValueError('n>=2')
    return gcd(n,d)


def brute_gauge_audit(n:int, horizons=(0,1,2,3,4,6)):
    """Exhaustively test all action words and all shifts for modest horizons."""
    if n<2:raise ValueError('n>=2')
    errors=[];checked=0
    for T in horizons:
        for word in product((0,1),repeat=T):
            byshift={d:distributions(n,d,word) for d in range(n)}
            for a,b in product(range(n),repeat=2):
                if gcd(n,a)==gcd(n,b):
                    checked+=1
                    if byshift[a]!=byshift[b]:errors.append((T,word,a,b))
    return {'n':n,'checked_equivalent_shift_pairs':checked,
            'counterexamples':errors}


if __name__=='__main__':
    for n in range(2,13):
        assert all(first_n_pulse_probability(n,d)==empirical_first_n_pulse_probability(n,d) for d in range(n))
        report=brute_gauge_audit(n)
        assert not report['counterexamples']
        print(report)
