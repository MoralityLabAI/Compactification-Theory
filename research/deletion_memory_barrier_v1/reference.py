"""DMB-01: exact finite deletion channel and streaming monitor oracle.

Pure standard library (Fraction). No claim to reproduce OpenAI Math #122
superpolynomial hard pairs or #140 high-dimensional Gaussian bounds.

Definitions count *ordinary deletion traces*: retained bits are in their
original order, positions/masks are not revealed, and output lengths
are observed. The monitor output uses only its final B-bit state.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, ceil, pi
from pathlib import Path


def words(n: int) -> tuple[str, ...]:
    if type(n) is not int or not 1 <= n <= 10:
        raise ValueError("word length must be 1..10")
    return tuple("".join(bits) for bits in itertools.product("01", repeat=n))


def rate(value) -> Fraction:
    q = Fraction(value)
    if not 0 <= q <= 1:
        raise ValueError("channel probability must lie in [0,1]")
    return q


def deletion_law(word: str, deletion: Fraction) -> dict[str, Fraction]:
    """Exact law of the trace after iid deletions of each input bit."""
    if not word or any(c not in "01" for c in word):
        raise ValueError("nonempty binary source word required")
    q = rate(deletion)
    p = 1 - q
    n = len(word)
    law = defaultdict(Fraction)
    for mask in range(1 << n):
        trace = "".join(word[i] for i in range(n) if mask & (1 << i))
        k = mask.bit_count()
        law[trace] += p ** k * q ** (n-k)
    if sum(law.values()) != 1:
        raise AssertionError("deletion trace law did not normalize")
    return dict(law)


def deletion_deck(word: str, length: int) -> Counter:
    """Occurrences of each length-k subsequence, counted with multiplicity."""
    n = len(word)
    if not 0 <= length <= n:
        raise ValueError("invalid deck length")
    return Counter("".join(word[i] for i in index)
                   for index in itertools.combinations(range(n), length))


def deck_equal_through(x: str, y: str, degree: int) -> bool:
    if len(x) != len(y) or degree < 0:
        raise ValueError("words must be same length and degree nonnegative")
    return all(deletion_deck(x, k) == deletion_deck(y, k)
               for k in range(min(degree, len(x)) + 1))


def total_variation(a: dict, b: dict) -> Fraction:
    if sum(a.values()) != 1 or sum(b.values()) != 1:
        raise ValueError("inputs must be probability distributions")
    return sum((abs(a.get(t, 0)-b.get(t, 0))
                for t in set(a) | set(b)), Fraction()) / 2


def retained_length_tail(n: int, deletion: Fraction, degree: int) -> Fraction:
    q = rate(deletion)
    p = 1 - q
    if not 0 <= degree <= n:
        raise ValueError("degree outside length")
    return sum((Fraction(comb(n,k)) * p**k * q**(n-k)
                for k in range(degree+1, n+1)), Fraction())


def product_law(law: dict[str,Fraction], count: int) -> dict[tuple[str,...],Fraction]:
    if not 0 <= count <= 4:
        raise ValueError("tensor count restricted to 0..4")
    output={():Fraction(1)}
    for _ in range(count):
        next_distribution=defaultdict(Fraction)
        for prev,mp in output.items():
            for trace,prob in law.items():
                next_distribution[prev+(trace,)]+=mp*prob
        output=dict(next_distribution)
    return output


def hard_pair(n: int, deletion: Fraction) -> dict:
    law={w:deletion_law(w,deletion) for w in words(n)}
    pair=None
    best=Fraction(2)
    for x,y in itertools.combinations(law,2):
        tv=total_variation(law[x],law[y])
        if tv<best:
            best=tv
            pair=(x,y)
    if pair is None:
        raise AssertionError("no distinct binary source words")
    return {"x":pair[0],"y":pair[1],"one_trace_tv":str(best)}


def trace_code(trace: str) -> int:
    # Prefix-free *integer representation* of a binary word including length.
    return (1 << len(trace)) + (int(trace, 2) if trace else 0)


def update_state(state: int, trace: str, bits: int) -> int:
    if type(bits) is not int or bits < 1:
        raise ValueError("bits must be positive")
    return (3*state + trace_code(trace) + 1) % (1 << bits)


def streaming_state_laws(n: int, deletion: Fraction, count: int,
                         memory_bits: int) -> dict[str,dict[int,Fraction]]:
    """Exact distribution of a B-bit deterministic update-state monitor."""
    if type(count) is not int or count < 0 or count > 8:
        raise ValueError("count must be 0..8")
    if type(memory_bits) is not int or not 1 <= memory_bits <= 12:
        raise ValueError("memory bits must be 1..12")
    out={}
    for x in words(n):
        law=deletion_law(x,deletion)
        state_probs={0:Fraction(1)}
        for _ in range(count):
            updated=defaultdict(Fraction)
            for s,mass in state_probs.items():
                for trace,prob in law.items():
                    updated[update_state(s,trace,memory_bits)]+=mass*prob
            state_probs=dict(updated)
        if sum(state_probs.values())!=1 or any(
                not 0<=s<(1<<memory_bits) for s in state_probs):
            raise AssertionError("terminal state escaped B-bit alphabet")
        out[x]=state_probs
    return out


def optimal_terminal_success(laws: dict[str,dict[int,Fraction]]) -> Fraction:
    """MAP success for uniform input X from ONLY the final state."""
    if not laws:
        raise ValueError("missing source words")
    states=set().union(*(set(v) for v in laws.values()))
    return sum((max(probs.get(s,Fraction(0)) for probs in laws.values())
                for s in states), Fraction()) / len(laws)


def optimal_full_trace_success(n: int, deletion: Fraction,
                               count: int) -> Fraction:
    """Bayes exact reconstruction using all complete deletion traces."""
    if not 0<=count<=3 or n>5:
        raise ValueError("full exact decoder restricted to n<=5, m<=3")
    laws={x:product_law(deletion_law(x,deletion),count) for x in words(n)}
    observations=set().union(*(set(v) for v in laws.values()))
    return sum((max(p.get(obs,Fraction(0)) for p in laws.values())
                for obs in observations),Fraction()) / (1<<n)


def terminal_state_ceiling(n: int, B: int, m: int, clock: bool=False) -> Fraction:
    """Universal exact-success ceiling, with optional public stop index."""
    if n<1 or B<0 or m<0 or any(type(v) is not int for v in (n,B,m)):
        raise ValueError("n positive, B and m nonnegative integers")
    alphabet=(1<<B)*(m+1 if clock else 1)
    return min(Fraction(1), Fraction(alphabet,1<<n))


def hard_pair_success_ceiling(tv_single: Fraction, count: int) -> Fraction:
    delta=rate(tv_single)
    if type(count) is not int or count<0:
        raise ValueError("nonnegative trace count")
    # Coupling of m iid trace pairs yields tensor TV <= 1-(1-tv)^m.
    return (2-(1-delta)**count)/2


def universal_information_ceiling(n: int, retained: Fraction, m: int,
                                   B: int, clock: bool=False) -> dict:
    """Fano/data-processing necessary info lower bound for uniform X.

    I(X;terminal) <= min(B+log2(m+1), m*n*retained) for a
    variable stop index, or <=min(B,m*n*retained) otherwise.
    This returns the exact *retained-bit volume* term, and
    its finite alphabet cap, omitting transcendental logarithms.
    """
    r=rate(retained)
    if any(type(v) is not int for v in (n,m,B)) or n<1 or m<0 or B<0:
        raise ValueError("invalid n,m,B")
    return {"expected_retained_trace_bits":str(m*n*r),
            "terminal_alphabet":(1<<B)*(m+1 if clock else 1),
            "source_entropy_bits":n}


def gaussian_cap_upper(d: int, epsilon: float) -> float:
    """Surface cap measure <=2/pi * epsilon^(d-1), epsilon in (0,pi/2]."""
    if type(d) is not int or d<2 or not 0<epsilon<=pi/2:
        raise ValueError("require d>=2, angle eps in (0,pi/2]")
    return (2/pi)*epsilon**(d-1)


def gaussian_terminal_ceiling(d: int, epsilon: float,
                              B: int, m: int=0, clock: bool=False)->float:
    """Prior-uniform sphere angular recovery ceiling for B terminal bits."""
    if type(B) is not int or B<0 or type(m) is not int or m<0:
        raise ValueError("invalid state bits or index count")
    return min(1.0, (1<<B)*(m+1 if clock else 1)*gaussian_cap_upper(d,epsilon))


def binomial_tail(m: int, retention: Fraction, k: int)->Fraction:
    """Exact Pr[Binomial(m,retention)>k]."""
    r=rate(retention)
    if type(m) is not int or type(k) is not int or m<0 or k<0:
        raise ValueError("nonnegative m,k required")
    return sum((Fraction(comb(m,j))*r**j*(1-r)**(m-j)
                for j in range(k+1,m+1)), Fraction())


def thinning_markov_cap(m: int, retention: Fraction) -> int:
    """K=ceil(6*m*r) suffices for probability of erasure overrun <=1/6."""
    r=rate(retention)
    if type(m) is not int or m<0:
        raise ValueError("m nonnegative")
    count=6*m*r
    return (count.numerator+count.denominator-1)//count.denominator




def structured_two_word_success(n: int, m: int, deletion: Fraction) -> Fraction:
    """Constructive 1-bit monitor on prior {0^n,1^n}, each prob 1/2.

    A retained 1 immediately sets terminal bit to 1; absent evidence
    outputs 0. All-zero source is always correct. All-one source fails
    precisely when every bit across m traces is deleted.
    """
    q=rate(deletion)
    if type(n) is not int or n<1 or type(m) is not int or m<0:
        raise ValueError("positive n, nonnegative trace count")
    return 1-Fraction(1,2)*q**(n*m)


def control_report() -> dict:
    n=4
    q=Fraction(4,5)
    x,y="0110","1001"
    assert deck_equal_through(x,y,2)
    lx,ly=deletion_law(x,q),deletion_law(y,q)
    tv=total_variation(lx,ly)
    tail=retained_length_tail(n,q,2)
    assert tv<=tail and tv>0
    source_laws={str(B):{} for B in (1,2,3,4,5)}
    for B in (1,2,3,4,5):
        for m in (0,1,2,3):
            prob=optimal_terminal_success(streaming_state_laws(n,q,m,B))
            assert prob<=terminal_state_ceiling(n,B,m)
            source_laws[str(B)][str(m)]=str(prob)
    pair_tvs=[]
    for m in (1,2,3):
        observed=total_variation(product_law(lx,m),product_law(ly,m))
        assert observed<=1-(1-tv)**m
        pair_tvs.append({"m":m,"tv":str(observed),
                         "coupling_upper":str(1-(1-tv)**m),
                         "pair_success_ceiling":str(hard_pair_success_ceiling(tv,m))})
    full=[{"m":m,"optimal_full_trace_success":
           str(optimal_full_trace_success(n,q,m))} for m in (0,1,2)]
    thinnings=[]
    for offered in (1,5,10,25,50):
        k=thinning_markov_cap(offered,Fraction(1,5))
        p=binomial_tail(offered,Fraction(1,5),k)
        assert p<=Fraction(1,6)
        thinnings.append({"offered":offered,"retained_budget":k,
                          "overrun_probability":str(p)})
    return {
        "status":"EXACT_FINITE_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "word_length":n,
        "deletion_probability":str(q),
        "deck_witness":{"x":x,"y":y,"equal_through":2,
                        "single_trace_tv":str(tv),"length_tail":str(tail)},
        "hard_pair_over_sample_count":pair_tvs,
        "best_unbounded_terminal_from_full_traces":full,
        "actual_bounded_state_success":source_laws,
        "thinning_overruns":thinnings,
        "structured_prior_one_bit_example_n8_m1":str(structured_two_word_success(8,1,q)),
        "source_122_superpolynomial_hard_pair_reproduced":False,
        "source_140_gaussian_lower_bound_reproduced":False,
        "trained_neural_checkpoint_evaluated":False,
    }


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--out",type=Path)
    a=ap.parse_args(argv)
    result=control_report()
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
