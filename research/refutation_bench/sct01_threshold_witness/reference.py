"""SCT-01: exact SAT cores, expectation-cover costs, hitting times, DFA memory.

All finite formula probabilities and moments use Python Fraction. The model
has r disjoint blocks of three variables. A complete block's eight signed
3-clauses forbid every assignment to that block and are minimally UNSAT.
For r=2, all 65,536 subformulas are enumerated, with a separate full
assignment bitset SAT oracle.

OpenAI Math #175 covers general expectation thresholds; #235 covers the
full uniform random k-SAT distribution on growing n. Our disjoint-gadget
distribution is a restricted finite example, not the source limit theorem.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from math import comb, factorial, ceil, log2, sqrt
import hashlib
import json
import random
from pathlib import Path

K = 3
GADGET_CLAUSES = 1 << K
R = 2
M = R * GADGET_CLAUSES
ALL_CLAUSES = (1 << M) - 1


def valid(r: int, k: int) -> tuple[int,int]:
    if type(r) is not int or type(k) is not int or not 1 <= r <= 32 or not 1 <= k <= 16:
        raise ValueError("1<=gadgets<=32, 1<=clauses/gadget<=16 required")
    return r,k


def gadget_full_mask(g: int, k: int=GADGET_CLAUSES) -> int:
    if type(g) is not int or g<0 or type(k) is not int or k<1:
        raise ValueError("nonnegative gadget and positive clause count")
    return ((1<<k)-1) << (g*k)


def core_masks(r: int=R, k: int=GADGET_CLAUSES) -> tuple[int,...]:
    valid(r,k)
    return tuple(gadget_full_mask(g,k) for g in range(r))


def unsat(mask: int, r: int=R, k: int=GADGET_CLAUSES) -> bool:
    valid(r,k)
    if type(mask) is not int or not 0<=mask<(1<<(r*k)):
        raise ValueError("clause-subset bitmask outside the source universe")
    return any((mask & core)==core for core in core_masks(r,k))


def clause_false_assignment_bits(g: int, pattern: int, r: int=R) -> int:
    if type(g) is not int or type(pattern) is not int or not 0<=g<r or not 0<=pattern<8:
        raise ValueError("each clause forbids one assignment to three fixed variables")
    bits=0
    for assignment in range(1 << (3*r)):
        if (assignment>>(3*g))&7 == pattern:
            bits |= 1<<assignment
    return bits


def exhaustive_sat_oracle() -> dict:
    """Build all 2^16 SAT states using independent 64-assignment bitsets."""
    forbidden=[clause_false_assignment_bits(g,t) for g in range(R)
               for t in range(GADGET_CLAUSES)]
    remaining=[0]*(1<<M)
    remaining[0]=(1<<(1<<(K*R)))-1
    winners=[False]*(1<<M)
    winning_by_cardinality=[0]*(M+1)
    for mask in range(1,1<<M):
        lsbit=mask&-mask
        i=lsbit.bit_length()-1
        remaining[mask]=remaining[mask^lsbit] & ~forbidden[i]
        winners[mask]=remaining[mask]==0
    for mask,w in enumerate(winners):
        if w != unsat(mask):
            raise AssertionError(f"SAT oracle contradicts union-of-complete-blocks at {mask}")
        if w:
            winning_by_cardinality[mask.bit_count()] += 1
    minimal=[]
    for mask,w in enumerate(winners):
        if w and all(not winners[mask^(1<<j)] for j in range(M) if mask&(1<<j)):
            minimal.append(mask)
    if tuple(minimal)!=core_masks():
        raise AssertionError("unexpected minimal unsat core")
    expected=tuple(
        2*comb(8,k-8) - int(k==16) if 8<=k<=16 else 0
        for k in range(M+1))
    if tuple(winning_by_cardinality)!=expected:
        raise AssertionError("winning polynomial differs from exact core-union formula")
    return {
        "n_vars":6,"clause_universe":M,
        "all_formula_subsets":1<<M,
        "satisfiable_subsets":winners.count(False),
        "unsatisfiable_subsets":winners.count(True),
        "minimal_unsat_cores":minimal,
        "unsat_size_histogram":winning_by_cardinality,
        "independent_assignment_oracle":True,
    }


def brute_sat(mask:int) -> bool:
    """Separate 64-assignment SAT check for individual formula masks."""
    if type(mask) is not int or not 0<=mask<(1<<M):
        raise ValueError("invalid formula")
    for assignment in range(1<<(K*R)):
        if all(not (mask & (1 << (g*8 + ((assignment>>(3*g)) & 7))))
               for g in range(R)):
            return True
    return False


def bernoulli_unsat(p, r:int=R,k:int=GADGET_CLAUSES) -> F:
    valid(r,k)
    p=F(p)
    if not 0<=p<=1:
        raise ValueError("inclusion probability must be in [0,1]")
    return 1-(1-p**k)**r


def histogram_probability(hist:list[int],p) -> F:
    p=F(p)
    if not 0<=p<=1 or len(hist)!=M+1:
        raise ValueError("bad histogram or inclusion probability")
    return sum((count*p**k*(1-p)**(M-k) for k,count in enumerate(hist)),F())


def optimal_cover_cost(p,r:int=R,k:int=GADGET_CLAUSES) -> F:
    """Exact integral AND fractional cover optimum with empty set allowed."""
    valid(r,k)
    p=F(p)
    if not 0<=p<=1:
        raise ValueError("inclusion probability must lie in [0,1]")
    return min(F(1),r*p**k)


def cover_threshold_float(r:int=R,k:int=GADGET_CLAUSES) -> float:
    valid(r,k)
    return (1/(2*r))**(1/k)


def median_threshold_float(r:int=R,k:int=GADGET_CLAUSES) -> float:
    valid(r,k)
    return (1-2**(-1/r))**(1/k)


def complete_group_by_time(t:int,group_size:int=8,total:int=M)->F:
    """Inclusion-exclusion: fixed group has all its coupons by time t."""
    if type(t) is not int or t<0 or type(group_size) is not int or not (
            1<=group_size<=total):
        raise ValueError("invalid time or coupon universe")
    return sum(((-1)**j*comb(group_size,j)*F(total-j,total)**t
                for j in range(group_size+1)),F())


def hit_cdf(t:int)->F:
    """Two disjoint groups of eight; stop on the first complete group."""
    return 2*complete_group_by_time(t,8,M)-complete_group_by_time(t,M,M)


def stirling_second(t:int,max_k:int=M)->tuple[int,...]:
    if type(t) is not int or t<0 or type(max_k) is not int or max_k<0:
        raise ValueError("nonnegative Stirling row parameters required")
    cur=[1]+[0]*max_k
    for _ in range(t):
        nxt=[0]*(max_k+1)
        for k in range(1,max_k+1):
            nxt[k]=k*cur[k]+cur[k-1]
        cur=nxt
    return tuple(cur)


def hit_cdf_by_histogram(t:int,hist:tuple[int,...])->F:
    """Independent surjection counting of winning distinct-clause supports."""
    S=stirling_second(t,M)
    return F(sum((hist[k]*factorial(k)*S[k] for k in range(M+1))),M**t)


def hitting_moments()->dict:
    """Exact first/second moments from geometric survival decomposition.

    P(H>t)=sum_{j=1}^16 (-1)^j*(C(16,j)-2*C(8,j))*
             ((16-j)/16)^t
    """
    coeff=lambda j: (-1)**j*(comb(M,j)-2*(comb(8,j) if j<=8 else 0))
    expected=sum((F(M*coeff(j),j) for j in range(1,M+1)),F())
    second=sum((F(M*(2*M-j)*coeff(j),j*j) for j in range(1,M+1)),F())
    var=second-expected**2
    if expected<=0 or var<=0:
        raise AssertionError("invalid stopping time moments")
    return {"mean":str(expected),"mean_float":float(expected),
            "second_moment":str(second),"variance":str(var),
            "variance_float":float(var)}


def hitting_moments_independent_dp_float()->dict:
    """Distinct recursion: expected time and second moment on all 2^16 masks."""
    count=1<<M
    first=[0.0]*count
    second=[0.0]*count
    for mask in range(count-1,-1,-1):
        if unsat(mask):
            continue
        remaining=M-mask.bit_count()
        next_first=next_second=0.0
        for j in range(M):
            if not(mask&(1<<j)):
                next_first+=first[mask|(1<<j)]
                next_second+=second[mask|(1<<j)]
        T=(M+next_first)/remaining
        first[mask]=T
        second[mask]=(2*M*T-M+next_second)/remaining
    return {"mean":first[0],"variance":second[0]-first[0]**2,
            "state_count":count,
            "algorithm":"independent subset-Markov dynamic programming, float64"}


def one_gadget_coupon_moments(k:int=8)->dict:
    if type(k) is not int or k<1:
        raise ValueError("k>=1")
    mean=k*sum((F(1,j) for j in range(1,k+1)),F())
    var=k*k*sum((F(1,j*j) for j in range(1,k+1)),F())-mean
    return {"clause_types":k,"mean":str(mean),"variance":str(var),
            "model":"uniform with replacement, n=3 full proper clause universe"}


def minimal_monitor_states(r:int=R,k:int=GADGET_CLAUSES)->int:
    valid(r,k)
    return ((1<<k)-1)**r+1


def monitor_bits(r:int=R,k:int=GADGET_CLAUSES)->int:
    return (minimal_monitor_states(r,k)-1).bit_length()


def encode_monitor(mask:int,r:int=R,k:int=GADGET_CLAUSES)->int:
    if type(mask) is not int or not 0<=mask<(1<<(r*k)):
        raise ValueError("invalid clause mask")
    base=(1<<k)-1
    state=0
    for g in range(r):
        chunk=(mask>>(g*k)) & ((1<<k)-1)
        if chunk==base:
            return base**r
        state+=chunk*(base**g)
    return state


def step_monitor(state:int,clause:int,r:int=R,k:int=GADGET_CLAUSES)->int:
    valid(r,k)
    base=(1<<k)-1
    final=base**r
    if type(state) is not int or not 0<=state<=final or (
            type(clause) is not int or not 0<=clause<r*k):
        raise ValueError("invalid monitor state or clause")
    if state==final:
        return final
    group,offset=divmod(clause,k)
    old=(state//(base**group))%base
    new=old|(1<<offset)
    if new==base:
        return final
    return state+(new-old)*base**group


def separating_suffix(a:int,b:int,r:int=R,k:int=GADGET_CLAUSES):
    """Myhill-Nerode witness distinguishing any two incomplete supports."""
    valid(r,k)
    if a==b or unsat(a,r,k) or unsat(b,r,k):
        raise ValueError("distinct losing supports required")
    if not a & ~b:
        a,b=b,a
    differing=a&~b
    j=(differing&-differing).bit_length()-1
    group=j//k
    missing=gadget_full_mask(group,k) & ~a
    suffix=[i for i in range(r*k) if missing&(1<<i)]
    A=a|missing;B=b|missing
    if not unsat(A,r,k) or unsat(B,r,k):
        raise AssertionError("suffix failed to distinguish incomplete supports")
    return {"oriented_A":a,"oriented_B":b,"gadget":group,
            "appended_clause_types":suffix,
            "A_unsat_after_suffix":True,"B_sat_after_suffix":True}


def promised_unique_witness_success_ceiling(r:int, B:int)->F:
    if type(r) is not int or r<1 or type(B) is not int or B<0:
        raise ValueError("positive family and nonnegative bits")
    return min(F(1),F(1<<B,r))


def source_report()->dict:
    all_masks=exhaustive_sat_oracle()
    cover_rows=[]
    for p in (F(1,4),F(1,2),F(3,4),F(4,5),F(9,10),F(1)):
        exact=bernoulli_unsat(p)
        from_oracle=histogram_probability(all_masks["unsat_size_histogram"],p)
        if exact!=from_oracle:
            raise AssertionError("independent 65k-set polynomial mismatch")
        cover=optimal_cover_cost(p)
        cover_rows.append({"p":str(p),"unsat_probability":str(exact),
                           "unsat_probability_float":float(exact),
                           "exact_integral_cover_cost":str(cover),
                           "exact_fractional_cover_cost":str(cover)})
    times=[]
    for t in (0,7,8,12,16,24,30,31,32,40,48,64):
        cdf=hit_cdf(t)
        via_sets=hit_cdf_by_histogram(t,tuple(all_masks["unsat_size_histogram"]))
        if cdf!=via_sets:
            raise AssertionError("independent Stirling / inclusion-exclusion CDF mismatch")
        times.append({"t":t,"first_UNSAT_cdf":str(cdf),
                      "first_UNSAT_probability_float":float(cdf),
                      "independent_surjection_verified":True})
    moments=hitting_moments()
    dp=hitting_moments_independent_dp_float()
    if abs(moments["mean_float"]-dp["mean"])>1e-9 or (
            abs(moments["variance_float"]-dp["variance"])>1e-8):
        raise AssertionError("independent state DP moment mismatch")
    if not hit_cdf(30)<F(1,2)<=hit_cdf(31):
        raise AssertionError("with-replacement median changed")
    enc=set(encode_monitor(mask) for mask in range(1<<M))
    if len(enc)!=minimal_monitor_states():
        raise AssertionError("minimal finite stream automaton count disagrees")
    return {
        "schema":"sct01.sat_threshold_witness.v1",
        "status":"EXACT_FINITE_SAT_CORE_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_families":[175,235],
        "finite_sat_oracle":all_masks,
        "cover_costs":cover_rows,
        "integral_fractional_cover_threshold_float":cover_threshold_float(),
        "Bernoulli_UNSAT_median_probability_float":median_threshold_float(),
        "with_replacement_CDF":times,
        "with_replacement_first_UNSAT_median_draw":31,
        "first_unsat_moments":moments,
        "independent_markov_moment_oracle":dp,
        "full_n3_random_3SAT_coupon_collector":one_gadget_coupon_moments(),
        "streaming_exact_min_states":minimal_monitor_states(),
        "streaming_exact_min_bits":monitor_bits(),
        "streaming_exact_encoded_states_enumerated":len(enc),
        "minimal_unsat_proof_clause_count":8,
        "promise_exactly_one_full_gadget_unique_witness_bits_r8":monitor_bits(8,1),
        "general_source_175_constant_gap_proved_here":False,
        "general_source_235_limiting_density_proved_here":False,
        "full_random_n6_three_SAT_clause_model":False,
        "trained_reasoner_evaluated":False,
    }


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    result=source_report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    brief={k:result[k] for k in (
        "status","integral_fractional_cover_threshold_float",
        "Bernoulli_UNSAT_median_probability_float",
        "with_replacement_first_UNSAT_median_draw",
        "first_unsat_moments","streaming_exact_min_states",
        "streaming_exact_min_bits","minimal_unsat_proof_clause_count")}
    print(json.dumps(brief,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
