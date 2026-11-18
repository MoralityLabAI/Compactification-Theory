"""ITRC-REV-01 exact bounded predictive-capacity and resource accounting.

The finite correction interface is synthetically defined. Source/identity,
versioning and authentication costs are external to the count-bit figures.
"""
from itertools import combinations
from math import comb


def capacity(n:int=12,j:int=6)->dict:
    if type(n)!=int or type(j)!=int or not 2<=n<=12 or not 1<=j<n:
        raise ValueError("proper nonempty delayed correction subset required")
    largest=max(comb(n,m) for m in range(n+1))
    return {
        "n":n,"j":j,"present_count_classes":n+1,
        "delayed_continuation_classes":1<<n,
        "delayed_self_contained_bits":n,
        "extra_bits_with_retained_present_posterior":(largest-1).bit_length(),
        "early_initial_valid_count_states":n-j+1,
        "early_initial_bits":(n-j).bit_length(),
        "early_peak_count_bits":n.bit_length(),
        "external_verified_old_sum_message_bits":j.bit_length(),
        "external_old_archive_minimum_raw_bits":n,
        "archive_message_optimum_claim":False,
        "physical_memory_budget_reported":False,
        "causal_neural_claim":False,
    }


def signature(mask:int,n:int,j:int)->tuple:
    """Current count and all correction-future retained counts."""
    capacity(n,j)
    if type(mask)!=int or not 0<=mask<1<<n:
        raise ValueError("invalid history")
    return (mask.bit_count(),tuple(
        sum((mask>>i)&1 for i in range(n) if i not in correction)
        for correction in combinations(range(n),j)))


def witness(a:int,b:int,n:int,j:int)->tuple[int,...]:
    """One distinguished future subset for equal-count old vectors."""
    capacity(n,j)
    if (type(a)!=int or type(b)!=int or not 0<=a<1<<n
            or not 0<=b<1<<n or a==b or a.bit_count()!=b.bit_count()):
        raise ValueError("distinct same-count vectors required")
    for subset in combinations(range(n),j):
        u=sum((a>>i)&1 for i in range(n) if i not in subset)
        v=sum((b>>i)&1 for i in range(n) if i not in subset)
        if u!=v:
            return subset
    raise AssertionError("finite quotient theorem violated")


def finite_audit(max_n:int=7)->dict:
    if type(max_n)!=int or not 2<=max_n<=7:
        raise ValueError("bounded independent-source audit")
    checked=0
    worlds=0
    for n in range(2,max_n+1):
        for j in range(1,n):
            signatures={signature(mask,n,j) for mask in range(1<<n)}
            if len(signatures)!=1<<n:
                raise AssertionError("continuation histories collided")
            worlds+=1
            for a,b in combinations(range(1<<n),2):
                if a.bit_count()!=b.bit_count():
                    continue
                correction=witness(a,b,n,j)
                if (sum((a>>i)&1 for i in range(n) if i not in correction)
                    ==sum((b>>i)&1 for i in range(n) if i not in correction)):
                    raise AssertionError("witness did not break current-equivalent pair")
                checked+=1
    return {"source_only":True,"exact_domains":worlds,
            "same_count_pair_checks":checked,
            "neural_models_evaluated":0}
