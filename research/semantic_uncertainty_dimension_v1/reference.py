"""Finite BSC + Cantor posterior reference for SUD-01 (stdlib only).
Finite sums are enumerated exactly; reported entropies use float arithmetic.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math

def binary_entropy(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

def truth_mask(n: int, fn) -> int:
    return sum(1 << x for x in range(1 << n) if fn(x))

def is_balanced(n: int, mask: int) -> bool:
    return mask.bit_count() == (1 << (n-1))

def stats(n: int, mask: int, p: float, r: float=1/3) -> dict:
    if not (1 <= n <= 8 and 0 <= mask < (1 << (1 << n))):
        raise ValueError("require n=1..8 and a valid truth-table mask")
    if not (0 <= p <= .5 and 0 < r < .5):
        raise ValueError("require p in [0,1/2], r in (0,1/2)")
    k = 1 << n
    prior = mask.bit_count() / k
    entropy = error = 0.0
    for y in range(k):
        q = 0.0
        for x in range(k):
            if (mask >> x) & 1:
                d = (x ^ y).bit_count()
                q += p**d * (1-p)**(n-d)
        entropy += binary_entropy(q)/k
        error += min(q, 1-q)/k
    chi = math.log2(1/r)
    return {
        "n": n, "mask": mask, "p": p, "r": r,
        "balanced": is_balanced(n, mask),
        "conditional_entropy_bits": entropy,
        "mutual_information_bits": binary_entropy(prior)-entropy,
        "bayes_error": error,
        "posterior_dimension": entropy/chi,
        "CK_dimension_floor": binary_entropy(p)/chi,
        "CK_error_floor": p
    }

def parity_error(n: int, p: float) -> float:
    return (1-(1-2*p)**n)/2

def balanced_masks(n: int):
    width = 1 << n
    for ones in itertools.combinations(range(width), width//2):
        yield sum(1 << x for x in ones)

def exhaustive_balanced(n: int, p: float, r: float=1/3) -> dict:
    emin = float("inf")
    errmin = float("inf")
    imax = -float("inf")
    count = 0
    for mask in balanced_masks(n):
        a = stats(n, mask, p, r)
        emin = min(emin, a["conditional_entropy_bits"])
        errmin = min(errmin, a["bayes_error"])
        imax = max(imax, a["mutual_information_bits"])
        count += 1
    return {
        "n": n, "p": p, "balanced_functions_checked": count,
        "smallest_entropy": emin, "smallest_bayes_error": errmin,
        "largest_mutual_information": imax,
        "predicted_entropy_floor": binary_entropy(p),
        "predicted_bayes_error_floor": p,
        "predicted_max_info": 1-binary_entropy(p)
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--p", type=float, default=.1)
    parser.add_argument("--r", type=float, default=1/3)
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    rows = []
    for name, mask in [
        ("dictator", truth_mask(args.n, lambda x: x & 1)),
        ("parity", truth_mask(args.n, lambda x: x.bit_count() % 2))
    ]:
        item = stats(args.n, mask, args.p, args.r)
        item["family"] = name
        rows.append(item)
    out = {"instances": rows}
    if args.exhaustive:
        if args.n > 4:
            parser.error("exhaustive enumeration limited to n<=4")
        out["exhaustive"] = exhaustive_balanced(args.n, args.p, args.r)
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
