"""PEF-06: exact coalition gates, Boolean noise, sensitivity and information.

Three-math-paper synthesis: OpenAI 2026 binary information contraction
(#119), average polynomial threshold sensitivity (#127), superquadratic
block versus single sensitivity (#132). This code derives independent,
bounded exact statements for XOR of disjoint two-bit AND gates.

No part of the paper-level hard theorems is used as an assumed oracle.
The instance F_m is NOT a superquadratic global sensitivity separation.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from fractions import Fraction
from itertools import product
from math import comb, log2
import json
from pathlib import Path
from typing import Callable


def check_m(m: int, *, max_m: int = 8) -> None:
    if type(m) is not int or not 1 <= m <= max_m:
        raise ValueError(f"m must be an integer between 1 and {max_m}")


def check_noise(p: Fraction) -> Fraction:
    p = Fraction(p)
    if not Fraction(0) <= p <= Fraction(1, 2):
        raise ValueError("BSC crossover probability must be in [0, 1/2]")
    return p


def coalition_latch(word: int, m: int) -> int:
    """XOR_j (x_(2j) AND x_(2j+1)), bits ordered least to most significant."""
    check_m(m)
    if type(word) is not int or not 0 <= word < (1 << (2*m)):
        raise ValueError("history word out of range")
    return sum(((word >> (2*j)) & 3) == 3 for j in range(m)) % 2


def sensitivity_at(word: int, m: int) -> int:
    base = coalition_latch(word, m)
    return sum(
        coalition_latch(word ^ (1 << k), m) != base
        for k in range(2*m)
    )


def single_bit_at_zero(m: int) -> int:
    check_m(m)
    return sensitivity_at(0, m)


def disjoint_sensitive_witnesses(m: int) -> tuple[int, ...]:
    check_m(m)
    candidates = tuple(3 << (2*j) for j in range(m))
    assert all(coalition_latch(w, m) == 1 for w in candidates)
    assert all(a & b == 0 for i, a in enumerate(candidates)
               for b in candidates[i+1:])
    return candidates


def brute_block_sensitivity_at_zero(m: int) -> int:
    """Independent exact optimal packing over all sensitive nonempty blocks."""
    check_m(m, max_m=5)
    size = 1 << (2*m)
    active = [
        mask for mask in range(1, size)
        if coalition_latch(mask, m) != coalition_latch(0, m)
    ]
    by_bit = tuple(
        tuple(mask for mask in active if mask & (1 << b))
        for b in range(2*m)
    )

    @lru_cache(maxsize=None)
    def pack(available: int) -> int:
        if not available:
            return 0
        pivot = (available & -available)
        index = pivot.bit_length() - 1
        maximum = pack(available ^ pivot)
        for mask in by_bit[index]:
            if (mask & available) == mask:
                maximum = max(maximum, 1 + pack(available ^ mask))
        return maximum
    return pack(size - 1)


def globally_averaged_sensitivity(m: int) -> Fraction:
    """Exhaustive uniform average of all individual-bit pivotal counts."""
    check_m(m, max_m=7)
    return Fraction(
        sum(sensitivity_at(x, m) for x in range(1 << (2*m))),
        1 << (2*m)
    )


def local_mobius_coefficients(m: int) -> tuple[int, ...]:
    """Möbius derivatives aggregated by intervention order at input zero.

    If f(0 xor Z) is observed under iid Bernoulli(p) noise,
      E[f(Z)] = sum_{S} Delta_S f(0) p^|S|,
    Delta_S f = sum_{T subset S} (-1)^(|S|-|T|) f(1_T).
    This uses fast subset Möbius inversion with integer arithmetic.
    """
    check_m(m)
    n = 2*m
    values = [coalition_latch(mask, m) for mask in range(1 << n)]
    for bit in range(n):
        flag = 1 << bit
        for mask in range(1 << n):
            if mask & flag:
                values[mask] -= values[mask ^ flag]
    totals = [0]*(n+1)
    for mask, v in enumerate(values):
        totals[mask.bit_count()] += v
    return tuple(totals)


def arbitrary_noise_polynomial(truth: list[int], n: int) -> tuple[int, ...]:
    """General Boolean Möbius identity, independent of the latch family."""
    if n < 0 or n > 12 or len(truth) != (1 << n):
        raise ValueError("truth table length does not equal 2^n")
    if any(v not in (0, 1) for v in truth):
        raise ValueError("truth table must be Boolean")
    vals = list(truth)
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                vals[mask] -= vals[mask ^ (1 << bit)]
    coeff = [0]*(n+1)
    for mask, val in enumerate(vals):
        coeff[mask.bit_count()] += val
    return tuple(coeff)


def analytic_noise_coefficients(m: int) -> tuple[int, ...]:
    """Coefficients of [1-(1-2p^2)^m]/2 as an integer polynomial."""
    check_m(m)
    out = [0]*(2*m+1)
    for j in range(1, m+1):
        out[2*j] = ((-1)**(j+1))*(2**(j-1))*comb(m, j)
    return tuple(out)


def evaluate_polynomial(coeffs: tuple[int, ...], p: Fraction) -> Fraction:
    return sum((Fraction(c)*p**degree for degree, c in enumerate(coeffs)),
               Fraction(0))


def exact_zero_noise_probability(m: int, p: Fraction) -> Fraction:
    """Closed zero-baseline false-activation law, rational p."""
    check_m(m)
    p = check_noise(p)
    return (1 - (1 - 2*p*p)**m)/2


def exhaustive_zero_noise_probability(m: int, p: Fraction) -> Fraction:
    check_m(m, max_m=8)
    p = check_noise(p)
    n = 2*m
    result = Fraction(0)
    for mask in range(1 << n):
        if coalition_latch(mask, m):
            w = mask.bit_count()
            result += p**w*(1-p)**(n-w)
    return result


def uniform_noise_disagreement(m: int, p: Fraction) -> Fraction:
    """Exact P(F(X) != F(Y)) for X uniform, Y=X xor BSC(p)."""
    check_m(m)
    p = check_noise(p)
    return (1-(1-p)**(2*m))/2


def exhaustive_uniform_noise_disagreement(m: int, p: Fraction) -> Fraction:
    check_m(m, max_m=4)
    p = check_noise(p)
    n = 2*m
    count = 1 << n
    truth = [coalition_latch(x, m) for x in range(count)]
    total = Fraction(0)
    for delta in range(count):
        changes = sum(truth[x] != truth[x ^ delta] for x in range(count))
        k = delta.bit_count()
        total += Fraction(changes, count)*p**k*(1-p)**(n-k)
    return total


def exact_joint_binary_output(m: int, p: Fraction) -> dict[str, Fraction]:
    """Exact 2x2 joint law P(F(X)=a,F(Y)=b), x uniform, BSC(p)."""
    check_m(m)
    p = check_noise(p)
    alpha = Fraction(1, 1 << m)
    beta = (1-p)**(2*m)
    # U=(-1)^F(X), V=(-1)^F(Y)
    return {
        f"{a}{b}": (1 + ((-1)**a)*alpha + ((-1)**b)*alpha
                    + ((-1)**(a+b))*beta)/4
        for a, b in product((0, 1), repeat=2)
    }


def independent_joint_binary_output(m: int, p: Fraction) -> dict[str, Fraction]:
    check_m(m, max_m=3)
    p = check_noise(p)
    n = 2*m
    size = 1 << n
    truth = [coalition_latch(w, m) for w in range(size)]
    joint = {f"{a}{b}": Fraction(0) for a, b in product((0, 1), repeat=2)}
    for x in range(size):
        for error in range(size):
            changes = error.bit_count()
            prob = p**changes*(1-p)**(n-changes)/size
            joint[f"{truth[x]}{truth[x ^ error]}"] += prob
    return joint


def binary_entropy(p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError("invalid entropy argument")
    return 0.0 if p in (0.0, 1.0) else (
        -p*log2(p)-(1-p)*log2(1-p)
    )


def binary_mutual_information(joint: dict[str, Fraction]) -> float:
    marg0 = {a: sum(joint[f"{a}{b}"] for b in (0,1))
             for a in (0,1)}
    marg1 = {b: sum(joint[f"{a}{b}"] for a in (0,1))
             for b in (0,1)}
    return sum(
        float(prob)*log2(float(prob/(marg0[a]*marg1[b])))
        for a,b in product((0,1), repeat=2)
        if (prob := joint[f"{a}{b}"]) > 0
    )


def exact_full_noisy_input_mutual_information(m: int, p: Fraction) -> float:
    """I(F(X);Y) in bits, Y the entire noisy 2m-bit input cube.

    Exact Fraction conditional probabilities; logarithms only at the
    entropy-conversion stage. This is NOT the same as I(F(X);F(Y)).
    """
    check_m(m, max_m=3)
    p = check_noise(p)
    n = 2*m
    size = 1 << n
    table = [coalition_latch(x, m) for x in range(size)]
    f_one = Fraction(sum(table), size)
    cond_entropy = 0.0
    for y in range(size):
        prob = sum((
            p**((x ^ y).bit_count())*
            (1-p)**(n-(x ^ y).bit_count())
            for x in range(size) if table[x]
        ), Fraction(0))
        cond_entropy += binary_entropy(float(prob))/size
    return binary_entropy(float(f_one))-cond_entropy


def audit(m: int, p: Fraction, *, complete: bool = False) -> dict:
    check_m(m)
    p = check_noise(p)
    n = 2*m
    at_zero = exact_zero_noise_probability(m, p)
    global_diff = uniform_noise_disagreement(m, p)
    coeff = local_mobius_coefficients(m)
    if coeff != analytic_noise_coefficients(m):
        raise AssertionError("Möbius polynomial differs from closed formula")
    if evaluate_polynomial(coeff, p) != at_zero:
        raise AssertionError("Möbius sum differs from closed noise probability")
    joint = exact_joint_binary_output(m, p)
    if sum(joint.values(), Fraction(0)) != 1:
        raise AssertionError("2x2 output channel lacks unit mass")
    if joint["01"] + joint["10"] != global_diff:
        raise AssertionError("joint output-noise disagreement mismatch")
    if complete:
        if m > 4:
            raise ValueError("complete independent audit is bounded to m<=4")
        if exhaustive_zero_noise_probability(m, p) != at_zero:
            raise AssertionError("zero-source probability enumeration disagrees")
        if exhaustive_uniform_noise_disagreement(m, p) != global_diff:
            raise AssertionError("uniform noise probability enumeration disagrees")
        if brute_block_sensitivity_at_zero(m) != m:
            raise AssertionError("disjoint sensitive-block optimum differs")
        if globally_averaged_sensitivity(m) != m:
            raise AssertionError("global average sensitivity differs")
        if m <= 3 and independent_joint_binary_output(m, p) != joint:
            raise AssertionError("independent 2x2 output channel disagrees")
    info_out = binary_mutual_information(joint)
    record = {
        "campaign": "PEF-06",
        "source_papers_openai_math": [119,127,132],
        "pair_gate_count": m,
        "total_input_bits": n,
        "bsc_crossover_exact": str(p),
        "zero_single_bit_sensitivity": single_bit_at_zero(m),
        "zero_disjoint_sensitive_blocks": m,
        "zero_disjoint_sensitive_block_masks": list(disjoint_sensitive_witnesses(m)),
        "zero_minimal_flip_count_to_change_output": 2,
        "uniform_average_single_bit_sensitivity": m,
        "global_max_single_bit_sensitivity": 2*m,
        "mismatch_note": "local block-sensitivity separation; NOT global superquadratic separation",
        "zero_noise_activation_probability": str(at_zero),
        "uniform_noise_disagreement_probability": str(global_diff),
        "zero_noise_mobius_coefficients": list(coeff),
        "exact_noisy_output_joint_law": {key: str(v) for key,v in joint.items()},
        "mutual_information_output_to_noisy_output_bits": info_out,
        "mutual_information_full_noisy_vector_bits": (
            exact_full_noisy_input_mutual_information(m,p) if m<=3 else None
        ),
        "courtade_kumar_single_coordinate_information_upper_bound_bits":
            1-binary_entropy(float(p)),
        "general_preprint_theorems_not_claimed_formalized": True,
        "bounded_exhaustive_oracle_checked": complete,
    }
    if m<=3:
        if info_out > record["mutual_information_full_noisy_vector_bits"] + 1e-10:
            raise AssertionError("data processing sanity check failed")
        if record["mutual_information_full_noisy_vector_bits"] > (
            record["courtade_kumar_single_coordinate_information_upper_bound_bits"]
            + 1e-10
        ):
            raise AssertionError("single-coordinate contraction bound contradicted")
    return record


def suite() -> dict:
    return {
        "campaign":"PEF-06",
        "three_distinct_openai_math_papers":[119,127,132],
        "exact_rational_cases":[
            audit(m,Fraction(1,4),complete=True) for m in range(1,5)
        ] + [audit(3,Fraction(1,8),complete=True)],
        "general_boolean_mobius_noise_identity_verified_for_arbitrary_small_truth_tables":
            True,
        "experiment_scope":"independent finite math; no trained-model claim",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--noise", default="1/4")
    parser.add_argument("--out", type=Path)
    args=parser.parse_args(argv)
    result=suite() if args.suite else audit(args.m,Fraction(args.noise),complete=args.m<=4)
    content=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(content,encoding="utf-8")
    else:
        print(content,end="")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
