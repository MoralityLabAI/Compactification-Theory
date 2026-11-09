"""RRC-01: exact Rogers--Ramanujan coefficient oracles and periodic-rule recovery.

Standard-library only. No neural model is trained and no infinite identity is proved
by the finite experiments here.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ResidueRule:
    modulus: int
    residues: tuple[int, ...]

    @property
    def code_bits(self) -> int:
        """Illustrative grammar: Elias-gamma code for m + m-bit residue mask.

        Excludes the cost of the program, coefficients and output storage.
        """
        return 2 * (self.modulus.bit_length() - 1) + 1 + self.modulus

    def as_dict(self) -> dict:
        return {
            "modulus": self.modulus,
            "residues": list(self.residues),
            "grammar_bits_excluding_decoder": self.code_bits,
        }


def gap_coefficients(degree: int, *, gap: int = 2, minimum_part: int = 1) -> list[int]:
    """Coefficient vector of partitions with successive parts >=gap apart.

    F_L(q)=F_{L-1}(q)+q^L F_{L-gap}(q), initialized by F_L=1 for
    L<minimum_part. Exact integer truncated polynomial arithmetic.
    """
    if degree < 0 or gap < 1 or minimum_part < 1:
        raise ValueError("degree must be >=0 and gap/minimum_part must be >=1")
    one = [1] + [0] * degree
    history: deque[list[int]] = deque((one.copy() for _ in range(gap)), maxlen=gap)
    for part in range(minimum_part, degree + 1):
        current = history[-1].copy()
        prior = history[0]
        for n in range(part, degree + 1):
            current[n] += prior[n - part]
        history.append(current)
    return history[-1]


def staircase_coefficients(degree: int, *, minimum_part: int = 1) -> list[int]:
    """Sum q^(k^2+(s-1)k)/(q;q)_k using unrestricted incremental coin DP.

    For k gap-separated parts, remove the staircase
    [s+2(k-1), ..., s+2, s]; the remainder is a partition with <=k parts.
    """
    if degree < 0 or minimum_part < 1:
        raise ValueError("invalid degree/minimum_part")
    pochhammer_inverse = [1] + [0] * degree
    answer = pochhammer_inverse.copy()  # k=0 term
    k = 1
    while k * (k + minimum_part - 1) <= degree:
        for n in range(k, degree + 1):
            pochhammer_inverse[n] += pochhammer_inverse[n - k]
        offset = k * (k + minimum_part - 1)
        for n in range(degree - offset + 1):
            answer[n + offset] += pochhammer_inverse[n]
        k += 1
    return answer


def euler_exponents(coefficients: Sequence[int]) -> list[int]:
    """Unique integral a_m with F(q)=prod_m (1-q^m)^(-a_m) mod q^(N+1).

    qF'/F has nth coefficient b_n=sum_{d|n} d*a_d; solving
    n*f_n=sum_{k=1}^n b_k*f_(n-k) gives an exact Euler transform.
    """
    if not coefficients or coefficients[0] != 1:
        raise ValueError("series must have constant term 1")
    degree = len(coefficients) - 1
    b = [0] * (degree + 1)
    a = [0] * (degree + 1)
    proper_divisors = [0] * (degree + 1)
    for n in range(1, degree + 1):
        b[n] = n * coefficients[n] - sum(b[k] * coefficients[n - k] for k in range(1, n))
        numerator = b[n] - proper_divisors[n]
        if numerator % n:
            raise ArithmeticError(f"nonintegral Euler exponent at {n}")
        a[n] = numerator // n
        for multiple in range(2 * n, degree + 1, n):
            proper_divisors[multiple] += n * a[n]
    return a


def infer_periodic_rule(
    train_coefficients: Sequence[int], *, modulus_cap: int = 12, min_observations: int = 2
) -> ResidueRule | None:
    """Choose the smallest period with a stable binary Euler exponent mask.

    Uses only the supplied training coefficients. A residue needs at least two
    observed instances, so large periods cannot simply memorize the prefix.
    """
    if modulus_cap < 1 or min_observations < 2:
        raise ValueError("modulus_cap >=1 and min_observations >=2 are required")
    a = euler_exponents(train_coefficients)
    n_train = len(a) - 1
    for modulus in range(1, modulus_cap + 1):
        observed = [[a[n] for n in range(1, n_train + 1) if n % modulus == r]
                    for r in range(modulus)]
        if any(len(entries) < min_observations for entries in observed):
            continue
        if all(entries[0] in (0, 1) and len(set(entries)) == 1 for entries in observed):
            allowed = tuple(r for r, entries in enumerate(observed) if entries[0] == 1)
            return ResidueRule(modulus, allowed)
    return None


def product_coefficients(degree: int, rule: ResidueRule) -> list[int]:
    """Independent unlimited-part coin-change oracle for an inferred mask."""
    if degree < 0 or rule.modulus < 1:
        raise ValueError("invalid degree or modulus")
    if any(r < 0 or r >= rule.modulus for r in rule.residues):
        raise ValueError("residue out of range")
    allowed = set(rule.residues)
    answer = [1] + [0] * degree
    for part in range(1, degree + 1):
        if part % rule.modulus in allowed:
            for n in range(part, degree + 1):
                answer[n] += answer[n - part]
    return answer


def first_mismatch(left: Sequence[int], right: Sequence[int], *, start: int = 0) -> int | None:
    if len(left) != len(right):
        raise ValueError("length mismatch")
    return next((n for n in range(start, len(left)) if left[n] != right[n]), None)


def run_experiment(*, train_degree: int = 64, eval_degree: int = 256,
                   modulus_cap: int = 12) -> dict:
    if train_degree < 2 * modulus_cap or eval_degree <= train_degree:
        raise ValueError("require train_degree >= 2*modulus_cap and eval_degree > train_degree")
    families: list[dict] = []
    for minimum_part in (1, 2):
        gap = gap_coefficients(eval_degree, minimum_part=minimum_part)
        series = staircase_coefficients(eval_degree, minimum_part=minimum_part)
        staircase_mismatch = first_mismatch(gap, series)
        if staircase_mismatch is not None:
            raise AssertionError(f"local/q-series disagreement at {staircase_mismatch}")
        rule = infer_periodic_rule(gap[:train_degree + 1], modulus_cap=modulus_cap)
        if rule is None:
            raise AssertionError(f"no periodic rule inferred for minimum part {minimum_part}")
        predicted = product_coefficients(eval_degree, rule)
        family = {
            "minimum_part": minimum_part,
            "fitted_rule": rule.as_dict(),
            "training_mismatch": first_mismatch(gap[:train_degree + 1], predicted[:train_degree + 1]),
            "heldout_first_mismatch": first_mismatch(gap, predicted, start=train_degree + 1),
            "staircase_first_mismatch": staircase_mismatch,
            "heldout_coefficient_count": eval_degree - train_degree,
            "coefficients_0_to_12": gap[:13],
            "oracle_coefficients_sha256": hashlib.sha256(json.dumps(gap).encode()).hexdigest(),
        }
        families.append(family)
    # Control 1: a locally modified combinatorial rule is outside the two RR families.
    gap3 = gap_coefficients(eval_degree, gap=3, minimum_part=1)
    spur = infer_periodic_rule(gap3[:train_degree + 1], modulus_cap=modulus_cap)
    gap3_heldout_match = None if spur is None else first_mismatch(
        gap3, product_coefficients(eval_degree, spur), start=train_degree + 1
    )
    # Control 2: the same product grammar, with a wrong periodic mask.
    correct = infer_periodic_rule(
        gap_coefficients(train_degree)[:train_degree + 1], modulus_cap=modulus_cap
    )
    assert correct is not None
    rotated = ResidueRule(correct.modulus, tuple(sorted((r + 1) % correct.modulus for r in correct.residues)))
    wrong = product_coefficients(eval_degree, rotated)
    wrong_first_mismatch = first_mismatch(gap_coefficients(eval_degree), wrong, start=train_degree + 1)
    result = {
        "schema": "rrc01.v1",
        "status": "exact_cpu_finite_degree_only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {
            "train_degree": train_degree,
            "eval_degree": eval_degree,
            "modulus_cap": modulus_cap,
            "minimum_residue_observations": 2,
        },
        "families": families,
        "negative_controls": {
            "gap_3_rule": None if spur is None else spur.as_dict(),
            "gap_3_heldout_first_mismatch": gap3_heldout_match,
            "rotated_rule": rotated.as_dict(),
            "rotated_rule_heldout_first_mismatch": wrong_first_mismatch,
        },
        "scope": "finite exact integer tests; known Rogers-Ramanujan theorem is an external mathematical fact",
    }
    if any(f["fitted_rule"]["modulus"] != 5 or
           f["fitted_rule"]["residues"] != ([1, 4] if f["minimum_part"] == 1 else [2, 3]) or
           f["training_mismatch"] is not None or f["heldout_first_mismatch"] is not None
           for f in families):
        raise AssertionError("Rogers-Ramanujan reference gate failed")
    if spur is not None and gap3_heldout_match is None:
        raise AssertionError("gap=3 control unexpectedly generalizes exactly")
    if wrong_first_mismatch is None:
        raise AssertionError("rotated-mask control unexpectedly generalizes exactly")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--degree", type=int, default=256)
    p.add_argument("--modulus-cap", type=int, default=12)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true", help="compare --out to regenerated canonical result")
    args = p.parse_args()
    result = run_experiment(train_degree=args.train, eval_degree=args.degree,
                            modulus_cap=args.modulus_cap)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.out is None or not args.out.is_file() or args.out.read_text() != payload:
            raise SystemExit("missing or stale receipt; regenerate without --check")
        print("RRC-01 exact CPU receipt verified")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
