"""PEF-01: exact depth / fixed-point precision / information frontier.

This CPU-only, finite integer and rational oracle concerns an idealized
input-conditioned self-similar recurrence, not trained transformer weights.
For base=2**k and bit b, x'=(x + (base-1)*b)/base from x0=0.
It computes exact quantization fibres, Bayes history ambiguity and
intervention visibility without floating-point arithmetic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


@dataclass(frozen=True)
class Case:
    k: int
    depth: int
    precision_bits: int

    def validate(self, max_depth: int = 12) -> None:
        if type(self.k) is not int or self.k < 1:
            raise ValueError("k must be a positive integer")
        if type(self.depth) is not int or not 0 <= self.depth <= max_depth:
            raise ValueError(f"depth must be between 0 and {max_depth}")
        if type(self.precision_bits) is not int or self.precision_bits < 0:
            raise ValueError("precision_bits must be a nonnegative integer")
        if self.precision_bits > 128:
            raise ValueError("bounded exact audit supports <= 128 readout bits")


def encode_integer(history: tuple[int, ...], k: int) -> int:
    """Integer numerator with the *oldest* input in lowest base digit."""
    base = 1 << k
    return (base - 1) * sum(bit * base**i for i, bit in enumerate(history))


def rational_orbit(history: tuple[int, ...], k: int) -> Fraction:
    """Independent recurrence evaluation, intentionally separate from encoder."""
    base = 1 << k
    x = Fraction(0)
    for bit in history:
        if bit not in (0, 1):
            raise ValueError("history bits must be binary")
        x = (x + (base - 1)*bit) / base
    return x


def quantize_integer(history: tuple[int, ...], k: int, precision_bits: int) -> int:
    if k < 1 or precision_bits < 0:
        raise ValueError("invalid contraction or precision")
    if any(bit not in (0, 1) for bit in history):
        raise ValueError("history bits must be binary")
    base = 1 << k
    numerator = encode_integer(history, k)
    return ((1 << precision_bits) * numerator) // (base ** len(history))


def quantize_independent(history: tuple[int, ...], k: int, precision_bits: int) -> int:
    x = rational_orbit(history, k)
    return ((1 << precision_bits) * x.numerator) // x.denominator


def retained_bits(k: int, depth: int, precision_bits: int) -> int:
    if k < 1 or depth < 0 or precision_bits < 0:
        raise ValueError("invalid parameters")
    return min(depth, (precision_bits + k - 1) // k)


def minimum_full_recovery_precision(k: int, depth: int) -> int:
    if k < 1 or depth < 0:
        raise ValueError("invalid parameters")
    return 0 if depth == 0 else k * (depth - 1) + 1


def exact_audit(case: Case, *, check_interventions: bool = True) -> dict:
    """Enumerate each uniform history, prove the balanced fibre partition."""
    case.validate()
    k, depth, bits = case.k, case.depth, case.precision_bits
    t = retained_bits(k, depth, bits)
    histories = list(product((0, 1), repeat=depth))
    code_to_histories: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    suffix_to_code: dict[tuple[int, ...], int] = {}
    code_to_suffix: dict[int, tuple[int, ...]] = {}
    intervention_trials = 0
    for hist in histories:
        code = quantize_integer(hist, k, bits)
        assert code == quantize_independent(hist, k, bits)
        suffix = hist[depth-t:] if t else ()
        if suffix in suffix_to_code:
            assert suffix_to_code[suffix] == code, "older input affects readout"
        if code in code_to_suffix:
            assert code_to_suffix[code] == suffix, "visible suffix was collapsed"
        suffix_to_code[suffix] = code
        code_to_suffix[code] = suffix
        code_to_histories[code].append(hist)
        if check_interventions:
            for position in range(depth):
                flipped = list(hist)
                flipped[position] ^= 1
                changed = quantize_integer(tuple(flipped), k, bits) != code
                expected_change = position >= depth-t
                assert changed == expected_change, (
                    "unexpected intervention visibility", case, hist, position)
                intervention_trials += 1
    unique_codes = len(code_to_histories)
    fibre_size = 1 << (depth - t)
    assert unique_codes == 1 << t
    assert len(suffix_to_code) == unique_codes
    assert {len(fibre) for fibre in code_to_histories.values()} == {fibre_size}
    # An optimal decoder chooses one representative of each fibre; under
    # uniform histories the correct probability is #codes / #histories.
    correct = Fraction(unique_codes, len(histories))
    assert correct == Fraction(1, fibre_size)
    return {
        "campaign": "PEF-01",
        "model": "binary_extreme_digit_self_similar_ifs",
        "k": k,
        "contraction_ratio": f"1/{1 << k}",
        "depth": depth,
        "precision_bits": bits,
        "ifs_dimension_in_this_disjoint_example": {
            "numerator": 1, "denominator": k},
        "recovered_most_recent_history_bits": t,
        "lost_older_history_bits": depth - t,
        "conditional_history_entropy_bits": depth - t,
        "total_histories": 1 << depth,
        "observed_distinct_quantizer_codes": unique_codes,
        "histories_per_observed_code": fibre_size,
        "optimal_exact_history_decoding_probability": {
            "numerator": correct.numerator, "denominator": correct.denominator},
        "minimum_precision_for_full_history": minimum_full_recovery_precision(k, depth),
        "readout_visible_input_positions_zero_based": list(range(depth-t, depth)),
        "readout_hidden_input_positions_zero_based": list(range(0, depth-t)),
        "independent_fraction_oracle_checked": True,
        "exact_intervention_checks": intervention_trials,
        "all_fibres_equal_size": True,
        "no_model_training_or_transformer_measurement": True,
    }


def suite_cases() -> list[Case]:
    return [
        Case(1, 8, 8),
        Case(2, 8, 8),
        Case(2, 8, 15),
        Case(3, 8, 8),
        Case(3, 8, 22),
        Case(2, 12, 8),
    ]


def suite() -> dict:
    cases = [exact_audit(c) for c in suite_cases()]
    return {
        "campaign": "PEF-01",
        "source_papers": [
            {"family": 148,
             "title": "The entropy-rate dimension formula for self-similar measures on the line",
             "url": "https://github.com/openai/math/blob/main/preprints/The-entropy-rate-dimension-formula-for-self-similar-measures-on-the-line-September-24-2026/main.pdf"},
            {"family": 140,
             "title": "Memory and precision in noiseless Gaussian regression",
             "url": "https://github.com/openai/math/blob/main/preprints/Memory-and-precision-in-noiseless-Gaussian-regression-September-27-2026/paper.pdf"},
        ],
        "statement": "finite exact independent toy theorem; no empirical neural claim",
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--no-interventions", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = suite() if args.suite else exact_audit(
        Case(args.k, args.depth, args.bits),
        check_interventions=not args.no_interventions)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
