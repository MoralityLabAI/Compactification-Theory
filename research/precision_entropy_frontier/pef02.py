"""PEF-02: sharp suffix gaps, recurrent floor commutation, and prior leakage.

Extends PEF-01, using only exact integers and fractions. Every statement is
about a finite affine recurrence, not a trained transformer. Probe priors and
quantizer gauge are separately auditable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

from pef01 import quantize_integer, rational_orbit, retained_bits


def validate(k: int, depth: int, t: int) -> None:
    if type(k) is not int or k < 1 or k > 16:
        raise ValueError("k must be in [1,16]")
    if type(depth) is not int or not 1 <= depth <= 12:
        raise ValueError("depth must be in [1,12]")
    if type(t) is not int or not 1 <= t <= depth:
        raise ValueError("retained suffix length must be in [1,depth]")


def minimum_suffix_gap(k: int, depth: int, t: int) -> Fraction:
    """Min distance between real states from distinct t-suffix classes."""
    validate(k, depth, t)
    base = 1 << k
    return Fraction(base - 2, base**t) + Fraction(1, base**depth)


def exhaustive_suffix_gap(k: int, depth: int, t: int) -> Fraction:
    """Independent exact integer exhaustive oracle, no use of closed formula."""
    validate(k, depth, t)
    base = 1 << k
    classes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for history in product((0, 1), repeat=depth):
        numerator = (base - 1) * sum(b*base**i for i, b in enumerate(history))
        classes[history[depth-t:]].append(numerator)
    spans = sorted((min(v), max(v)) for v in classes.values())
    if any(spans[i+1][0] <= spans[i][1] for i in range(len(spans)-1)):
        raise AssertionError("distinct suffix hulls overlap")
    minimum = min(spans[i+1][0] - spans[i][1] for i in range(len(spans)-1))
    return Fraction(minimum, base**depth)


def geometric_error_gain(k: int, depth: int) -> Fraction:
    """End-state gain for bounded adversarial additive errors at every step."""
    if k < 1 or depth < 0:
        raise ValueError("invalid contraction or depth")
    base = 1 << k
    return sum((Fraction(1, base**j) for j in range(depth)), Fraction(0))


def robustness(k: int, depth: int, precision_bits: int) -> dict:
    """Sufficient exact noise bounds; raw analog threshold is sharp.

    If each stage has absolute additive error <=eta, end error <=eta*S.
    Analog robust suffix decoding: 2*eta*S < gap (sharp).
    Fixed final floor quantizer sufficient guard: 2*eta*S+2^-B <= gap.
    """
    if precision_bits < 1:
        raise ValueError("nontrivial suffix proof requires at least one bit")
    t = retained_bits(k, depth, precision_bits)
    validate(k, depth, t)
    gap = minimum_suffix_gap(k, depth, t)
    step_gain = geometric_error_gain(k, depth)
    width = Fraction(1, 2**precision_bits)
    analog = gap / (2 * step_gain)
    quantized = max(Fraction(0), (gap - width) / (2 * step_gain))
    return {
        "k": k,
        "depth": depth,
        "readout_bits": precision_bits,
        "retained_suffix_bits": t,
        "minimum_cross_suffix_distance": str(gap),
        "cumulative_step_error_gain": str(step_gain),
        "analog_step_error_threshold_strict": str(analog),
        "quantized_step_error_sufficient_max": str(quantized),
        "fixed_point_bin_width": str(width),
        "raw_analog_bound_is_sharp": True,
        "floor_quantized_bound_is_sufficient_only": True,
    }


def recurrent_floor(history: tuple[int, ...], k: int, precision_bits: int) -> int:
    """Perform floor rounding after *every* step of the finite recurrence."""
    if k < 1 or precision_bits < 0 or any(b not in (0, 1) for b in history):
        raise ValueError("invalid recurrence inputs")
    base = 1 << k
    injection = (base - 1) << precision_bits
    state = 0
    for bit in history:
        state = (state + injection * bit) // base
    return state


def nearest_round_at_each_step(history: tuple[int, ...], k: int, precision_bits: int) -> int:
    """Half-up rounding per step. Included as a counterexample control."""
    base = 1 << k
    injection = (base - 1) << precision_bits
    state = 0
    for bit in history:
        state = (2*(state + injection*bit) + base) // (2*base)
    return state


def nearest_round_once(history: tuple[int, ...], k: int, precision_bits: int) -> int:
    base = 1 << k
    numerator = (base - 1) * sum(b*base**i for i, b in enumerate(history))
    denominator = base**len(history)
    return (2 * (1 << precision_bits)*numerator + denominator) // (2*denominator)


def affine_gauge_codes(history: tuple[int, ...], k: int, bits: int,
                       scale: Fraction, shift: Fraction) -> dict:
    """Readout after an affine change, with and without range calibration.

    B is fractional-bit precision only; uncalibrated scaled states can need
    additional integer bits, so no fixed-word-size capacity gain is implied.
    """
    if bits < 0 or scale <= 0:
        raise ValueError("invalid scale or readout bits")
    x = rational_orbit(history, k)
    transformed = scale*x + shift
    raw_code = (transformed * (1 << bits)).numerator // (transformed * (1 << bits)).denominator
    normalized = (transformed - shift) / scale
    normalized_code = ((1 << bits)*normalized).numerator // ((1 << bits)*normalized).denominator
    assert normalized == x
    assert normalized_code == quantize_integer(history, k, bits)
    return {
        "raw_absolute_code": raw_code,
        "calibrated_code": normalized_code,
        "untransformed_code": quantize_integer(history, k, bits),
        "calibration_range": [str(shift), str(shift + scale)],
    }


def markov_probability(history: tuple[int, ...], stay: Fraction) -> Fraction:
    if not 0 <= stay <= 1:
        raise ValueError("stay probability must be in [0,1]")
    if not history:
        return Fraction(1)
    p = Fraction(1, 2)
    for i in range(1, len(history)):
        p *= stay if history[i] == history[i-1] else (1-stay)
    return p


def markov_prior_audit(k: int, depth: int, bits: int, stay: Fraction) -> dict:
    """Exact Bayes MAP using the same quantized channel and Markov sources."""
    if depth < 1 or depth > 12 or k < 1 or bits < 0 or not 0 <= stay <= 1:
        raise ValueError("invalid Markov audit")
    t = retained_bits(k, depth, bits)
    outcomes: dict[int, list[Fraction]] = defaultdict(list)
    align = Fraction(0)
    edge = depth-t-1
    for hist in product((0, 1), repeat=depth):
        p = markov_probability(hist, stay)
        outcomes[quantize_integer(hist, k, bits)].append(p)
        if t > 0 and edge >= 0 and hist[edge] == hist[edge+1]:
            align += p
    maximum_success = sum((max(probs) for probs in outcomes.values()), Fraction(0))
    best = max(stay, 1-stay)
    prediction = (best**(depth-t) if t >= 1
                  else Fraction(1,2)*best**(depth-1))
    if maximum_success != prediction:
        raise AssertionError("observational Markov MAP derivation failed")
    if t > 0 and edge >= 0 and align != stay:
        raise AssertionError("nearest hidden bit prior correlation mismatch")
    return {
        "depth": depth,
        "k": k,
        "precision_bits": bits,
        "retained_suffix_bits": t,
        "markov_stay_probability": str(stay),
        "observed_codes": len(outcomes),
        "optimal_full_history_map_success": str(maximum_success),
        "uniform_history_success": str(Fraction(1, 2**(depth-t))),
        "hidden_edge_observational_agreement": str(align) if edge >= 0 and t > 0 else None,
        "hidden_edge_direct_causal_readout_effect": 0,
        "conditional_entropy_bits_symbolic": (
            f"{depth-t} * h2({stay})" if t >= 1
            else f"1 + {depth-1} * h2({stay})"
        ),
        "no_new_successful_history_communication": True,
    }


def case_report(k: int, depth: int, bits: int) -> dict:
    info = robustness(k, depth, bits)
    t = info["retained_suffix_bits"]
    brute = exhaustive_suffix_gap(k, depth, t)
    if brute != minimum_suffix_gap(k, depth, t):
        raise AssertionError("exact margin formula conflicts with exhaustive search")
    info["independent_exhaustive_gap_checked"] = True
    info["stagewise_floor_commutes_all_histories"] = all(
        recurrent_floor(hist, k, bits) == quantize_integer(hist, k, bits)
        for hist in product((0, 1), repeat=depth))
    if not info["stagewise_floor_commutes_all_histories"]:
        raise AssertionError("stagewise floor did not commute")
    return info


def suite() -> dict:
    rows = [case_report(*case) for case in
            [(1, 8, 8), (2, 8, 8), (3, 8, 8), (2, 12, 8), (2, 8, 15)]]
    markov = markov_prior_audit(2, 8, 8, Fraction(19, 20))
    return {
        "campaign": "PEF-02",
        "parent": "PEF-01",
        "model": "finite_exact_affine_ifs",
        "independent_brute_gap_and_staged_floor": rows,
        "source_prior_example": markov,
        "nearest_rounding_noncommutation_witness": {
            "history_oldest_first": [1,0,0],
            "k": 2, "precision_bits": 3,
            "stagewise_nearest": nearest_round_at_each_step((1,0,0), 2, 3),
            "endpoint_nearest": nearest_round_once((1,0,0), 2, 3),
        },
        "fixed_readout_model_distinct_from_IEEE_floating_point": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--stay", default="19/20")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = suite() if args.suite else {
        "robustness": case_report(args.k, args.depth, args.bits),
        "source_prior": markov_prior_audit(
            args.k, args.depth, args.bits, Fraction(args.stay)),
    }
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
