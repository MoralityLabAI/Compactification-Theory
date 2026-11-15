"""PEF-02S: mixed-depth contraction schedules, exact prefix-coded history.

Chronological k_j determines each bit's binary repetition width. Computes
exact suffix capacity, trajectory separation and sequential floor behavior.
No neural activation data are claimed.
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
class Schedule:
    widths: tuple[int, ...]  # oldest -> newest, each contraction 2^(-width)
    precision_bits: int

    def validate(self) -> None:
        if not 1 <= len(self.widths) <= 12:
            raise ValueError("schedule must have between one and twelve loop steps")
        if any(type(w) is not int or not 1 <= w <= 16 for w in self.widths):
            raise ValueError("each width must be an integer in [1,16]")
        if type(self.precision_bits) is not int or not 0 <= self.precision_bits <= 128:
            raise ValueError("precision must be an integer in [0,128]")


def integer_numerator(history: tuple[int, ...], widths: tuple[int, ...]) -> int:
    if len(history) != len(widths) or any(b not in (0, 1) for b in history):
        raise ValueError("history/schedule mismatch")
    total = shift = 0
    for width, bit in zip(widths, history):
        total += (((1 << width) - 1) * bit) << shift
        shift += width
    return total


def independent_rational_orbit(history: tuple[int, ...],
                               widths: tuple[int, ...]) -> Fraction:
    x = Fraction(0)
    for width, bit in zip(widths, history):
        x = (x + ((1 << width)-1)*bit) / (1 << width)
    return x


def quantized(history: tuple[int, ...], widths: tuple[int, ...],
              precision_bits: int) -> int:
    return (integer_numerator(history, widths) << precision_bits) // (1 << sum(widths))


def stagewise_floor(history: tuple[int, ...], widths: tuple[int, ...],
                    precision_bits: int) -> int:
    if len(history) != len(widths):
        raise ValueError("history/schedule mismatch")
    state = 0
    for width, bit in zip(widths, history):
        state = (state + (((1 << width) - 1) << precision_bits) * bit) // (1 << width)
    return state


def visible_suffix_bits(widths: tuple[int, ...], precision_bits: int) -> int:
    if precision_bits < 0:
        raise ValueError("negative precision")
    count = cumulative = 0
    for width in reversed(widths):
        if cumulative >= precision_bits:
            break
        count += 1
        cumulative += width
    return count


def full_history_precision_threshold(widths: tuple[int, ...]) -> int:
    if not widths:
        raise ValueError("empty schedule")
    return 1 + sum(widths[1:])


def mixed_schedule_gap(widths: tuple[int, ...], t: int) -> Fraction:
    """Sharp inter-suffix gap, excluding unobserved trailing old-history tail.

    gap = 2^(-K_total) + min_{r<=t} (2^(k_latest_r)-2) 2^(-K_r)
    K_r = sum of the r most recent contraction widths.
    """
    if not 1 <= t <= len(widths) or any(k < 1 for k in widths):
        raise ValueError("invalid schedule or suffix size")
    cumulative = 0
    terms: list[Fraction] = []
    for width in tuple(reversed(widths))[:t]:
        cumulative += width
        terms.append(Fraction((1 << width)-2, 1 << cumulative))
    return Fraction(1, 1 << sum(widths)) + min(terms)


def independent_exhaustive_gap(widths: tuple[int, ...], t: int) -> Fraction:
    if not 1 <= len(widths) <= 10 or not 1 <= t <= len(widths):
        raise ValueError("bounded exact oracle needs depth<=10 and nonempty suffix")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for history in product((0, 1), repeat=len(widths)):
        groups[history[len(widths)-t:]].append(integer_numerator(history, widths))
    ordered = sorted((min(v), max(v)) for v in groups.values())
    if any(ordered[i][1] >= ordered[i+1][0] for i in range(len(ordered)-1)):
        raise AssertionError("history suffix hulls overlap")
    gap_integer = min(ordered[i+1][0] - ordered[i][1] for i in range(len(ordered)-1))
    return Fraction(gap_integer, 1 << sum(widths))


def per_step_error_gain(widths: tuple[int, ...]) -> Fraction:
    """The bound for the same absolute additive error after every loop."""
    gain = Fraction(0)
    multiplier = Fraction(1)
    for width in reversed(widths):
        gain += multiplier
        multiplier /= 1 << width
    return gain


def audit(s: Schedule, *, exhaustive: bool = True) -> dict:
    s.validate()
    t = visible_suffix_bits(s.widths, s.precision_bits)
    histories = list(product((0, 1), repeat=len(s.widths)))
    code_group: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    suffix_codes: dict[tuple[int, ...], int] = {}
    for hist in histories:
        x = independent_rational_orbit(hist, s.widths)
        assert x == Fraction(integer_numerator(hist, s.widths),
                             1 << sum(s.widths))
        code = quantized(hist, s.widths, s.precision_bits)
        assert code == ((1 << s.precision_bits)*x.numerator)//x.denominator
        assert stagewise_floor(hist, s.widths, s.precision_bits) == code
        suffix = hist[len(s.widths)-t:] if t else ()
        if suffix in suffix_codes and suffix_codes[suffix] != code:
            raise AssertionError("older hidden bits changed readout")
        suffix_codes[suffix] = code
        code_group[code].append(hist)
    if len(code_group) != 1 << t or len(suffix_codes) != len(code_group):
        raise AssertionError("injected-history suffix code collision")
    if {len(v) for v in code_group.values()} != {1 << (len(s.widths)-t)}:
        raise AssertionError("unequal conditional readout fibres")
    gap = mixed_schedule_gap(s.widths, t) if t else None
    brute = independent_exhaustive_gap(s.widths, t) if exhaustive and t else None
    if brute is not None and brute != gap:
        raise AssertionError("mixed-radix gap is incorrect")
    amplification = per_step_error_gain(s.widths)
    return {
        "chronological_widths_old_to_new": list(s.widths),
        "total_contraction_exponent": sum(s.widths),
        "readout_bits": s.precision_bits,
        "retained_most_recent_inputs": t,
        "distinct_codes": len(code_group),
        "histories_per_code": 1 << (len(s.widths)-t),
        "sharp_full_history_precision": full_history_precision_threshold(s.widths),
        "exact_cross_suffix_gap": str(gap) if gap is not None else None,
        "exact_gap_independently_enumerated": brute is not None,
        "per_step_error_amplification": str(amplification),
        "sharp_analog_per_step_error_threshold": str(gap/(2*amplification)) if gap else None,
        "fixed_floor_per_step_error_sufficient_max": str(
            max(Fraction(0),(gap-Fraction(1,1 << s.precision_bits))/(2*amplification))
        ) if gap else None,
        "stagewise_integer_floor_commutes_with_endpoint_quantization": True,
        "model": "heterogeneous_binary_block_ifs",
    }


def suite() -> dict:
    a = audit(Schedule((2,2,4),6))
    b = audit(Schedule((2,4,2),6))
    c = audit(Schedule((4,2,2),6))
    assert (a["retained_most_recent_inputs"],
            b["retained_most_recent_inputs"],
            c["retained_most_recent_inputs"]) == (2,2,3)
    assert (a["exact_cross_suffix_gap"],
            b["exact_cross_suffix_gap"]) == ("9/256", "57/256")
    return {
        "campaign": "PEF-02S",
        "source_pair": [148, 140],
        "same_total_exponent_same_capacity_unequal_margin": [a,b],
        "same_total_exponent_different_capacity": c,
        "gap_ratio_b_over_a": str(Fraction(57,9)),
        "same_exponent_schedule_does_not_determine_robustness": True,
        "no_trained_model_results": True,
    }


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--widths", default="2,2,4",
                        help="chronological integer contraction exponents")
    parser.add_argument("--bits",type=int,default=6)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    result=suite() if args.suite else audit(
        Schedule(tuple(int(w) for w in args.widths.split(",")),args.bits))
    content=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(content,encoding="utf-8")
    else:
        print(content,end="")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
