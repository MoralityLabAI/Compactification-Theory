"""COL-COMP-01: exact decoder and side-message limits in finite dynamical cores.

Frontiers allow an encoder with access to the true basin. They characterize
information budgets; a deployable encoder must be evaluated separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import gzip
import heapq
import json
from math import gcd, log2
from pathlib import Path

try:
    from .collatz_core import analyze, modular_labels
except ImportError:
    from collatz_core import analyze, modular_labels


def grouped_counts(labels: dict[int, int], observations: dict[int, object]) -> dict:
    if set(labels) != set(observations):
        raise ValueError("every labeled state needs exactly one observation")
    counts = defaultdict(Counter)
    for state, basin in labels.items():
        counts[observations[state]][basin] += 1
    return dict(counts)


def message_success(counts: dict, messages: int) -> int:
    """Best correct count when each observation permits at most M messages."""
    if type(messages) is not int or messages < 1:
        raise ValueError("messages must be a positive integer")
    return sum(sum(sorted(cell.values(), reverse=True)[:messages]) for cell in counts.values())


def huffman_cost(weights) -> int:
    """Minimum weighted length of a binary prefix code, including empty singleton."""
    heap = list(weights)
    if not heap or any(type(x) is not int or x <= 0 for x in heap):
        raise ValueError("positive integer symbol weights are required")
    heapq.heapify(heap)
    cost = 0
    while len(heap) > 1:
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        cost += merged
        heapq.heappush(heap, merged)
    return cost


def conditional_entropy(counts: dict) -> float:
    total = sum(sum(cell.values()) for cell in counts.values())
    return sum(value / total * log2(sum(cell.values()) / value)
               for cell in counts.values() for value in cell.values())


def run_parameter(d: int) -> dict:
    core, quotient = analyze(d), modular_labels(d)
    observations = {n: quotient[n % d] for n in core.states}
    counts = grouped_counts(core.basin, observations)
    gcd_counts = grouped_counts(core.basin, {n: gcd(n, d) for n in core.states})
    cycles = {cycle["minimum"]: cycle["minimum"] for cycle in core.cycles}
    cycle_counts = grouped_counts(cycles, {n: quotient[n % d] for n in cycles})
    total = len(core.states)
    max_basins = max(map(len, counts.values()))
    zero_error_bits = (max_basins - 1).bit_length()
    rows = []
    for bits in range(zero_error_bits + 1):
        messages = 1 << bits
        exact = message_success(counts, messages)
        shuffled = []
        for seed in (11, 29, 47, 71):
            obs = {n: (observations[n], int.from_bytes(hashlib.sha256(
                f"col-side-message-v1:{seed}:{d}:{n}".encode()).digest()[:8], "big") % messages)
                for n in core.states}
            sham_counts = grouped_counts(core.basin, obs)
            shuffled.append({"seed": seed, "correct": message_success(sham_counts, 1),
                             "occupied_cells": len(sham_counts)})
        rows.append({"extra_fixed_bits": bits, "messages_per_quotient": messages,
                     "uniform_core_correct": exact, "uniform_core_total": total,
                     "uniform_core_fraction": str(Fraction(exact, total)),
                     "uniform_cycles_correct": message_success(cycle_counts, messages),
                     "uniform_cycles_total": len(cycles),
                     "hash_side_messages": shuffled})
    return {"parameter": d, "states": total, "cycles": len(cycles),
            "quotient_classes_on_core": len(counts),
            "gcd_only_correct": message_success(gcd_counts, 1),
            "quotient_only_correct": message_success(counts, 1),
            "quotient_only_uniform_cycles_fraction": str(Fraction(len(counts), len(cycles))),
            "max_basins_in_one_quotient": max_basins,
            "minimum_extra_fixed_bits_for_zero_error": zero_error_bits,
            "conditional_basin_entropy_bits_float": round(conditional_entropy(counts), 12),
            "optimal_prefix_weighted_bit_sum": sum(huffman_cost(cell.values()) for cell in counts.values()),
            "optimal_prefix_mean_bits_fraction": str(Fraction(
                sum(huffman_cost(cell.values()) for cell in counts.values()), total)),
            "frontier": rows,
            "cells": [{"quotient": q, "states": sum(cell.values()),
                       "basins": [{"basin_minimum": b, "states": c} for b, c in sorted(cell.items())]}
                      for q, cell in sorted(counts.items())]}


def run(max_d: int = 1025, anchor: int = 2021) -> dict:
    if type(max_d) is not int or max_d < 1 or max_d > 10001:
        raise ValueError("max_d must be in [1,10001]")
    parameters = [d for d in range(1, max_d + 1, 4) if gcd(d, 6) == 1]
    results = [run_parameter(d) for d in parameters]
    total = sum(row["states"] for row in results)
    correct = sum(row["quotient_only_correct"] for row in results)
    return {"schema": "col-quotient-frontier-v1", "evidence_class": "exact finite domain",
            "encoder_access": "oracle basin label; computational cost of encoding excluded",
            "selection_exposure": "exploratory complete finite sweep; all labels exposed",
            "n_units": len(parameters), "n_cases": total, "parameters": parameters,
            "core_sha256": hashlib.sha256(Path(__file__).with_name("collatz_core.py").read_bytes()).hexdigest(),
            "summary": {"quotient_only_correct": correct, "total": total,
                        "micro_fraction": str(Fraction(correct, total)),
                        "macro_fraction": str(sum((Fraction(row["quotient_only_correct"], row["states"])
                                                  for row in results), Fraction()) / len(results)),
                        "parameters_with_quotient_collision": sum(row["minimum_extra_fixed_bits_for_zero_error"] > 0
                                                                  for row in results),
                        "fixed_side_bits_histogram": dict(sorted(Counter(
                            row["minimum_extra_fixed_bits_for_zero_error"] for row in results).items()))},
            "results": results,
            "known_anchor": {"selection": "previously inspected stress case, outside sweep",
                             "result": run_parameter(anchor)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-d", type=int, default=1025)
    parser.add_argument("--anchor", type=int, default=2021)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    receipt = run(args.max_d, args.anchor)
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.check:
        data = args.check.read_bytes()
        if args.check.suffix == ".gz":
            data = gzip.decompress(data)
        if json.loads(serialized) != json.loads(data):
            raise SystemExit("receipt replay differs")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = serialized.encode()
        args.out.write_bytes(gzip.compress(data, mtime=0) if args.out.suffix == ".gz" else data)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        brief = {key: value for key, value in receipt.items() if key != "results"}
        args.summary.write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_units": receipt["n_units"], "n_cases": receipt["n_cases"],
                      "summary": receipt["summary"],
                      "receipt_replay": "passed" if args.check else "generated"}))


if __name__ == "__main__":
    main()
