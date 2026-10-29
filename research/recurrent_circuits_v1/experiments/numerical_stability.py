"""CPU-only numerical reach experiment for RC-01 restored NAND rings.

This is scalar software emulation, not evidence about CUDA kernels or GPU timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

HORIZONS = (32, 128, 512, 2048)
WIDTHS = (4, 16)
IDLE_WIDTHS = (16, 64, 256)
PRECISIONS = ("float32", "bfloat16", "float16")


def cast(x: float, precision: str) -> float:
    if precision == "float32":
        return struct.unpack("!f", struct.pack("!f", x))[0]
    if precision == "float16":
        return struct.unpack("!e", struct.pack("!e", x))[0]
    if precision == "bfloat16":
        bits = struct.unpack("!I", struct.pack("!f", x))[0]
        # Round to nearest, ties to even before retaining the top 16 bits.
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return struct.unpack("!f", struct.pack("!I", bits & 0xFFFF0000))[0]
    raise ValueError(precision)


def relu(x, p):
    return cast(max(0.0, x), p)


def restore_scalar(x, p):
    a = relu(cast(x - 0.25, p), p)
    b = relu(cast(x - 0.75, p), p)
    return cast(cast(2.0 * a, p) - cast(2.0 * b, p), p)


def nand_step(x, y, precision, gate_restoration):
    s = relu(cast(cast(x + y, precision) - 1.0, precision), precision)
    if gate_restoration:
        r = restore_scalar(s, precision)
    else:
        r = s
    return cast(1.0 - r, precision)


def initial_states(width):
    if width == 4:
        return [(seed, tuple((seed >> i) & 1 for i in range(width))) for seed in range(1 << width)]
    seeds = (0, 1, 2, 10, 11)
    result = []
    for seed in seeds:
        rng = random.Random(seed * 1009 + width)
        result.append((seed, tuple(rng.randrange(2) for _ in range(width))))
    return result


def run_ring(width, seed, initial, precision, noise, gate_restoration, max_horizon):
    actual = tuple(cast(float(v), precision) for v in initial)
    symbolic = initial
    first_bad = None
    for t in range(max_horizon):
        expected = tuple(1 - symbolic[i] * symbolic[(i - 1) % width] for i in range(width))
        out = []
        for i in range(width):
            value = nand_step(actual[i], actual[(i - 1) % width], precision, gate_restoration)
            if noise:
                perturb = 0.125  # coherent corner of the declared bounded-error box
                value = cast(value + perturb, precision)
            out.append(value)
        actual = tuple(out)
        decoded = tuple(int(v >= 0.5) for v in actual)
        if first_bad is None and decoded != expected:
            first_bad = {"transition": t + 1, "instruction": next(i for i, (a, b) in enumerate(zip(decoded, expected)) if a != b),
                         "actual_bit": decoded[next(i for i, (a, b) in enumerate(zip(decoded, expected)) if a != b)],
                         "oracle_bit": expected[next(i for i, (a, b) in enumerate(zip(decoded, expected)) if a != b)]}
        symbolic = expected
    return first_bad


def run_idle(width, precision, restore_idle, max_horizon):
    rng = random.Random(914 + width)
    initial = tuple(rng.randrange(2) for _ in range(width))
    actual = tuple(cast(float(v), precision) for v in initial)
    symbolic = initial
    first_bad = None
    for t in range(max_horizon):
        expected = list(symbolic)
        out = list(actual)
        for i in range(width):
            active = i % 4 == t % 4
            if active:
                expected[i] = 1 - symbolic[i] * symbolic[(i - 1) % width]
                out[i] = nand_step(actual[i], actual[(i - 1) % width], precision, True)
            elif restore_idle:
                out[i] = restore_scalar(actual[i], precision)
            out[i] = cast(out[i] + 0.125, precision)
        actual = tuple(out)
        symbolic = tuple(expected)
        decoded = tuple(int(v >= 0.5) for v in actual)
        if first_bad is None and decoded != symbolic:
            i = next(j for j, (a, b) in enumerate(zip(decoded, symbolic)) if a != b)
            first_bad = {"transition": t + 1, "instruction": i, "active": i % 4 == t % 4,
                         "actual_bit": decoded[i], "oracle_bit": symbolic[i]}
            return first_bad
    return first_bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results" / "numerical_stability_cpu.json")
    args = parser.parse_args()
    started = time.perf_counter()
    cases = []
    for width in WIDTHS:
        for precision in PRECISIONS:
            for noise, gate_restoration in ((False, True), (True, True), (True, False)):
                first = []
                states = initial_states(width)
                for seed, state in states:
                    failure = run_ring(width, seed, state, precision, noise, gate_restoration, max(HORIZONS))
                    if failure:
                        first.append({"seed": seed, **failure})
                earliest = min(first, key=lambda x: (x["transition"], x["seed"])) if first else None
                cases.append({"experiment": "nand_ring", "width": width, "precision": precision,
                              "bounded_perturbation": noise, "perturbation_pattern": "all coordinates +1/8 each transition" if noise else "none",
                              "gate_restoration": gate_restoration,
                              "initial_states": len(states),
                              "correct_through_horizon": {str(h): sum(1 for f in first if f["transition"] > h) + (len(states)-len(first))
                                                          for h in HORIZONS},
                              "first_incorrect": earliest})
    for width in IDLE_WIDTHS:
        for precision in PRECISIONS:
            for restore_idle in (True, False):
                cases.append({"experiment": "idle_memory", "width": width, "precision": precision,
                              "idle_restoration": restore_idle,
                              "perturbation_per_transition": 0.125,
                              "schedule": "one quarter of cells active per step, cells cycle every four steps",
                              "first_incorrect": run_idle(width, precision, restore_idle, max(HORIZONS))})
    sources = [HERE / "THEORY.md", HERE / "EXPERIMENTS.md", HERE / "reference.py", Path(__file__)]
    result = {
        "schema_version": 1,
        "evidence": "cpu_scalar_software_emulation",
        "device": "CPU; no CUDA/PyTorch installed in available Python",
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "horizons": list(HORIZONS), "ring_widths": list(WIDTHS), "idle_widths": list(IDLE_WIDTHS), "precisions": list(PRECISIONS),
        "oracle": "independent symbolic NAND transition recomputed from integer bits at every step",
        "decoder": "threshold at 0.5 after every transition; first differing lane recorded",
        "gate": "T H E O R Y.md clipped-ReLU restoration, operation-level rounding emulation",
        "theorem_reference": "THEORY.md Theorem 5: rho<1, eta<=(1-rho)r, 2r<Delta; explicit NAND permits additional error <=1/8/update",
        "claims": {"gpu_device_precision": False, "uniform_theorem_proved_by_measurement": False},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(f"cases={len(cases)} runtime_s={result['runtime_seconds']}")

if __name__ == "__main__":
    main()
