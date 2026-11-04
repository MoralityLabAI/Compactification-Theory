"""Frozen generator-disjoint pointer workloads for real LOTUS latent-state probes.

This tool has no model dependencies. Split membership, instance order, trace,
and schema are deterministic functions of the saved settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..spec import case, digest

FAMILIES = (("train", "random_function"),
            ("calibration", "permutation"),
            ("test", "long_cycle"),
            ("test", "sink"))


def build_roster(*, seed: int = 20261104, n: int = 6, steps: int = 16,
                 train: int = 80, calibration: int = 24, test_per_family: int = 24):
    if not 2 <= n <= 16 or not 1 <= steps <= 64:
        raise ValueError("n and steps outside pointer-machine limits")
    if min(train, calibration, test_per_family) < 1:
        raise ValueError("positive number of cases required in each split")
    sizes = {"train": train, "calibration": calibration, "test": test_per_family}
    selected = []
    occupied = set()
    for split, family in FAMILIES:
        count = sizes[split]
        accepted = 0
        idx = 0
        while accepted < count:
            # The finite set of sink maps has at most n*n map/start pairs.
            if idx >= 250000:
                raise ValueError("cannot find enough distinct (mapping,start) pairs")
            task = case(family, seed, idx, n=n, steps=steps)
            idx += 1
            key = (task.successors, task.initial)
            if key in occupied:
                continue
            occupied.add(key)
            selected.append({"id": f"{split}/{family}/{accepted}",
                             "split": split, "family": family,
                             "successors": list(task.successors),
                             "initial": task.initial, "trace": task.trace()})
            accepted += 1
    params = {"seed": seed, "n": n, "steps": steps, "train": train,
              "calibration": calibration, "test_per_family": test_per_family}
    return {"schema_version": "algernon-lotus-pointer-roster-v1",
            "generator": "research.algernon.circuit_mirror.spec.case",
            "parameters": params,
            "roster_sha256": digest({"parameters": params, "cases": selected}),
            "cases": selected}


def validate_roster(roster):
    if roster.get("schema_version") != "algernon-lotus-pointer-roster-v1":
        raise ValueError("unsupported roster schema")
    cases = roster["cases"]
    if roster.get("roster_sha256") != digest({"parameters":roster["parameters"], "cases":cases}):
        raise ValueError("roster hash mismatch")
    seen = set()
    for row in cases:
        key = (tuple(row["successors"]), row["initial"])
        if key in seen:
            raise ValueError("overlapping split or duplicate pointer instance")
        seen.add(key)
        state = row["initial"]
        expected = [state]
        for _ in range(roster["parameters"]["steps"]):
            state = row["successors"][state]
            expected.append(state)
        if row["trace"] != expected:
            raise ValueError("invalid reference trace")
        if not (row["split"], row["family"]) in FAMILIES:
            raise ValueError("unexpected family assignment")
    return len(cases)


def prompt_for(row):
    """No target trace or answer appears anywhere in this model input."""
    mapping = ", ".join(f"{i}->{j}" for i, j in enumerate(row["successors"]))
    steps = len(row["trace"]) - 1
    return ("A machine has numbered states and one deterministic successor per state. "
            f"The successor mapping is {mapping}. Starting at state {row['initial']}, "
            f"follow the successor exactly {steps} times. "
            f"Which state is reached after {steps} steps?\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--calibration", type=int, default=24)
    ap.add_argument("--test-per-family", type=int, default=24)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    obj = build_roster(seed=a.seed, n=a.nodes, steps=a.steps, train=a.train,
                       calibration=a.calibration, test_per_family=a.test_per_family)
    validate_roster(obj)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(obj, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(f"frozen {len(obj['cases'])} cases; {obj['roster_sha256']}; {a.out}")


if __name__ == "__main__":
    main()
