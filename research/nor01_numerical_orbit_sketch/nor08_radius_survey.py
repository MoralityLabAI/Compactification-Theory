"""NOR-08: frozen squared-family survey against the SECOND certified attractor.

All candidate crossing times are FP64 diagnostics. Candidate invariant
ball radii are admitted only by exact Fraction checks of the NOR-07S
stationary-center Volterra tube, using the original source-locked GRU.
The subsequent NOR-08 capture pass will require a separate rational
state-error enclosure along the original squared source continuation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import nor07_stationary_tube as tube
from research.nor01_numerical_orbit_sketch import trained_gru_cert as native

SOURCE_SHA = survey.KNOWN_SHA
REFERENCE = HERE / "results" / "nor07_two_attractors_summary.json"
RADII = tuple(Q(1, 10 ** p) for p in (20, 16, 12, 10, 8, 7, 6, 5, 4, 3))
HORIZON = 3600
STRIDE = 12


def source_lock():
    checkpoint, digests = bench25.checkpoints()
    assert digests["recurrent"] == SOURCE_SHA
    receipt = json.loads(REFERENCE.read_text())
    assert receipt["original_checkpoint_sha256"] == SOURCE_SHA
    assert receipt["two_locally_attracting_fixed_points_certified"]
    center = tuple(Q(x) for x in receipt["second_attractor"]["center_decimals"])
    return checkpoint["recurrent"], center


def independent_radius_sweep(ck, center):
    weights = native.get_weights(ck)
    records = []
    for radius in RADII:
        try:
            result = tube.stationary_tube(weights, center, radius, 24)
            K = Q(result["block_operator_norm_upper_fraction"])
            residual = Q(result["center_residual_upper_fraction"])
            margin = radius - K * radius - residual
            records.append({
                "radius_fraction": str(radius),
                "K_fraction": str(K),
                "s_fraction": str(residual),
                "margin_fraction": str(margin),
                "radius_float": float(radius),
                "K": float(K),
                "positive_invariance": K < 1 and margin >= 0,
                "tube_method": result["certification_method"],
                "status": "certified" if (K < 1 and margin >= 0) else "inconclusive_bound",
            })
        except (ArithmeticError, OverflowError, ValueError):
            # A large-radius quadratic envelope may grow enormous even
            # after sound outward rounding. Such a proof attempt is
            # inconclusive; it must never count as a contracting ball.
            records.append({
                "radius_fraction": str(radius),
                "radius_float": float(radius),
                "K_fraction": None, "s_fraction": None, "margin_fraction": None,
                "K": None, "positive_invariance": False,
                "status": "inconclusive_unbounded_envelope",
            })
    return records


def squared_measured_horizon(ck, center, max_steps):
    if max_steps < 24 or max_steps > 12000:
        raise ValueError("bounded horizon 24..12000")
    cases = survey.public_cases()
    candidate = next(c for c in cases if c.family == "squared")
    hidden, parity, count = survey.prefix_hidden(candidate, ck)
    crossed = {str(r): None for r in RADII}
    snapshot = []
    for step in range(max_steps + 1):
        distance = math.sqrt(math.fsum(
            (v - float(c)) ** 2 for v, c in zip(hidden, center, strict=True)))
        for radius in RADII:
            key = str(radius)
            if crossed[key] is None and distance < float(radius):
                crossed[key] = step
        if step in (0, 12, 96, 240, 480, 720, 960, 1200, 1440, 1680,
                    1920, 2400, 2880, 3360, max_steps):
            snapshot.append({"step": step, "distance_to_second_center": distance})
        if step < max_steps:
            hidden = native.native_zero_step(hidden, ck)
    return {"source_family": candidate.family,
            "source_native_logits_parity": parity,
            "header_count": count,
            "candidate_crossings_fp64_only": crossed,
            "sampled_distances": snapshot,
            "final_center_distance_fp64": distance,
            "interval_certified_entry": False}


def run(max_steps=HORIZON, scan=True):
    ck, center = source_lock()
    return {
        "schema": "nor08.second_basin.radius_survey.v1",
        "source_checkpoint_sha256": SOURCE_SHA,
        "target_source": "squared NOR06 original nine-row public input",
        "certified_center_radius": str(Q(1, 10 ** 20)),
        "radius_sweep": independent_radius_sweep(ck, center) if scan else [],
        "measured": squared_measured_horizon(ck, center, max_steps),
        "max_steps": max_steps,
        "two_attractors_certified": True,
        "source_to_second_basin_certified": False,
        "scope": "Exact-rational local ball radii with FP64 input trajectory crossing diagnostics; no source capture claimed",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=HORIZON)
    p.add_argument("--no-radius-sweep", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    result = run(args.max_steps, scan=not args.no_radius_sweep)
    if args.check:
        assert result["source_checkpoint_sha256"] == SOURCE_SHA
        assert result["measured"]["source_family"] == "squared"
        assert result["measured"]["source_native_logits_parity"] < 1e-11
        assert result["source_to_second_basin_certified"] is False
        assert result["radius_sweep"] or args.no_radius_sweep
        for row in result["radius_sweep"]:
            if row["K_fraction"] is None:
                assert not row["positive_invariance"]
                continue
            K, residual, radius, margin = map(Q, (
                row["K_fraction"], row["s_fraction"],
                row["radius_fraction"], row["margin_fraction"]))
            assert radius - residual - K * radius == margin
            assert row["positive_invariance"] == (K < 1 and margin >= 0)
    blob = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob, end="")


if __name__ == "__main__":
    main()
