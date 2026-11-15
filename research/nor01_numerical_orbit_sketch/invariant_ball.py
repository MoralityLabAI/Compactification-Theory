"""NOR-04: rigorous local nonlinear stability of the trained GRU.

Certify a nonempty invariant Euclidean ball for G=F^12 and a uniform
norm bound K<1 on DG throughout that ball. The result is a Banach
fixed-point certificate for the *actual nonlinear iterated map G*.
It does not claim a basin that includes unrelated RRC-25 input cases.

All operational bounds come from outward-rounded rational intervals
and the source-locked existing RRC-25 recurrent checkpoint.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

from trained_gru_cert import (
    Interval, N, build_certificate, gates, get_weights, sqrt_upper, ZERO,
)
from research.rrc_compactification import bench25


# Preselected from the unmodified checkpoint's zero-input nominal fixed-point
# trajectory, iterated to convergence and Newton-polished in double precision.
# Rigorous existence and uniqueness are established only by the rational
# invariant-ball and contraction gates below.
CENTER_DECIMALS = (
    "-0.46412196023148555",
    "0.93954570430325590",
    "0.99399466867049846",
    "0.20452417668606748",
    "0.79833877601235637",
    "-0.58436562208826126",
    "0.83985584817478498",
    "-0.90808625028131418",
    "0.85066367836037171",
    "0.72318142076044045",
    "-0.46744168835369559",
    "-0.36109479031967046",
)
CENTER = tuple(Q(z) for z in CENTER_DECIMALS)
RADIUS = Q(1, 10**11)
DEPTH = 12


def centered_ball(c, radius):
    if radius <= 0 or len(c) != N:
        raise ValueError("positive radius and 12 hidden coordinates required")
    if any(abs(x) + radius >= 1 for x in c):
        raise ValueError("ball must lie strictly inside invariant GRU cube")
    return [Interval(x - radius, x + radius) for x in c]


def fixed_ball_certificate() -> dict:
    ck, digests = bench25.checkpoints()
    recurrent = ck["recurrent"]
    weights = get_weights(recurrent)
    box = centered_ball(CENTER, RADIUS)

    # Uniformly bound ||D G(h)||_2 over every h in B(CENTER,RADIUS):
    # the interval box contains the ball and is propagated through all
    # true changing nonlinear GRU Jacobians.
    deriv = build_certificate(
        weights, box, length=DEPTH, label="nor04_invariant_ball")
    K = Q(deriv["block_operator_frobenius_upper_fraction"])

    # Independently enclose G(c) with an exact rational center.
    point = [Interval.point(c) for c in CENTER]
    for _ in range(DEPTH):
        point = gates(weights, point)[0]
    diffs = [max(abs(x.lo - c), abs(x.hi - c))
             for x,c in zip(point, CENTER, strict=True)]
    residual = sqrt_upper(sum((x*x for x in diffs), ZERO), 10**24)

    # If s+K R <= R and K<1, then G maps this ball into itself:
    # ||G(x)-c|| <= ||G(x)-G(c)||+||G(c)-c|| <= KR+s.
    # Banach's contraction theorem yields a unique fixed point of G
    # in the closed ball, with all-depth convergence rate K**m.
    self_map = residual + K * RADIUS <= RADIUS
    positive = K < 1 and self_map
    return {
        "schema": "nor04.trained_gru.invariant_ball.v1",
        "checkpoint_sha256": digests["recurrent"],
        "checkpoint_parameters": recurrent["parameters"],
        "source_hashes_verified": recurrent["source_sha256"],
        "center_exact_rationals": [str(x) for x in CENTER],
        "center_literals": list(CENTER_DECIMALS),
        "radius_fraction": str(RADIUS),
        "depth_per_nonlinear_block": DEPTH,
        "operator_upper_fraction": str(K),
        "operator_upper_float": float(K),
        "center_residual_upper_fraction": str(residual),
        "center_residual_upper_float": float(residual),
        "invariance_margin_fraction": str(RADIUS - residual - K*RADIUS),
        "strict_contraction": K < 1,
        "closed_ball_invariant": self_map,
        "banach_fixed_point_certificate": positive,
        "fixed_point_distance_from_center_upper_fraction": (
            str(residual / (1-K)) if K < 1 else None),
        "proof_scope": "unique fixed point of true zero-input F^12 inside certified ball; all-depth F^(12m) convergence for starts in same ball",
        "precision": "decimal-rational checkpoint weights, adaptive Taylor enclosures, 32-digit outward interval arithmetic",
        "block_disk_upper": deriv["disk_q_float"],
        "block_operator_derivative_interval_method": deriv["disk_method"],
        "analytic_hessian_upper": deriv["global_one_step_hessian_upper"],
        "anchor_box_max_radius": deriv["anchor_box_max_radius"],
        "fixed_point_of_one_step_F_proved": False,
        "trained_source_family_in_basin_proved": False,
        "native_proof_ranking_gain_proved": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    receipt = fixed_ball_certificate()
    if args.check:
        assert receipt["checkpoint_sha256"] == (
            "54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b")
        assert receipt["banach_fixed_point_certificate"], receipt
        assert receipt["operator_upper_float"] < 1
        assert receipt["center_residual_upper_float"] < float(RADIUS)
    data = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
