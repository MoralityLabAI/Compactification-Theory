"""NOR-05: certify period-one status and full-step convergence for trained GRU.

NOR-04 gives a unique fixed point p of G=F^12 inside a rigorously
certified ball B(c,R). This module proves F(p) remains in the same ball,
then uses G(F(p))=F(G(p)) and uniqueness to conclude F(p)=p.

This replaces a possible 12-cycle with a bona fide fixed point of
the original nonlinear one-step zero-input GRU, on the same tiny basin.
The proof is conditional only on exact-rational inequalities checked
here and the standard Banach fixed-point and mean-value theorems.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from invariant_ball import CENTER, RADIUS, centered_ball, fixed_ball_certificate
from trained_gru_cert import Interval, N, build_certificate, gates, get_weights, sqrt_upper, ZERO
from research.rrc_compactification import bench25


def point_residual(weights, center, steps: int):
    h = [Interval.point(c) for c in center]
    for _ in range(steps):
        h = gates(weights, h)[0]
    d = [max(abs(z.lo - c), abs(z.hi - c))
         for z,c in zip(h, center, strict=True)]
    return sqrt_upper(sum((v*v for v in d), ZERO), 10**24)


def period_one_certificate() -> dict:
    earlier = fixed_ball_certificate()
    if not earlier["banach_fixed_point_certificate"]:
        raise AssertionError("NOR-04 invariant ball prerequisite failed")
    ck, _ = bench25.checkpoints()
    weights = get_weights(ck["recurrent"])

    K12 = Q(earlier["operator_upper_fraction"])
    s12 = Q(earlier["center_residual_upper_fraction"])
    dstar = s12 / (1 - K12)
    one_step_box = centered_ball(CENTER, RADIUS)
    local = build_certificate(
        weights, one_step_box, length=1, label="nor05_one_step_local")
    L1 = Q(local["block_operator_frobenius_upper_fraction"])
    s1 = point_residual(weights, CENTER, 1)

    # p is the unique G fixed point of NOR-04, at distance <= dstar from c.
    # The segment c--p lies in the certified ball. Mean value gives
    # ||F(p)-c|| <= ||F(c)-c|| + L1*||p-c|| <= s1+L1*dstar.
    # This is strictly inside NOR-04's invariant ball. F(p) is also a
    # fixed point of G (since F commutes with its 12th iterate),
    # so uniqueness of G's ball-fixed-point forces F(p)=p.
    mapped_fixed_point_inside = s1 + L1 * dstar < RADIUS

    # Global derivative over invariant unit cube for a conservative
    # all-time bound at each residue class t mod 12.
    cube = [Interval(Q(-1), Q(1)) for _ in range(N)]
    global_step = build_certificate(
        weights, cube, length=1, label="nor05_one_step_global_cube")
    L = Q(global_step["block_operator_frobenius_upper_fraction"])
    all_steps_factor = max(Q(1), L)**11

    return {
        "schema": "nor05.trained_gru.one_step_fixed_point.v1",
        "checkpoint_sha256": earlier["checkpoint_sha256"],
        "checkpoint_parameters": 1000,
        "all_depth_block_prerequisite": {
            "K12_fraction": str(K12),
            "R_fraction": str(RADIUS),
            "center_residual_12_fraction": str(s12),
            "fixed_point_distance_bound_fraction": str(dstar),
        },
        "one_step_local_operator_norm_upper_fraction": str(L1),
        "one_step_residual_upper_fraction": str(s1),
        "one_step_fixed_point_image_distance_upper_fraction": str(s1+L1*dstar),
        "one_step_fixed_point_in_original_ball": mapped_fixed_point_inside,
        "one_step_fixed_point_proved": mapped_fixed_point_inside,
        "global_one_step_cube_lipschitz_fraction": str(L),
        "all_step_intermediate_factor_float": float(all_steps_factor),
        "all_step_rate": "||F^(12m+j)(x)-p|| <= L^j K12^m ||x-p|| for 0<=j<12 and all x in NOR04 ball",
        "scope": "unique one-step fixed point and all-depth convergence for real-valued trained zero-input GRU within NOR04 local ball",
        "heldout_input_basin_entry_proved": False,
        "native_proof_search_gain_proved": False,
        "precision": "32-decimal outward rational intervals, adaptive Taylor exp",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    args=p.parse_args()
    receipt=period_one_certificate()
    if args.check:
        assert receipt["checkpoint_sha256"]=="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
        assert receipt["one_step_fixed_point_proved"], receipt
        assert receipt["heldout_input_basin_entry_proved"] is False
    output=json.dumps(receipt,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(output)
    else:
        print(output,end="")


if __name__=="__main__":
    main()
