"""NOR-07C: independent exact-rational audit of trained two-attractor receipt.

This quick verification checks invariance, one-step return and disjoint
basins using the source-bound NOR-04/05 first-attractor receipt and the
NOR-07 second-attractor frozen output. The full 24-step interval proof is
performed in the archived NOR-07 Actions run; this is an exact arithmetic
replay of its published inequalities.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
NOR04=HERE/"results"/"nor04_invariant_ball.json"
NOR05=HERE/"results"/"nor05_period_one.json"
NOR07=HERE/"results"/"nor07_two_attractors_summary.json"


def validate():
    cksha=sha256(SOURCE.read_bytes()).hexdigest()
    old=json.loads(NOR04.read_text())
    firstperiod=json.loads(NOR05.read_text())
    new=json.loads(NOR07.read_text())
    assert new["schema"]=="nor07.trained_gru.two_attractors.frozen_summary.v1"
    assert cksha==new["original_checkpoint_sha256"]==old["checkpoint_sha256"]==firstperiod["checkpoint_sha256"]
    assert old["banach_fixed_point_certificate"] and firstperiod["one_step_fixed_point_proved"]
    assert new["two_locally_attracting_fixed_points_certified"]
    assert new["disjoint_balls_certified"]

    r1=Q(old["radius_fraction"])
    c1=tuple(Q(z) for z in old["center_literals"])
    second=new["second_attractor"]
    c2=tuple(Q(z) for z in second["center_decimals"])
    assert len(c1)==len(c2)==12
    r2=Q(second["radius_fraction"])
    k2=Q(second["uniform_block_operator_norm_upper_fraction"])
    s2=Q(second["center_block_residual_upper_fraction"])
    margin=Q(second["exact_ball_invariance_margin_fraction"])
    assert 0<k2<1 and 0<s2<r2
    assert r2-s2-k2*r2==margin and margin>0

    l1=Q(second["one_step_operator_norm_upper_fraction"])
    s1=Q(second["one_step_center_residual_upper_fraction"])
    image_bound=Q(second["one_step_fixed_point_image_upper_fraction"])
    assert s1+l1*s2/(1-k2)==image_bound and image_bound<r2
    assert second["certified_period_one"] is True

    sep=Q(new["exact_coordinate_separation_lower_fraction"])
    assert abs(c1[1]-c2[1])-r1-r2==sep
    assert sep>0 and r1+r2<abs(c1[1]-c2[1])
    assert second["input_trajectory_entry_into_small_ball_certified"] is False

    return {
        "schema":"nor07.two_attractors.rational_receipt_audit.v1",
        "source_checkpoint_sha256":cksha,
        "block_depth":second["block_depth"],
        "second_block_K_upper_float":float(k2),
        "second_ball_radius_float":float(r2),
        "invariance_margin_float":float(margin),
        "one_step_image_upper_float":float(image_bound),
        "coordinate_separation_lower_float":float(sep),
        "two_distinct_period_one_fixed_points_certified":True,
        "squared_family_capture_into_second_ball_certified":False,
        "scope":"arithmetic audit of archived independent rational interval proof; no retraining",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    data=validate()
    if args.check:
        assert data["two_distinct_period_one_fixed_points_certified"]
        assert not data["squared_family_capture_into_second_ball_certified"]
    print(json.dumps(data,sort_keys=True,indent=2))

if __name__=="__main__":
    main()
