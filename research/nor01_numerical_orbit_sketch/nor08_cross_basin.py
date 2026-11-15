"""NOR-08D: exact cross-basin metric margin and one-bit label obligation.

Two certified invariant balls of the original trained GRU are separated
by a source-locked exact Euclidean center-distance lower bound. For any
pair of hidden states in those balls, their true Euclidean separation
exceeds a fixed rational margin. A common terminal code of metric zero
would incur at least that much additive cross-basin distortion.

This is a finite metric proof on the two certified balls. Source-to-
second-basin entry remains a separate empirical/interval obligation.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from math import isqrt
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIRST=HERE/"results"/"nor04_invariant_ball.json"
SECOND=HERE/"results"/"nor07_two_attractors_summary.json"
EXPANDED=HERE/"results"/"nor08_expanded_second_ball.json"
SIGNED=HERE/"results"/"nor06_certified_capture_summary.json"
CHECKPOINT=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
SCALE=10**18
SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def lower_sqrt_rational(sq,scale=SCALE):
    if sq<0 or scale<=0:
        raise ValueError("nonnegative square and positive scale required")
    numerator=sq.numerator*scale*scale
    y=isqrt(numerator//sq.denominator)
    val=Q(y,scale)
    assert val*val<=sq
    return val


def audit():
    first=json.loads(FIRST.read_text())
    second=json.loads(SECOND.read_text())
    expanded=json.loads(EXPANDED.read_text())
    signed=json.loads(SIGNED.read_text())
    assert sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    assert first["checkpoint_sha256"]==second["original_checkpoint_sha256"]==expanded["trained_checkpoint_sha256"]==SHA
    assert first["banach_fixed_point_certificate"] and second["two_locally_attracting_fixed_points_certified"]
    assert signed["signed"]["certified"]

    p=tuple(Q(x) for x in first["center_literals"])
    q=tuple(Q(x) for x in second["second_attractor"]["center_decimals"])
    assert len(p)==len(q)==12
    r1=Q(first["radius_fraction"])
    r2=Q(expanded["radius_fraction"])
    assert r1==Q(1,10**11) and r2==Q(1,10**6)
    gap_squared=sum(((a-b)**2 for a,b in zip(p,q)),Q(0))
    distance_lower=lower_sqrt_rational(gap_squared)
    epsilon=distance_lower-r1-r2
    assert epsilon>5
    # Any x in the first ball and y in the second has
    # distance(x,y) >= distance_lower-r1-r2 by the triangle inequality.
    # Assigning them identical terminal codes incurs at least epsilon
    # additive metric error in the cross-basin pair.
    return {
        "schema":"nor08.two_basin.metric_code_capacity.v1",
        "source_checkpoint_sha256":SHA,
        "first_ball_radius_fraction":str(r1),
        "second_ball_radius_fraction":str(r2),
        "exact_center_distance_squared_fraction":str(gap_squared),
        "center_distance_lower_fraction":str(distance_lower),
        "first_second_ball_separation_lower_fraction":str(epsilon),
        "first_second_ball_separation_lower":float(epsilon),
        "minimum_distinct_terminal_code_labels":2,
        "minimum_fixed_width_terminal_label_bits":1,
        "shared_code_minimum_additive_cross_distortion_fraction":str(epsilon),
        "signed_natural_source_capture":True,
        "squared_natural_source_capture_certified_by_this_audit":False,
        "proof_scope":"any states in respective certified invariant balls, with naturally generated squared source entry requiring its own rational proof",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    result=audit()
    if a.check:
        assert Q(result["first_second_ball_separation_lower_fraction"])>Q(5)
        assert result["minimum_fixed_width_terminal_label_bits"]==1
    blob=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
