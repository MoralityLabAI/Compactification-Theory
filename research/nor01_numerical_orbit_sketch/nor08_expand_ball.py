"""NOR-08: enlarge the rigorously certified second GRU attractor basin.

The original NOR-07 certificate proves a period-one fixed point of the
source-locked RRC25 zero-input real-analytic GRU inside a radius-1e-20
ball. This deterministic search tests a predeclared descending set of
larger rational radii, with exact stationary-center Volterra bounds.

Each candidate independently verifies (i) uniform contraction of the
true nonlinear F^24 block throughout the proposed ball, (ii) a strict
invariance margin, (iii) the one-step return uniqueness test, and
(iv) disjointness from the NOR-04/05 first attracting ball.

No trained weights change and no input-conditioned future observations
are consulted to select a candidate radius. Rational proof status is
reported separately from all float display quantities.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(__file__).resolve().parent
for d in (ROOT,LOCAL):
    if str(d) not in sys.path:
        sys.path.insert(0,str(d))

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch.invariant_ball import (
    CENTER as FIRST_CENTER,RADIUS as FIRST_RADIUS,
)
from nor07_stationary_tube import stationary_tube
from trained_gru_cert import get_weights

NOR07=LOCAL/"results"/"nor07_two_attractors_summary.json"
ORIGINAL_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
RADII=(Q(1,10**5),Q(3,10**6),Q(1,10**6),
       Q(3,10**7),Q(1,10**7),Q(1,10**8),Q(1,10**9))
DEPTH=24


def verified_source():
    ck,dig=bench25.checkpoints()
    if dig["recurrent"]!=ORIGINAL_SHA or ck["recurrent"]["parameters"]!=1000:
        raise AssertionError("original trained checkpoint hash changed")
    blob=NOR07.read_bytes()
    proof=json.loads(blob)
    assert proof["schema"]=="nor07.trained_gru.two_attractors.frozen_summary.v1"
    assert proof["original_checkpoint_sha256"]==ORIGINAL_SHA
    assert proof["two_locally_attracting_fixed_points_certified"]
    center=tuple(Q(s) for s in proof["second_attractor"]["center_decimals"])
    original_radius=Q(proof["second_attractor"]["radius_fraction"])
    assert len(center)==12 and original_radius==Q(1,10**20)
    return ck["recurrent"], center, original_radius, sha256(blob).hexdigest()


def run():
    ck,center,initial_radius,receipt_sha=verified_source()
    weights=get_weights(ck)
    tries=[]
    chosen=None
    for radius in RADII:
        block=stationary_tube(weights,center,radius,DEPTH)
        K=Q(block["block_operator_norm_upper_fraction"])
        s=Q(block["center_residual_upper_fraction"])
        one=stationary_tube(weights,center,radius,1)
        L1=Q(one["block_operator_norm_upper_fraction"])
        s1=Q(one["center_residual_upper_fraction"])
        margin=radius-K*radius-s
        one_return=None if K>=1 else s1+L1*s/(1-K)
        sep=abs(center[1]-FIRST_CENTER[1])-radius-FIRST_RADIUS
        accepted=(K<1 and margin>0 and one_return is not None
                  and one_return<radius and sep>0)
        row={
            "candidate_radius_fraction":str(radius),
            "block_K_fraction":str(K),
            "block_K_float":float(K),
            "block_center_residual_fraction":str(s),
            "invariance_margin_fraction":str(margin),
            "one_step_local_operator_fraction":str(L1),
            "one_step_residual_fraction":str(s1),
            "one_step_fixed_image_upper_fraction":str(one_return) if one_return is not None else None,
            "coordinate_separation_lower_fraction":str(sep),
            "all_rational_gates_passed":accepted,
        }
        tries.append(row)
        if accepted and chosen is None:
            chosen=row
    if chosen:
        radius=Q(chosen["candidate_radius_fraction"])
        assert radius>initial_radius
        assert Q(chosen["invariance_margin_fraction"])>0
        assert Q(chosen["one_step_fixed_image_upper_fraction"])<radius
    return {
        "schema":"nor08.trained_gru.second_ball_expansion.v1",
        "checkpoint_sha256":ORIGINAL_SHA,
        "nor07_prior_frozen_receipt_sha256":receipt_sha,
        "candidate_radii":"fixed descending rational list, independently checked",
        "candidate_attempts":tries,
        "selected_largest_certified":chosen,
        "certified_expanded_ball":chosen is not None,
        "initial_nor07_radius_fraction":str(initial_radius),
        "original_second_attractor_center_decimals":[str(c) for c in center],
        "first_attractor_radius_fraction":str(FIRST_RADIUS),
        "squared_family_source_entry_certified":False,
        "scope":"local real-analytic second fixed-point basin only; input-conditioned capture needs separate rational shadowing",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=run()
    if args.check:
        assert result["checkpoint_sha256"]==ORIGINAL_SHA
        assert result["certified_expanded_ball"], result
        assert result["squared_family_source_entry_certified"] is False
    blob=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
