"""NOR-08B: rigorously certify squared-family source entry into second basin.

The source-locked NOR-06B rational block shadowing algorithm is reused
without modification, with only its target ball replaced by the frozen
NOR-08 second attracting fixed-point center and enlarged radius.

Every 12-step block carries a certified Euclidean error radius under the
true decimal-rational trained GRU. A valid terminal certificate is solely
the exact Fraction inequality
    ||c_k - center_two|| + error_radius <= radius_two.

A verified entry combines with NOR-07/08 to prove all subsequent nonlinear
zero-input recurrence depths converge to the *second* fixed point.
Failed or resource-exhausted certification is explicitly inconclusive.
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
for path in (ROOT,LOCAL):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor06_capture as cap
from research.nor01_numerical_orbit_sketch.invariant_ball import (
    CENTER as FIRST_CENTER,RADIUS as FIRST_RADIUS
)
from trained_gru_cert import get_weights,sqrt_upper,ZERO

FAMILY="squared"
SOURCE_SHA=cap.SOURCE_SHA
NOR07=LOCAL/"results"/"nor07_two_attractors_summary.json"
NOR08=LOCAL/"results"/"nor08_second_basin_summary.json"
BLOCK=12


def certified_target():
    prior=json.loads(NOR07.read_text())
    expanded=json.loads(NOR08.read_text())
    assert prior["schema"]=="nor07.trained_gru.two_attractors.frozen_summary.v1"
    assert expanded["schema"]=="nor08.second_basin_expansion.frozen_summary.v1"
    assert prior["original_checkpoint_sha256"]==expanded["checkpoint_sha256"]==SOURCE_SHA
    assert prior["two_locally_attracting_fixed_points_certified"]
    assert expanded["selected"]["all_rational_gates_passed"]
    center=tuple(Q(x) for x in prior["second_attractor"]["center_decimals"])
    radius=Q(expanded["selected"]["radius_fraction"])
    K=Q(expanded["selected"]["block_operator_upper_fraction"])
    assert len(center)==12 and 0<K<1
    assert radius==Q(3,10**6)
    assert Q(expanded["selected"]["invariance_margin_fraction"])>0
    assert abs(center[1]-FIRST_CENTER[1])>radius+FIRST_RADIUS
    return center,radius,K,sha256(NOR07.read_bytes()).hexdigest(),sha256(NOR08.read_bytes()).hexdigest()


def run(max_blocks=180,record_stride=12):
    if not 1<=max_blocks<=512 or record_stride<1:
        raise ValueError("invalid finite certification horizon")
    models,digests=bench25.checkpoints()
    ck=models["recurrent"]
    assert ck["parameters"]==1000 and digests["recurrent"]==SOURCE_SHA
    weights=get_weights(ck)
    target,R,K,prior_hash,expanded_hash=certified_target()

    center,error,meta=cap.source_state(weights,FAMILY)
    trace=sha256()
    rows=[]
    witness=None
    max_error=error
    max_operator=ZERO
    for k in range(max_blocks+1):
        distance=sqrt_upper(sum((
            (x-y)**2 for x,y in zip(center,target,strict=True)),ZERO))
        clearance=R-distance-error
        if k%record_stride==0 or clearance>=0 or k==max_blocks:
            rows.append({
                "block":k,"extra_zero_steps":k*BLOCK,
                "target_center_distance_upper":float(distance),
                "tracked_error_upper":float(error),
                "exact_clearance_float":float(clearance),
                "captured":clearance>=0,
            })
        trace.update((f"{k};{distance};{error};{clearance}\n").encode())
        if clearance>=0:
            witness={
                "certified":True,"block":k,"extra_zero_input_steps":k*BLOCK,
                "target_center_distance_upper_fraction":str(distance),
                "trajectory_error_upper_fraction":str(error),
                "exact_clearance_fraction":str(clearance),
            }
            break
        if k==max_blocks:break
        center,error,K_block,residual=cap.one_block(weights,center,error)
        max_error=max(error,max_error)
        max_operator=max(max_operator,K_block)
        trace.update((f"K={K_block};e={residual}\n").encode())
        if error>=100:
            break

    return {
        "schema":"nor08.squared_public_source.second_basin_rational_capture.v1",
        "checkpoint_sha256":digests["recurrent"],
        "source_model_parameters":ck["parameters"],
        "public_source":meta,
        "target_second_ball_disjoint_from_first":True,
        "target_second_center_sha256":prior_hash,
        "target_expanded_ball_receipt_sha256":expanded_hash,
        "target_ball_radius_fraction":str(R),
        "target_block_contraction_fraction":str(K),
        "requested_max_blocks":max_blocks,
        "processed_blocks":k,
        "block_length":BLOCK,
        "capture_status":"certified" if witness else "not_established",
        "capture_witness":witness,
        "last_center_distance_upper_fraction":str(distance),
        "last_tracked_error_upper_fraction":str(error),
        "max_tracked_error_upper_fraction":str(max_error),
        "max_actual_block_derivative_upper_fraction":str(max_operator),
        "exact_trace_sha256":trace.hexdigest(),
        "sampled_trace":rows,
        "scope":"rational certified zero-input dynamics after nine public RRC25 feature steps; no source-label leakage or native ranking gain",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--blocks",type=int,default=180)
    p.add_argument("--stride",type=int,default=12)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=run(args.blocks,args.stride)
    if args.check:
        assert result["checkpoint_sha256"]==SOURCE_SHA
        if result["capture_witness"]:
            w=result["capture_witness"]
            assert Q(w["target_center_distance_upper_fraction"])+Q(w["trajectory_error_upper_fraction"])<=Q(result["target_ball_radius_fraction"])
    blob=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
