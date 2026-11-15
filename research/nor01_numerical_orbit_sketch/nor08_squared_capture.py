"""NOR-08B: exact-rational squared-source capture into second GRU basin.

Proves that the original RRC25 squared-family public-input GRU state
enters the independently source-locked enlarged NOR-08 invariant ball
near the SECOND fixed point. Every extrapolated 12-step recurrence block
carries a rational nominal hidden vector and a rigorous Euclidean error
radius. The nonnormal GRU Jacobian is enclosed on the whole state box.

The recorded FP64 first crossing (step 651) selects the bounded search
horizon; it is never used as evidence inside the capture calculation.
A proof is accepted only if the exact Fraction center clearance exceeds
the propagated state-error upper bound.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(__file__).resolve().parent
for directory in (ROOT,LOCAL):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor06_capture as capture
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import nor07_stationary_tube as stationary

HERE=Path(__file__).resolve().parent
BASIN=HERE/"results"/"nor08_expanded_second_ball.json"
ATTRACTORS=HERE/"results"/"nor07_two_attractors_summary.json"
BLOCK=12
SHA=survey.KNOWN_SHA


def frozen_second_basin():
    frozen=json.loads(BASIN.read_text())
    source=json.loads(ATTRACTORS.read_text())
    assert frozen["schema"]=="nor08.second_basin.certified_expanded_ball.v1"
    assert frozen["trained_checkpoint_sha256"]==SHA
    assert source["original_checkpoint_sha256"]==SHA
    assert source["two_locally_attracting_fixed_points_certified"]
    center=tuple(Q(x) for x in source["second_attractor"]["center_decimals"])
    radius=Q(frozen["radius_fraction"])
    K=Q(frozen["uniform_derivative_K_upper_fraction"])
    residual=Q(frozen["center_residual_upper_fraction"])
    margin=Q(frozen["exact_invariance_margin_fraction"])
    assert radius==Q(1,10**6)
    assert 0<K<1
    assert radius-K*radius-residual==margin and margin>0
    assert frozen["certified_invariant_ball"] is True
    return center,radius,K,residual


def run(max_blocks=70,record_stride=6):
    if not 1<=max_blocks<=128:
        raise ValueError("bounded 1..128 blocks required")
    if not 1<=record_stride<=max_blocks:
        raise ValueError("invalid trace stride")
    target,R,K,residual=frozen_second_basin()
    ck,digests=bench25.checkpoints()
    model=ck["recurrent"]
    assert digests["recurrent"]==SHA and model["parameters"]==1000
    weights=capture.get_weights(model)
    center,error,meta=capture.source_state(weights,"squared")
    records=[]
    proof_trace=sha256()
    witness=None
    max_error=error
    block_max_k=Q(0)
    for k in range(max_blocks+1):
        # Center-to-second-attractor distance is certified by an upward
        # exact rational square root. This keeps target-set membership
        # independent of floating-point behavior.
        ds=capture.sqrt_upper(sum(((x-y)**2 for x,y in
              zip(center,target,strict=True)),Q(0)))
        clearance=R-ds-error
        proof_trace.update((f"{k};{ds};{error};{clearance}\n").encode())
        if k%record_stride==0 or clearance>=0 or k==max_blocks:
            records.append({
                "block":k,"zero_input_steps":k*BLOCK,
                "target_center_distance_upper_float":float(ds),
                "state_error_radius_upper_float":float(error),
                "clearance_float":float(clearance),
                "certified_inside":clearance>=0
            })
        if clearance>=0:
            witness={
                "block_index":k,
                "zero_input_steps":k*BLOCK,
                "target_center_distance_upper_fraction":str(ds),
                "tracked_state_error_upper_fraction":str(error),
                "exact_clearance_fraction":str(clearance),
                "true_squared_source_state_inside_second_invariant_ball":True,
            }
            break
        if k==max_blocks:
            break
        center,error,jacobian_norm,residual_step=capture.one_block(weights,center,error)
        block_max_k=max(block_max_k,jacobian_norm)
        max_error=max(max_error,error)
        proof_trace.update((f"K={jacobian_norm};center_residual={residual_step}\n").encode())
        if error>=100:
            break

    return {
        "schema":"nor08.second_basin.squared_natural_source_capture.v1",
        "checkpoint_sha256":SHA,
        "checkpoint_parameters":model["parameters"],
        "source_hashes_verified":model["source_sha256"],
        "source_metadata":meta,
        "basin_center_decimals":[str(x) for x in target],
        "certified_second_basin_radius_fraction":str(R),
        "certified_second_basin_24_step_K_fraction":str(K),
        "certified_second_basin_24_step_center_residual_fraction":str(residual),
        "certified_capture":witness,
        "certification":"certified" if witness else "not_established",
        "checked_blocks":k,
        "block_stride":BLOCK,
        "requested_max_blocks":max_blocks,
        "initial_source_state":"exact interval replay of nine public RRC25 feature rows",
        "last_target_distance_upper_fraction":str(ds),
        "last_trajectory_error_upper_fraction":str(error),
        "last_clearance_fraction":str(clearance),
        "max_tracked_error_fraction":str(max_error),
        "max_12_step_operator_bound_fraction":str(block_max_k),
        "exact_trace_sha256":proof_trace.hexdigest(),
        "trace":records,
        "source_labels_or_future_used":False,
        "floating_point_acceptance":False,
        "scope":"natural squared-family input, real-analytic decimal-rational trained GRU, zero-input continuation; no native proof ranking outcome asserted"
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-blocks",type=int,default=70)
    ap.add_argument("--stride",type=int,default=6)
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--require-capture",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    res=run(args.max_blocks,args.stride)
    if args.check:
        assert res["checkpoint_sha256"]==SHA
        assert res["source_metadata"]["family"]=="squared"
        assert res["floating_point_acceptance"] is False
        if res["certified_capture"] is not None:
            w=res["certified_capture"]
            assert Q(w["target_center_distance_upper_fraction"])+Q(
                w["tracked_state_error_upper_fraction"]
            )<=Q(res["certified_second_basin_radius_fraction"])
    if args.require_capture:
        assert res["certified_capture"] is not None, (
          "Source-to-basin entry inconclusive at bounded horizon; inspect full artifact")
    body=json.dumps(res,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(body)
    else:
        print(body,end="")


if __name__=="__main__":
    main()
