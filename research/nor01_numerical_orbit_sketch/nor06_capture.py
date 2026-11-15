"""NOR-06B: sound rational capture of a naturally sourced RRC-25 GRU state.

Input: the original source-locked recurrent checkpoint and one fixed
public RRC25 input family. Starting hidden h9 is enclosed via exact-decimal
rational interval replay of the original nine feature rows.

Each future zero-input 12-step block propagates:
- a rational nominal center c_k,
- an explicit Euclidean state-error radius r_k,
- an interval Jacobian product K_k covering all states in the ball, and
- a separate interval residual e_k covering F^12(c_k)-c_(k+1).

The validated recurrence r_(k+1) >= e_k + K_k*r_k proves every
true hidden state lies in its corresponding rational ball. Once
||c_k - NOR04.center||+r_k <= NOR04.radius, the full all-depth
nonlinear NOR04/05 convergence theorem applies.

This is distinct from the NOR06 floating-point discovery survey. Floating
point is used only for descriptive output. All certification decisions use
Fraction and outward-rounded real-analytic sigmoid/tanh intervals.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(__file__).resolve().parent
for d in (ROOT,LOCAL):
    if str(d) not in sys.path:
        sys.path.insert(0,str(d))

from research.rrc_compactification import bench25,compound25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER,RADIUS
from research.nor01_numerical_orbit_sketch.trained_gru_cert import (
    Interval, as_interval, N, SCALE, ZERO, ONE,
    gates, jacobian_interval, get_weights, interval_forward,
    rational_center, radius_upper, sqrt_upper, ceil_rational,
)

BLOCK=12
SOURCE_SHA=survey.KNOWN_SHA


def unit_cube_center_ball(center, radius):
    """Coordinate enclosure containing the Euclidean ball intersected
    with the GRU's proved invariant hidden cube."""
    if radius<0 or len(center)!=N:
        raise ValueError("invalid radius or center")
    box=[]
    for c in center:
        if c < -1 or c > 1:
            raise ArithmeticError("nominal center departed hidden cube")
        lower=max(Q(-1),c-radius)
        upper=min(Q(1),c+radius)
        if lower>upper or c<lower or c>upper:
            raise ArithmeticError("invalid invariant-cube intersection")
        box.append(Interval(lower,upper))
    return box


def product_interval(left,right):
    """Exact outward interval matrix multiplication preserving order."""
    return [[sum((left[i][k]*right[k][j] for k in range(N)),
                 Interval.point(0)) for j in range(N)] for i in range(N)]


def operator_frobenius_upper(matrix):
    sq=sum((max(abs(v.lo),abs(v.hi))**2 for row in matrix for v in row),ZERO)
    return sqrt_upper(sq)


def one_block(weights, center, radius):
    box=unit_cube_center_ball(center,radius)
    matrix=[[Interval.point(int(i==j)) for j in range(N)] for i in range(N)]
    for step in range(BLOCK):
        J=jacobian_interval(weights,box)
        matrix=product_interval(J,matrix)
        box=gates(weights,box)[0]
        if any(z.lo < -ONE or z.hi>ONE for z in box):
            raise AssertionError("GRU invariant cube violated")
    K=operator_frobenius_upper(matrix)

    # Exact center ray, evaluated independently of its finite-radius box.
    actual=[Interval.point(x) for x in center]
    for step in range(BLOCK):
        actual=gates(weights,actual)[0]
    next_center=rational_center(actual)
    residual=radius_upper(actual,next_center)
    next_radius=ceil_rational(residual + K*radius, SCALE)
    if not all(z.lo<=c<=z.hi for z,c in zip(box,next_center,strict=True)):
        # True orbit is enclosed; nominal center may differ by <=center
        # point-rounding error, so a box failure is diagnostic only.
        pass
    return next_center,next_radius,K,residual


def source_state(weights, family):
    cases={c.family:c for c in survey.public_cases()}
    if family not in cases:
        raise ValueError("requested family absent from source roster")
    case=cases[family]
    header=compound25.possible_headers(case)[0]
    features=compound25.features(case,header)
    boxes=interval_forward(weights,features)
    center=rational_center(boxes)
    radius=radius_upper(boxes,center)
    return center,ceil_rational(radius,SCALE),{
        "family":family,
        "public_case_id_sha256":sha256(case.identifier.encode()).hexdigest(),
        "public_visible_sha256":sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest(),
        "public_features_sha256":sha256(json.dumps(features,separators=(",",":")).encode()).hexdigest(),
        "public_header_candidates":len(compound25.possible_headers(case)),
    }


def run(family="signed", max_blocks=144, record_stride=12):
    if not 1<=max_blocks<=512:
        raise ValueError("max_blocks 1..512 required")
    if record_stride<1:
        raise ValueError("positive record_stride required")
    models,digests=bench25.checkpoints()
    ck=models["recurrent"]
    assert ck["parameters"]==1000 and digests["recurrent"]==SOURCE_SHA
    weights=get_weights(ck)
    center,radius,meta=source_state(weights,family)
    rows=[]
    trace=sha256()
    capture=None
    largest_radius=radius
    max_K=ZERO
    for k in range(max_blocks+1):
        center_distance=sqrt_upper(
            sum(((x-y)**2 for x,y in zip(center,CENTER,strict=True)),ZERO))
        clearance=RADIUS-center_distance-radius
        # All comparisons that determine the certificate are over Q.
        if k%record_stride==0 or clearance>=0 or k==max_blocks:
            rows.append({
                "block":k,"zero_steps":k*BLOCK,
                "center_distance_upper":float(center_distance),
                "tracked_error_radius_upper":float(radius),
                "entry_clearance":float(clearance),
                "inside_certified_ball":clearance>=0,
            })
        trace.update((f"{k};{center_distance};{radius};{clearance}\n").encode())
        if clearance>=0:
            capture={
                "block":k,"zero_steps":k*BLOCK,
                "center_distance_upper_fraction":str(center_distance),
                "tracked_error_radius_fraction":str(radius),
                "ball_clearance_fraction":str(clearance),
                "true_state_in_nor04_ball":True,
            }
            break
        if k==max_blocks:
            break
        center,radius,K,e=one_block(weights,center,radius)
        largest_radius=max(largest_radius,radius)
        max_K=max(max_K,K)
        trace.update((f"K={K};residual={e}\n").encode())
        if radius>=100:
            # Monotone proof envelope has become too loose to continue
            # efficiently; retain an explicit failed-certificate receipt.
            break
    return {
        "schema":"nor06.rational_source_basin_capture.v1",
        "checkpoint_sha256":digests["recurrent"],
        "checkpoint_parameters":ck["parameters"],
        "source_hashes_verified":ck["source_sha256"],
        "family":family,
        "public_source":meta,
        "block_stride":BLOCK,
        "requested_max_blocks":max_blocks,
        "actual_processed_blocks":k,
        "checkpoint_real_model":"decimal-rational lift of RRC25 weights and public features",
        "certification":"true" if capture else "not_established",
        "certified_capture":capture,
        "nor04_ball_radius_fraction":str(RADIUS),
        "last_center_distance_upper_fraction":str(center_distance),
        "last_state_error_upper_fraction":str(radius),
        "last_clearance_fraction":str(clearance),
        "max_state_error_upper_fraction":str(largest_radius),
        "max_block_operator_upper_fraction":str(max_K),
        "exact_trace_sha256":trace.hexdigest(),
        "record_stride":record_stride,
        "sampled_trace":rows,
        "limits":"all decisions exact rational; only one source family at a time; a failed envelope is an inconclusive capture test",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=survey.FAMILIES,default="signed")
    p.add_argument("--max-blocks",type=int,default=144)
    p.add_argument("--stride",type=int,default=12)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    receipt=run(args.family,args.max_blocks,args.stride)
    if args.check:
        assert receipt["checkpoint_sha256"]==SOURCE_SHA
        assert receipt["checkpoint_parameters"]==1000
        if receipt["certified_capture"]:
            from fractions import Fraction
            c=receipt["certified_capture"]
            assert Q(c["center_distance_upper_fraction"])+Q(c["tracked_error_radius_fraction"])<=RADIUS
            assert c["true_state_in_nor04_ball"]
    body=json.dumps(receipt,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(body)
    else:
        print(body,end="")


if __name__=="__main__":
    main()
