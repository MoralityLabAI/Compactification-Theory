"""NOR-07: two disjoint attracting fixed points in the trained RRC25 GRU.

A second fixed-point candidate is selected from the terminal FP64 state
of the squared-family public-input NOR06 survey. Decimal Newton
refinement proposes a high-precision center, and independently
verified rational interval bounds prove existence, uniqueness and
period-one status in a tiny neighborhood of that center.

This is a theorem about the *real-analytic decimal-rational lift* of
the original trained zero-input GRU, not its native proof-ranking task.
The squared-family input trajectory's entry into the second basin
remains an independent formal obligation.
"""
from __future__ import annotations

import argparse
from decimal import Decimal,localcontext
from fractions import Fraction as Q
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(__file__).resolve().parent
for directory in (ROOT,LOCAL):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch.invariant_ball import (
    CENTER as FIRST_CENTER,RADIUS as FIRST_RADIUS,centered_ball,
)
# The existing invariant_ball module imports trained_gru_cert as a local
# module. Use the same module identity here so Interval instance checks
# remain valid across both independent certificates.
from trained_gru_cert import (
    Interval,N,build_certificate,gates,get_weights,sqrt_upper,ZERO,
)

SECOND_SEED=(
    "0.1451578041821899",
    "-0.8993696968577396",
    "-0.99235391607578038",
    "-0.99713057751398471",
    "-0.50766322687425225",
    "-0.7910793027003602",
    "-0.99849161886238103",
    "0.62481200060219655",
    "-0.46161146377127465",
    "-0.99154900735304718",
    "0.97883031983135038",
    "0.99368737135304197",
)
BLOCK=24
RADIUS=Q(1,10**20)
EXPECTED_CK="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def dsig(x):
    return Decimal(1)/(Decimal(1)+(-x).exp())


def decimal_step_and_jacobian(h,weights):
    vh=[weights["bh"][i]+sum((weights["hh"][i][j]*h[j] for j in range(N)),Decimal(0))
        for i in range(3*N)]
    result=[]
    J=[[Decimal(0) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        r=dsig(weights["bi"][i]+vh[i])
        z=dsig(weights["bi"][N+i]+vh[N+i])
        nv=(weights["bi"][2*N+i]+r*vh[2*N+i])
        t=(Decimal(2)*dsig(Decimal(2)*nv)-1)
        result.append(z*h[i]+(1-z)*t)
        for j in range(N):
            dr=r*(1-r)*weights["hh"][i][j]
            dz=z*(1-z)*weights["hh"][N+i][j]
            dt=(1-t*t)*(r*weights["hh"][2*N+i][j]+vh[2*N+i]*dr)
            J[i][j]=(z if i==j else Decimal(0))+(h[i]-t)*dz+(1-z)*dt
    return result,J


def decimal_solve(A,b):
    """Pivoted Gaussian elimination over the decimal rationals."""
    n=len(b)
    mat=[row[:]+[b[i]] for i,row in enumerate(A)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(mat[i][j]))
        if abs(mat[pivot][j])<Decimal("1e-50"):
            raise ArithmeticError("singular Newton tangent")
        mat[j],mat[pivot]=mat[pivot],mat[j]
        den=mat[j][j]
        for k in range(j,n+1):
            mat[j][k]/=den
        for i in range(j+1,n):
            z=mat[i][j]
            for k in range(j,n+1):
                mat[i][k]-=z*mat[j][k]
    out=[Decimal(0)]*n
    for i in range(n-1,-1,-1):
        out[i]=mat[i][n]-sum((mat[i][j]*out[j] for j in range(i+1,n)),Decimal(0))
    return out


def decimal_newton_center(checkpoint):
    s=checkpoint["state_dict"]
    with localcontext() as ctx:
        ctx.prec=80
        weights={
            "hh":[[Decimal(str(x)) for x in row] for row in s["rnn.weight_hh_l0"]],
            "bi":[Decimal(str(x)) for x in s["rnn.bias_ih_l0"]],
            "bh":[Decimal(str(x)) for x in s["rnn.bias_hh_l0"]],
        }
        h=[Decimal(s) for s in SECOND_SEED]
        residual=[]
        for _ in range(10):
            y,J=decimal_step_and_jacobian(h,weights)
            f=[v-x for v,x in zip(y,h,strict=True)]
            res=max(map(abs,f))
            residual.append(str(res))
            if res<Decimal("1e-65"):
                break
            M=[[J[i][j]-(1 if i==j else 0) for j in range(N)] for i in range(N)]
            d=decimal_solve(M,[-x for x in f])
            h=[x+dx for x,dx in zip(h,d,strict=True)]
        if max(abs(v-u) for v,u in zip(decimal_step_and_jacobian(h,weights)[0],h,strict=True))>=Decimal("1e-57"):
            raise ArithmeticError("Newton candidate did not converge")
        # Each decimal number is exact in the ensuing rational interval proof.
        values=tuple(format(x.quantize(Decimal("1e-55")),"f") for x in h)
        return tuple(Q(x) for x in values),list(values),residual


def point_residual(weights,center,steps):
    z=[Interval.point(x) for x in center]
    for _ in range(steps):
        z=gates(weights,z)[0]
    d=[max(abs(v.lo-c),abs(v.hi-c)) for v,c in zip(z,center,strict=True)]
    return sqrt_upper(sum((x*x for x in d),ZERO),10**24)


@lru_cache(maxsize=1)
def two_fixed_points_certificate():
    original,digests=bench25.checkpoints()
    ck=original["recurrent"]
    if ck["parameters"]!=1000 or digests["recurrent"]!=EXPECTED_CK:
        raise AssertionError("wrong trained checkpoint")
    center,center_literals,history=decimal_newton_center(ck)
    weights=get_weights(ck)
    box=centered_ball(center,RADIUS)
    block=build_certificate(weights,box,length=BLOCK,
                            label="squared_family_second_attractor")
    K=Q(block["block_operator_frobenius_upper_fraction"])
    s=point_residual(weights,center,BLOCK)
    margin=RADIUS-s-K*RADIUS
    if K>=1 or margin<=0:
        raise AssertionError(
            f"second 24-step block failed local contraction: K={K}, s={s}, R={RADIUS}, margin={margin}")
    dstar=s/(1-K)
    one=build_certificate(weights,box,length=1,
                          label="squared_family_period_one")
    L1=Q(one["block_operator_frobenius_upper_fraction"])
    s1=point_residual(weights,center,1)
    image_bound=s1+L1*dstar
    if image_bound>=RADIUS:
        raise AssertionError("second periodic orbit not excluded")
    # Distinctness is certified by a single coordinate lower bound:
    # |c_first[1]-c_second[1]| > R_first+R_second.
    coordinate_gap=abs(center[1]-FIRST_CENTER[1])-RADIUS-FIRST_RADIUS
    if coordinate_gap<=0:
        raise AssertionError("two fixed-point balls overlap")
    return {
        "schema":"nor07.trained_gru.two_distinct_attracting_fixed_points.v1",
        "source_checkpoint_sha256":digests["recurrent"],
        "model_parameters":ck["parameters"],
        "candidate_source":"squared-family NOR-06 FP64 terminal hidden vector after 4800 zero-input steps",
        "decimal_newton_residuals":history,
        "second_center_decimals":center_literals,
        "second_ball_radius_fraction":str(RADIUS),
        "block_depth":BLOCK,
        "second_block_operator_norm_upper_fraction":str(K),
        "second_block_operator_norm_upper_float":float(K),
        "second_block_residual_upper_fraction":str(s),
        "second_invariance_margin_fraction":str(margin),
        "second_invariant_block":True,
        "second_one_step_local_norm_upper_fraction":str(L1),
        "second_one_step_residual_upper_fraction":str(s1),
        "second_one_step_image_bound_fraction":str(image_bound),
        "second_period_one_certified":True,
        "known_first_ball_center_decimals":[str(x) for x in FIRST_CENTER],
        "known_first_ball_radius_fraction":str(FIRST_RADIUS),
        "disjoint_fixed_point_balls":True,
        "certified_coordinate_separation_lower_fraction":str(coordinate_gap),
        "lower_bound_on_fixed_point_distance_float":float(coordinate_gap),
        "two_distinct_attracting_fixed_points":True,
        "squared_input_entry_to_second_basin_certified":False,
        "scope":"two distinct locally attracting fixed points of exact-decimal real analytic zero-input RRC25 GRU; squared source trajectory entry is only FP64 discovery",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=two_fixed_points_certificate()
    if args.check:
        assert result["two_distinct_attracting_fixed_points"]
        assert result["second_invariant_block"] and result["second_period_one_certified"]
        assert not result["squared_input_entry_to_second_basin_certified"]
    blob=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
