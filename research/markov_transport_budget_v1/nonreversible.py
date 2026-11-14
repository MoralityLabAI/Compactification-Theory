"""MTB-01B: exact stationary, nonreversible Hilbert drift counterexample.

The directed walk on a square perimeter has uniform stationary law and
unit squared displacement per step, yet quarter-period displacement
grows quadratically in side length. Stationarity does not imply the
reversibility required by Markov type.

This control was added *after* the original MTB-01 protocol and first
exact CI pass; its mathematics and execution scope are separately tracked.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

SIZES=(2,4,8,16,32,64,128)


def square_point(side, index):
    if type(side) is not int or side<2 or (
            type(index) is not int or not 0<=index<4*side):
        raise ValueError("side >=2 and valid square-perimeter index required")
    m=side
    if index<m:
        return index,0
    if index<2*m:
        return m,index-m
    if index<3*m:
        return 3*m-index,m
    return 0,4*m-index


def sqdist(a,b):
    return sum((x-y)**2 for x,y in zip(a,b))


def perimeter_drift(side,steps):
    if type(side) is not int or side<2 or type(steps) is not int or steps<0:
        raise ValueError("side>=2 and nonnegative steps")
    n=4*side
    scores=tuple(sqdist(square_point(side,i),
                        square_point(side,(i+steps)%n))
                 for i in range(n))
    exact=F(sum(scores),n)
    if steps==1 and exact!=1:
        raise AssertionError("every directed step should have squared Euclidean cost one")
    return exact


def quarter_drift(side):
    m=side
    actual=perimeter_drift(m,m)
    expected=F(2*m*m+1,3)
    if actual!=expected:
        raise AssertionError("independent perimeter quadrature contradicts closed form")
    ratio=actual/m
    if ratio<=1:
        raise AssertionError("nonreversible chain did not violate M2=1 time budget")
    return {
        "side":m,"states":4*m,
        "stationary_distribution":"uniform on four times side vertices",
        "source_embedding":"points along square perimeter in R2",
        "steps":m,"one_step_squared_energy":"1",
        "quarter_drift_squared":str(actual),
        "analytic_quarter_drift_squared":str(expected),
        "erroneous_Hilbert_type2_upper_if_reversibility_assumed":str(m),
        "observed_squared_ratio_to_reversible_budget":str(ratio),
        "observed_ratio_float":float(ratio),
        "stationary":True,"reversible":False,
    }


def detailed_balance_violation(side):
    m=side
    if type(m) is not int or m<2:
        raise ValueError("side>=2")
    n=4*m
    # deterministic clockwise step P(0,1)=1, while P(1,0)=0
    pi=F(1,n)
    forward_flux=pi
    reverse_flux=F()
    return {
        "forward_edge":[0,1],
        "reverse_edge":[1,0],
        "stationary_uniform_mass":str(pi),
        "forward_stationary_flux":str(forward_flux),
        "reverse_stationary_flux":str(reverse_flux),
        "detailed_balance_rejected":forward_flux!=reverse_flux,
        "all_states_equal_indegree_outdegree":True,
    }


def report():
    rows=[quarter_drift(m) for m in SIZES]
    if any(not detailed_balance_violation(m)["detailed_balance_rejected"]
           for m in SIZES):
        raise AssertionError("directed cycle must violate reversibility")
    return {
        "schema":"mtb01b.nonreversible_square.v1",
        "status":"EXACT_STATIONARY_NONREVERSIBLE_HILBERT_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_papers":[327,374],
        "source_327_markov_type_applied_to_this_chain":False,
        "source_374_brenier_map_needed_for_this_control":False,
        "independent_postprotocol_extension":True,
        "rows":rows,
        "detailed_balance":detailed_balance_violation(4),
        "trained_neural_checkpoint_evaluated":False,
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path)
    args=ap.parse_args(argv)
    r=report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":r["status"],"rows":[
        {"side":x["side"],"drift2":x["quarter_drift_squared"],
         "ratio_to_reversible_budget":x["observed_ratio_float"]}
        for x in r["rows"]]},indent=2))


if __name__=="__main__":
    main()
