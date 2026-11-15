"""NOR-08C: exact rational audit of expanded second-attractor basin receipt."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RRC25=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
NOR07=HERE/"results"/"nor07_two_attractors_summary.json"
NOR08=HERE/"results"/"nor08_second_basin_summary.json"


def audit():
    prior_bytes=NOR07.read_bytes()
    previous=json.loads(prior_bytes)
    result=json.loads(NOR08.read_text())
    assert result["schema"]=="nor08.second_basin_expansion.frozen_summary.v1"
    checkpoint=sha256(RRC25.read_bytes()).hexdigest()
    assert checkpoint==result["checkpoint_sha256"]==previous["original_checkpoint_sha256"]
    assert sha256(prior_bytes).hexdigest()==result["original_nor07_summary_sha256"]
    assert previous["two_locally_attracting_fixed_points_certified"]
    attempts=result["radius_attempts"]
    assert attempts and attempts[0]["passed"] is False
    assert attempts[1]["passed"] is True
    assert all(Q(a["radius"])>Q(b["radius"]) for a,b in zip(attempts,attempts[1:]))
    sel=result["selected"]
    R=Q(sel["radius_fraction"])
    assert R==Q(attempts[1]["radius"])>Q(previous["second_attractor"]["radius_fraction"])
    K=Q(sel["block_operator_upper_fraction"])
    s=Q(sel["center_residual_upper_fraction"])
    margin=Q(sel["invariance_margin_fraction"])
    assert 0<K<1
    assert R-s-K*R==margin and margin>0
    L=Q(sel["one_step_operator_upper_fraction"])
    s1=Q(sel["one_step_center_residual_upper_fraction"])
    im=Q(sel["one_step_image_upper_fraction"])
    assert s1+L*s/(1-K)==im and im<R
    sep=Q(sel["coordinate_separation_lower_fraction"])
    nor04=json.loads((HERE/"results"/"nor04_invariant_ball.json").read_text())
    first=tuple(Q(z) for z in nor04["center_literals"])
    second=tuple(Q(z) for z in previous["second_attractor"]["center_decimals"])
    assert len(first)==len(second)==12
    assert abs(first[1]-second[1])-Q(nor04["radius_fraction"])-R==sep
    assert sep>0
    assert sel["same_period_one_fixed_point_as_nor07"]
    assert result["squared_family_source_basin_entry_certified"] is False
    return {
      "schema":"nor08.second_basin_expansion.rational_receipt_audit.v1",
      "checkpoint_sha256":checkpoint,
      "new_radius_float":float(R),
      "uniform_nonlinear_24_step_K_float":float(K),
      "invariant_margin_float":float(margin),
      "one_step_image_bound_float":float(im),
      "certified_second_basin_expanded":True,
      "squared_source_entry_proved":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    res=audit()
    if args.check: assert res["certified_second_basin_expanded"]
    print(json.dumps(res,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
