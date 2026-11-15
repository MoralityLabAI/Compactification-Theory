"""NOR-06C: independently audit the frozen rational capture receipt.

The expensive NOR-06B GitHub Actions job computes the certificate; this
separate 2-second auditor verifies exact Fraction clearance, source-lock
hashes and positive/negative scope. It does not re-execute 109 blocks,
and therefore cites the full independent workflow artifact and trace hash.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SUMMARY=Path(__file__).resolve().parent/"results"/"nor06_certified_capture_summary.json"
TRAINED=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
NOR04=Path(__file__).resolve().parent/"results"/"nor04_invariant_ball.json"
NOR05=Path(__file__).resolve().parent/"results"/"nor05_period_one.json"

def verified_summary():
    receipt=json.loads(SUMMARY.read_text())
    trained_hash=sha256(TRAINED.read_bytes()).hexdigest()
    nor04=json.loads(NOR04.read_text())
    nor05=json.loads(NOR05.read_text())
    assert receipt["schema"]=="nor06.rational_capture_summary.v1"
    assert trained_hash==receipt["original_checkpoint_sha256"]
    assert nor04["checkpoint_sha256"]==trained_hash
    assert nor04["banach_fixed_point_certificate"]
    assert nor05["checkpoint_sha256"]==trained_hash
    assert nor05["one_step_fixed_point_proved"]
    signed=receipt["signed"];squared=receipt["squared"]
    assert signed["family"]=="signed" and signed["certified"]
    assert squared["family"]=="squared" and not squared["certified_capture"]
    assert signed["zero_input_steps"]==signed["block_index"]*signed["block_stride"]
    assert signed["zero_input_steps"]==1308
    R=Q(receipt["fixed_ball_radius_fraction"])
    assert R==Q(nor04["radius_fraction"])==Q(nor05["all_depth_block_prerequisite"]["R_fraction"])
    center=Q(signed["center_distance_upper_fraction"])
    err=Q(signed["trajectory_error_upper_fraction"])
    clearance=Q(signed["exact_ball_clearance_fraction"])
    assert center>0 and err>0 and clearance>0
    assert center+err+clearance==R
    assert Q(squared["center_distance_upper_fraction"]) > R
    assert Q(squared["trajectory_error_upper_fraction"])>0
    assert squared["blocks_checked"]==12
    for rec in (signed,squared):
        for field in ("public_features_sha256","source_case_identifier_sha256","exact_trace_sha256"):
            digest=rec[field]; assert len(digest)==64
            int(digest,16)
    return {
        "schema":"nor06.rational_capture_summary.audit.v1",
        "trained_checkpoint_sha256":trained_hash,
        "signed": {
            "certified":True,
            "zero_input_steps":signed["zero_input_steps"],
            "center_distance_upper_float":float(center),
            "state_error_upper_float":float(err),
            "exact_clearance_float":float(clearance),
            "trace_sha256":signed["exact_trace_sha256"],
        },
        "squared": {
            "certified_capture":False,
            "blocks_checked":squared["blocks_checked"],
            "trace_sha256":squared["exact_trace_sha256"],
        },
        "proof_scope":"audit of committed source-locked GitHub Actions rational receipt, not new execution",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    result=verified_summary()
    print(json.dumps(result,indent=2,sort_keys=True))
    if a.check:
        assert result["signed"]["certified"]
        assert not result["squared"]["certified_capture"]

if __name__=="__main__":
    main()
