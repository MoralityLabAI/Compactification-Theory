"""NOR-08E: verify two independent natural-input basin capture certificates.

Combines NOR-06B signed-family interval entry (first attracting basin)
with NOR-08B squared-family interval entry (second attracting basin).
Every acceptance decision uses exact Fraction arithmetic and matches
the original checkpoint and public-source fingerprints.

The full interval traces were independently executed by GitHub Actions.
This fast audit checks their committed exact witness summaries and the
already kernel-audited MathCan basin-code injectivity premises.
"""
from __future__ import annotations

import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
from research.nor01_numerical_orbit_sketch.nor08_cross_basin import audit as cross_audit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREVIOUS=HERE/"results"/"nor06_certified_capture_summary.json"
NEW=HERE/"results"/"nor08_dual_source_capture.json"
SECOND=HERE/"results"/"nor08_expanded_second_ball.json"
FIRST=HERE/"results"/"nor04_invariant_ball.json"
CHECKPOINT=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def check():
    first=json.loads(PREVIOUS.read_text())
    second=json.loads(NEW.read_text())
    ball=json.loads(SECOND.read_text())
    firstball=json.loads(FIRST.read_text())
    cross=cross_audit()
    assert sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    assert first["original_checkpoint_sha256"]==second["original_trained_checkpoint_sha256"]==SHA
    assert ball["trained_checkpoint_sha256"]==firstball["checkpoint_sha256"]==SHA
    assert second["both_natural_sources_captured_in_distinct_verified_balls"] is True

    a=first["signed"]
    b=second["second_natural_source"]
    assert a["certified"] and second["first_natural_source"]["family"]=="signed"
    assert a["zero_input_steps"]==a["block_index"]*a["block_stride"]==1308
    assert Q(a["center_distance_upper_fraction"])+Q(a["trajectory_error_upper_fraction"])+Q(a["exact_ball_clearance_fraction"])==Q(firstball["radius_fraction"])
    assert Q(a["exact_ball_clearance_fraction"])>0
    assert second["first_natural_source"]["capture_artifact_hash"]==a["exact_trace_sha256"]

    assert b["family"]=="squared" and b["certified_capture"]
    assert b["source_case_id_sha256"]==first["squared"]["source_case_identifier_sha256"]
    assert b["public_features_sha256"]==first["squared"]["public_features_sha256"]
    assert b["zero_input_steps"]==b["certified_block_index"]*b["block_stride"]==660
    assert b["target_radius_fraction"]==ball["radius_fraction"]
    assert b["target_24_step_contraction_K_fraction"]==ball["uniform_derivative_K_upper_fraction"]
    d=Q(b["target_center_distance_upper_fraction"])
    e=Q(b["state_error_upper_fraction"])
    clearance=Q(b["exact_positive_clearance_fraction"])
    R2=Q(ball["radius_fraction"])
    assert d+e+clearance==R2 and clearance>0
    assert b["exact_capture_trace_sha256"] and len(b["exact_capture_trace_sha256"])==64

    separation=Q(cross["first_second_ball_separation_lower_fraction"])
    assert separation>Q(5)
    assert second["distinct_label_lower_bound"]>=2
    assert second["required_terminal_label_bits_lower_bound"]>=1
    assert second["distinct_label_lower_bound"]==cross["minimum_distinct_terminal_code_labels"]
    return {
        "schema":"nor08.dual_natural_source_capture.exact_receipt_audit.v1",
        "checkpoint_sha256":SHA,
        "first_captured_source_family":"signed",
        "first_steps":1308,
        "second_captured_source_family":"squared",
        "second_steps":660,
        "squared_capture_distance_upper":float(d),
        "squared_capture_error_upper":float(e),
        "squared_capture_clearance":float(clearance),
        "source_states_cross_basin_separation_lower":float(separation),
        "required_terminal_code_labels":2,
        "both_source_capture_certificates":True,
        "source_future_truth_used":False,
        "scope":"exact audit of two previously executed rational interval traces and their distinct invariant balls"
    }


if __name__=="__main__":
    print(json.dumps(check(),sort_keys=True,indent=2))
