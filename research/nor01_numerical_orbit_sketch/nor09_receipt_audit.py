"""NOR-09C: exact-arithmetic audit of the frozen dual-source receipt.

Recomputes the four recorded long-horizon clipping bounds from the
source-locked NOR-04/06/07/08 rational certificates. Exact Fraction
comparisons check the two independent capture clocks, one-bit code
separation, and nonzero rational-center residual floor.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from research.nor01_numerical_orbit_sketch import nor09_dual_history as proof

FROZEN=Path(__file__).resolve().parent/"results"/"nor09_dual_history_summary.json"


def audit():
    frozen=json.loads(FROZEN.read_text())
    assert frozen["schema"]=="nor09.certified.two_natural_sources.frozen_audit.v1"
    assert frozen["trained_checkpoint_sha256"]==proof.SHA
    assert frozen["stride"]==24
    const=proof.source_constants()
    assert frozen["source_captures"]["signed_extra_zero_steps"]==const["capture_1"]==1308
    assert frozen["source_captures"]["squared_extra_zero_steps"]==const["capture_2"]==660
    assert frozen["source_captures"]["signed_trace_sha256"]==const["first_trace"]
    assert frozen["source_captures"]["squared_trace_sha256"]==const["second_trace"]
    assert frozen["required_terminal_code_labels"]==2
    assert frozen["fixed_width_label_bits_lower_bound"]==1
    assert Q(frozen["certified_cross_basin_gap_lower_fraction"])==const["gap"]>5
    rows={}
    for key,value in frozen["exact_certified_tail_pairwise_error_ceilings"].items():
        index=int(key)
        snap=proof.horizon_record(index,const)
        assert Q(value)==Q(snap["two_anchor_clipped_tail_pairwise_error_upper_fraction"])
        assert snap["signed_original_zero_input_steps"]==frozen["sampled_clocks"][key]["signed"]
        assert snap["squared_original_zero_input_steps"]==frozen["sampled_clocks"][key]["squared"]
        rows[key]=float(Q(value))
    assert rows["64"] < 1.2e-11
    assert rows["96"] < 4e-14
    assert rows["128"] < 4e-14
    assert rows["256"] < 4e-14
    positive_center_floor=2*max(const["first_fixed_offset"],const["second_fixed_offset"])
    assert positive_center_floor>0
    assert Q(frozen["exact_certified_tail_pairwise_error_ceilings"]["256"])>=positive_center_floor
    assert rows["256"]<=rows["128"]<=rows["96"]<=rows["64"]
    return {
        "schema":"nor09.certified.dual_natural_source.audit.v1",
        "source_checkpoint_sha256":proof.SHA,
        "certified_separation_lower":float(const["gap"]),
        "two_anchor_tail_errors":rows,
        "certified_anchor_offset_floor_float":float(positive_center_floor),
        "exact_source_capture_witnesses":2,
        "required_terminal_code_bits":1,
        "only_24_step_sampled_histories":True,
        "native_ranking_capability_gain_claimed":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    out=audit()
    if args.check:
        assert out["exact_source_capture_witnesses"]==2
        assert out["required_terminal_code_bits"]==1
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
