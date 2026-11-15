"""NOR-10: constructive optimal one-bit terminal code for two captured GRU basins.

A self-contained source-locked certificate combining:
* two independently verified natural-input captures (NOR-06, NOR-08),
* invariant nonlinear contraction of both original trained GRU basins,
* exact rational two-center Euclidean geometry,
* a single rational distance parameter shared by encoder and decoder.

At every 24-step sampled checkpoint after each source's own capture,
the terminal encoder needs only one bit: first basin -> 0; second basin
-> rational lower approximation of the inter-anchor distance.
The gap quantization is outward-certified with integer square roots.

The terminal metric error is bounded by
max(2*B1(M), 2*B2(M), B1(M)+B2(M)+1/10**digits)
for all sampled times m,n >= M. B1 and B2 are the certified nonlinear
distances to the rational anchors from NOR-09, not empirical errors.

The construction assumes the basin identity is already known for these
two specifically certified source streams. It is not an inference model
for unseen inputs, a trained SAE, or a proof-search performance gain.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for d in (HERE,ROOT):
    if str(d) not in sys.path:
        sys.path.insert(0,str(d))

from research.nor01_numerical_orbit_sketch import nor09_dual_history as source

QUANTIZED_DIGITS=18
DEN=10**QUANTIZED_DIGITS
EPOCHS=(0,32,64,96,128,256)
DATA=HERE/"results"


def center_gap_interval():
    first=json.loads((DATA/"nor04_invariant_ball.json").read_text())
    second=json.loads((DATA/"nor07_two_attractors_summary.json").read_text())
    assert first["checkpoint_sha256"]==second["original_checkpoint_sha256"]==source.SHA
    c1=[Q(x) for x in first["center_literals"]]
    c2=[Q(x) for x in second["second_attractor"]["center_decimals"]]
    assert len(c1)==len(c2)==12
    sq=sum(((x-y)**2 for x,y in zip(c1,c2,strict=True)),Q(0))
    if sq<=0:
        raise ValueError("certified attractor centers must be distinct")
    qscaled=isqrt(sq.numerator*DEN*DEN//sq.denominator)
    lo=Q(qscaled,DEN)
    hi=Q(qscaled+1,DEN)
    assert lo*lo<=sq<=hi*hi
    return {"distance_squared_fraction":str(sq),
            "lower_fraction":str(lo),
            "upper_fraction":str(hi),
            "quantizer_error_fraction":str(hi-lo),
            "distance_lower_float":float(lo),
            "source_anchors_dimension":len(c1)}


def encoded_terminal_code(basin,quantizer):
    if basin=="signed":
        return Q(0)
    if basin=="squared":
        return Q(quantizer["lower_fraction"])
    raise ValueError("basin label must belong to one of the two certified sources")


def epoch_record(m,constants,quantizer):
    full=source.horizon_record(m,constants)
    a=Q(full["signed_to_rational_anchor_upper_fraction"])
    b=Q(full["squared_to_rational_anchor_upper_fraction"])
    eta=Q(quantizer["quantizer_error_fraction"])
    within_first=2*a
    within_second=2*b
    cross=a+b+eta
    bound=max(within_first,within_second,cross)
    gap=constants["gap"]
    assert gap>5 and bound>0
    return {
        "extra_24_step_blocks":m,
        "signed_original_zero_input_steps":full["signed_original_zero_input_steps"],
        "squared_original_zero_input_steps":full["squared_original_zero_input_steps"],
        "signed_center_radius_upper_fraction":str(a),
        "squared_center_radius_upper_fraction":str(b),
        "within_signed_pair_error_upper_fraction":str(within_first),
        "within_squared_pair_error_upper_fraction":str(within_second),
        "cross_basin_pair_error_upper_fraction":str(cross),
        "one_bit_terminal_all_pair_error_upper_fraction":str(bound),
        "one_bit_terminal_all_pair_error_upper_float":float(bound),
        "zero_bit_cross_basin_error_lower_fraction":str(gap),
        "zero_bit_cross_basin_error_lower_float":float(gap),
        "rigorous_zero_bit_to_one_bit_worst_error_ratio_lower_float":float(gap/bound),
        "one_bit_code_values":{"signed":"0","squared":quantizer["lower_fraction"]},
        "source_mode_label_is_given_by_two_independently_certified_capture_witnesses":True,
    }


def run(epochs=EPOCHS):
    constants=source.source_constants()
    quantizer=center_gap_interval()
    assert Q(quantizer["quantizer_error_fraction"])==Q(1,DEN)
    assert encoded_terminal_code("signed",quantizer)==0
    assert encoded_terminal_code("squared",quantizer)>5
    if len(set(epochs))!=len(epochs):
        raise ValueError("duplicate horizons")
    records=[epoch_record(int(m),constants,quantizer) for m in epochs]
    return {
        "schema":"nor10.optimal_one_bit_two_captured_basin_tail_code.v1",
        "trained_checkpoint_sha256":source.SHA,
        "source_1":"signed",
        "source_2":"squared",
        "source_capture_clocks":[constants["capture_1"],constants["capture_2"]],
        "sampled_clock_stride":24,
        "inter_anchor_quantization":quantizer,
        "terminal_codebook":{"signed":"0","squared":quantizer["lower_fraction"]},
        "terminal_runtime_basin_label_bits":1,
        "shared_rational_distance_metadata_stored_once":True,
        "certified_cross_basin_gap_lower_fraction":str(constants["gap"]),
        "rows":records,
        "scope":"Explicit 1D two-code late-history metric for source-verified signed and squared RRC25 trajectories at post-capture 24-step checkpoints; no unseen-input basin classifier is trained.",
        "claim_boundaries":{
            "finite_prefix_embedding_covered":False,
            "per_intermediate_step_24_residue_bounds_covered":False,
            "native_proof_ranking_improvement":False,
            "all_source_basin_labels_inferred_by_model":False,
            "terminal_codes_carry_basin_labels_from_existing_verified_source_witnesses":True,
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--epochs",type=str,default=",".join(map(str,EPOCHS)))
    args=parser.parse_args()
    epochs=tuple(int(v) for v in args.epochs.split(","))
    result=run(epochs)
    if args.check:
        assert result["trained_checkpoint_sha256"]==source.SHA
        assert result["terminal_runtime_basin_label_bits"]==1
        r={x["extra_24_step_blocks"]:x for x in result["rows"]}
        for key in (64,128):
            if key in r:
                b=Q(r[key]["one_bit_terminal_all_pair_error_upper_fraction"])
                assert b>0
                assert b<Q(12,10**12) if key==64 else b<Q(4,10**14)
                assert Q(r[key]["zero_bit_cross_basin_error_lower_fraction"])>5
    blob=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
