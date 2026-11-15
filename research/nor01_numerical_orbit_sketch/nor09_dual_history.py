"""NOR-09: exact two-source, two-anchor nonlinear history compactification.

A source-locked rational proof for the original trained RRC25 GRU.
The signed public-input hidden state has a certified first-basin entry
after 1308 zero-input steps; squared has a certified second-basin entry
after 660 zero-input steps. Both certified real-analytic nonlinear maps
are contractive on distinct invariant balls.

At each 24-step checkpoint after its own capture time, each stream
is exponentially close to its corresponding *rational center*, with
a small fixed-point-center approximation floor. The exact rational
bounds quantify a constructive two-anchor tail-clipping error.

Any representation that maps both streams to one terminal code has
additive cross-basin distortion above the certified invariant-ball gap.
The analytic finite-set Lp reduction by OpenAI Math #094 is separate.
All proof-obligation comparisons use fractions.Fraction, and displayed
floats are presentation only.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for directory in (ROOT,HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
DATA=HERE/"results"
CHECKPOINT=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
HORIZONS=(0,8,16,32,64,96,128,256)
DISPLAY_SCALE=10**30


def load(path):
    return json.loads((DATA/path).read_text())


def upward(x, scale=DISPLAY_SCALE):
    """Outward rational ceiling, deterministic and independent of FP64."""
    if x<0 or scale<=0:
        raise ValueError("upper bound requires nonnegative rational")
    y=x*scale
    return Q(-(-y.numerator//y.denominator),scale)


def source_constants():
    frozen=load("nor08_dual_source_capture.json")
    first=load("nor04_invariant_ball.json")
    second=load("nor08_expanded_second_ball.json")
    signed=load("nor06_certified_capture_summary.json")["signed"]
    squared=frozen["second_natural_source"]
    metric=load("nor08_two_basin_metric_code_capacity.json") if (DATA/"nor08_two_basin_metric_code_capacity.json").exists() else None
    if metric is None:
        # The exact lower bound is independently reproducible in
        # nor08_cross_basin, so favor its arithmetic implementation.
        from research.nor01_numerical_orbit_sketch.nor08_cross_basin import audit
        metric=audit()

    assert sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    assert frozen["original_trained_checkpoint_sha256"]==first["checkpoint_sha256"]==second["trained_checkpoint_sha256"]==SHA
    assert frozen["both_natural_sources_captured_in_distinct_verified_balls"] is True
    assert signed["certified"] and squared["certified_capture"]
    assert signed["zero_input_steps"]==1308 and squared["zero_input_steps"]==660
    assert signed["block_stride"]==squared["block_stride"]==12
    R1=Q(first["radius_fraction"]);R2=Q(second["radius_fraction"])
    K1=Q(first["operator_upper_fraction"])
    K2=Q(second["uniform_derivative_K_upper_fraction"])
    s1=Q(first["center_residual_upper_fraction"])
    s2=Q(second["center_residual_upper_fraction"])
    d1=Q(signed["center_distance_upper_fraction"])
    e1=Q(signed["trajectory_error_upper_fraction"])
    d2=Q(squared["target_center_distance_upper_fraction"])
    e2=Q(squared["state_error_upper_fraction"])
    assert 0<K1<1 and 0<K2<1
    assert d1+e1+Q(signed["exact_ball_clearance_fraction"])==R1
    assert d2+e2+Q(squared["exact_positive_clearance_fraction"])==R2
    assert s1+K1*R1<R1 and s2+K2*R2<R2

    dfix1=s1/(1-K1)
    dfix2=s2/(1-K2)
    A1=d1+e1+dfix1
    A2=d2+e2+dfix2
    gamma1=K1*K1 # two certified 12-step blocks
    gamma2=K2    # one certified 24-step block
    gap=Q(metric["first_second_ball_separation_lower_fraction"])
    assert gap>5 and gap > R1+R2

    return {
        "capture_1":1308,"capture_2":660,
        "first_ball_radius":R1,"second_ball_radius":R2,
        "first_block_K12":K1,"second_block_K24":K2,
        "first_K24":gamma1,"second_K24":gamma2,
        "first_fixed_offset":dfix1,"second_fixed_offset":dfix2,
        "first_source_amplitude":A1,"second_source_amplitude":A2,
        "gap":gap,
        "first_trace":signed["exact_trace_sha256"],
        "second_trace":squared["exact_capture_trace_sha256"],
        "first_source_sha":signed["public_features_sha256"],
        "second_source_sha":squared["public_features_sha256"],
    }


def horizon_record(k,c):
    if k<0 or k>4096:
        raise ValueError("checkpoint horizon outside 0..4096")
    first_true=c["first_source_amplitude"]*c["first_K24"]**k
    second_true=c["second_source_amplitude"]*c["second_K24"]**k
    # Using the *known rational centers* as clipping anchors adds the
    # independently proved center-to-true-fixed-point offsets.
    first_center=first_true+c["first_fixed_offset"]
    second_center=second_true+c["second_fixed_offset"]
    tail=2*max(first_center,second_center)
    # The source states remain inside the original certified invariant
    # balls at 24-step checkpoints. Their true cross-distance >= gap.
    # Identifying them with the same metric terminal code therefore
    # forces at least gap additive error at every such checkpoint.
    return {
        "extra_24_step_blocks":k,
        "signed_original_zero_input_steps":c["capture_1"]+24*k,
        "squared_original_zero_input_steps":c["capture_2"]+24*k,
        "signed_to_fixed_point_upper_fraction":str(upward(first_true)),
        "squared_to_fixed_point_upper_fraction":str(upward(second_true)),
        "signed_to_rational_anchor_upper_fraction":str(upward(first_center)),
        "squared_to_rational_anchor_upper_fraction":str(upward(second_center)),
        "two_anchor_clipped_tail_pairwise_error_upper_fraction":str(upward(tail)),
        "two_anchor_clipped_tail_pairwise_error_upper_float":float(upward(tail)),
        "one_code_cross_basin_additive_error_lower_fraction":str(c["gap"]),
        "one_code_cross_basin_additive_error_lower_float":float(c["gap"]),
        "source_states_both_remain_in_their_certified_balls_at_this_checkpoint":True,
    }


def run(horizons=HORIZONS):
    c=source_constants()
    rows=[horizon_record(int(k),c) for k in horizons]
    if len(set(int(x) for x in horizons))!=len(horizons):
        raise ValueError("duplicate horizons")
    assert all(Q(row["two_anchor_clipped_tail_pairwise_error_upper_fraction"])>0
               for row in rows)
    assert all(Q(row["one_code_cross_basin_additive_error_lower_fraction"])>5
               for row in rows)
    return {
        "schema":"nor09.certified.two_natural_sources.24_step_history.v1",
        "trained_checkpoint_sha256":SHA,
        "source_capture_1":"signed",
        "source_capture_2":"squared",
        "signed_capture_zero_input_steps":c["capture_1"],
        "squared_capture_zero_input_steps":c["capture_2"],
        "signed_capture_trace_sha256":c["first_trace"],
        "squared_capture_trace_sha256":c["second_trace"],
        "signed_public_features_sha256":c["first_source_sha"],
        "squared_public_features_sha256":c["second_source_sha"],
        "sampled_clock_stride":24,
        "signed_K24_exact_fraction":str(c["first_K24"]),
        "squared_K24_exact_fraction":str(c["second_K24"]),
        "signed_initial_fixed_distance_upper_fraction":str(c["first_source_amplitude"]),
        "squared_initial_fixed_distance_upper_fraction":str(c["second_source_amplitude"]),
        "signed_center_to_fixed_offset_upper_fraction":str(c["first_fixed_offset"]),
        "squared_center_to_fixed_offset_upper_fraction":str(c["second_fixed_offset"]),
        "certified_cross_ball_gap_lower_fraction":str(c["gap"]),
        "required_terminal_label_count":2,
        "required_fixed_width_label_bits":1,
        "rows":rows,
        "p_selection":"Theorem applies to finite pools of two captured 24-step trajectories; OpenAI Math #094 finite-set Lp result remains existential.",
        "important_scope":"All-horizon tail clipping for 24-step sampled nonlinear GRU histories after two DIFFERENT certified source capture times. Bounds for unobserved intermediate steps require an extra per-offset norm certificate.",
        "embedding_error_definition":"If finite-pool observed distances including the two rational centers have maximum additive error E_M, all-horizon clipped representation error is <= E_M + the row's certified two_anchor_clipped_tail_pairwise_error_upper_fraction.",
        "zero_input_real_analytic_map":True,
        "source_truth_or_private_future_used":False,
        "native_task_gain_claimed":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--horizons",type=str,default=",".join(map(str,HORIZONS)))
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    horizons=tuple(int(v) for v in args.horizons.split(","))
    result=run(horizons)
    if args.check:
        assert result["required_terminal_label_count"]==2
        assert result["required_fixed_width_label_bits"]==1
        assert result["signed_capture_zero_input_steps"]==1308
        assert result["squared_capture_zero_input_steps"]==660
        by_k={r["extra_24_step_blocks"]:r for r in result["rows"]}
        if 64 in by_k:
            assert Q(by_k[64]["two_anchor_clipped_tail_pairwise_error_upper_fraction"])<Q(12,10**12)
        if 128 in by_k:
            assert Q(by_k[128]["two_anchor_clipped_tail_pairwise_error_upper_fraction"])<Q(4,10**14)
    blob=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob,end="")


if __name__=="__main__":
    main()
