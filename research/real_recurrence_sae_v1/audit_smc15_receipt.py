"""Independent standard-library audit of the first frozen SMC-15 run.

Consumes only the archived calibrated pilot and fresh-case JSON, with
the original uploaded archive sha256. No model or test generator is loaded.
The report distinguishes a data-integrity PASS from the registered
scientific gate, which was UNDERPOWERED in every teacher-family cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

PILOT_SHA = "c8883ca53e1804cc5d39da0f32acfed6eb57bc7a94fd059201986f0a7fa21700"
FULL_SHA = "3ad00ce9f190363954b0b90df8ed33966fc687c520ddf9665c6575b62d75070c"
ARCHIVE_SHA = "fcfee553a6351e529dc276e5e15feb9e3cb713d4477b1310eb5ddda1f478e31c"
ORIGINAL = "7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087"
INDEPENDENT = "1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3"
CONDITIONED = "3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07"
FAMILIES = ("smc15_novel30", "smc15_novel34")
COMPONENTS = ("pretrainedA_71", "newB_501_SAE_71", "newB_501_SAE_89",
              "newB_701_SAE_71", "newB_701_SAE_89")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_archive(path):
    raw = Path(path).read_bytes()
    if sha256(raw) != ARCHIVE_SHA:
        raise ValueError("original SMC-15 upload archive SHA mismatch")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if names != {"calibration_pilot.json", "fresh_full.json"}:
            raise ValueError("unexpected files in source upload")
        pilot_raw = z.read("calibration_pilot.json")
        full_raw = z.read("fresh_full.json")
    if sha256(pilot_raw) != PILOT_SHA or sha256(full_raw) != FULL_SHA:
        raise ValueError("raw SMC-15 JSON receipt SHA mismatch")
    return json.loads(pilot_raw), json.loads(full_raw)


def read_jsons(pilot_file, full_file):
    p = Path(pilot_file).read_bytes()
    f = Path(full_file).read_bytes()
    if sha256(p) != PILOT_SHA or sha256(f) != FULL_SHA:
        raise ValueError("independent JSON file SHA mismatch")
    return json.loads(p), json.loads(f)


def assert_protocol(pilot, full):
    if pilot["status"] != "pilot" or full["status"] != "full":
        raise ValueError("pilot/full source stages mislabeled")
    if pilot["new_puzzle_manifest"]["type"] != "SMC15/calibration-only-pilot":
        raise ValueError("pilot accessed a non-calibration cohort")
    if pilot["new_puzzle_manifest"]["n"] != 4:
        raise ValueError("pilot roster count changed")
    if set(full["data"]) != set(FAMILIES):
        raise ValueError("full roster differs from two new families")
    for x in (pilot, full):
        if x["original_sae_receipt_sha256"] != ORIGINAL or x["independent_teacher_receipt_sha256"] != INDEPENDENT or x["source_conditioned_receipt_sha256"] != CONDITIONED:
            raise ValueError("source receipt hashes differ from frozen registration")
        if x["torch_compiled"] is not False or x["calibration_only_first"] is not True:
            raise ValueError("model runtime/protocol changed")
        if x["radius_small"] != .08 or x["radius_verify"] != .2 or x["predeclared_sigma_threshold"] != .10:
            raise ValueError("registered numerical doses changed")
        for index in range(4):
            record = x["calibration_permutation_receipts"][str(index)]
            if record["hashes"]["matched_to_original_smc13c_receipt"] is not True:
                raise ValueError("unverified SMC13C feature assignment")
            match = record["permutations"]["matched"]
            hard = record["permutations"]["clue_shuffled"]
            if sorted(match) != list(range(96)) or sorted(hard) != list(range(96)):
                raise ValueError("feature correspondence is not a permutation")
            # Recompute both binary correspondence receipts with the exact producer JSON encoding.
            for arr, key in ((match, "matched_assignment_sha256"),
                             (hard, "clue_shuffled_assignment_sha256")):
                if sha256(json.dumps(arr).encode()) != record["hashes"][key]:
                    raise ValueError("feature assignment checksum fails replay")
    manifest = full["new_puzzle_manifest"]
    if manifest["sources"] != {"smc15_novel30":202615030000,
                               "smc15_novel34":202615040000}:
        raise ValueError("test cohort seeds changed after preregistration")
    if manifest["puzzles_unique"] is not True or manifest["targets_unique"] is not True:
        raise ValueError("source puzzle identity gate failed")
    if manifest["excluded_previous_target_count"] != 512:
        raise ValueError("old source exclusion roster changed")
    for f in FAMILIES:
        rec = manifest["cohorts"][f]
        if rec["n"] != 64 or rec["min_rectangles"] <= 0 or len(rec["fingerprint_sha256"]) != 64:
            raise ValueError("bad source-disjoint cohort manifest")


def inspect_component(full, family, component):
    group = full["data"][family][component]
    rows = group["per_puzzle"]
    if len(rows) != 64 or sorted(row["puzzle_index"] for row in rows) != list(range(64)):
        raise ValueError("incorrect independent puzzle roster")
    goals = []
    low_puzzle = []
    high_puzzle = []
    for row in rows:
        if len(row["goal_trials"]) != 8 or row["n_measured"] != 8:
            raise ValueError("missing/invalid preregistered goal")
        low = []
        high = []
        for goal in row["goal_trials"]:
            if goal["status"] != "measured":
                raise ValueError("unexpected goal-level abstention")
            selected = goal["selected_atoms"]
            top3 = goal["selected_top3"]
            if len(selected) != 2 or len(set(selected)) != 2 or not all(0 <= i < 96 for i in selected):
                raise ValueError("invalid physical SAE pair selection")
            if not (top3[0] > top3[1] > top3[2] >= 0):
                raise ValueError("SAE selection violated strict top3 gate")
            if not (goal["strictly_active"] and goal["source_clean_discrepancy"] <= 1e-8):
                raise ValueError("native source parity failed")
            for name in ("dose_min", "dose_max"):
                if abs(goal[name] - 0.20) > 1e-10:
                    raise ValueError("min/max physical intervention dose mismatch")
            if abs(goal["joint_physical_norm"] - goal["sham_physical_norm"]) > 1e-10:
                raise ValueError("matched nuisance norm mismatched")
            if not (0 <= goal["token"] < 81 and
                    0 <= goal["sigma_output_min"] <= goal["sigma_output_max"]):
                raise ValueError("invalid causal tangent geometry")
            if not (goal["sigma_physical_min"] > 0 and
                    math.isfinite(goal["minmax_finite_response_ratio"])):
                raise ValueError("invalid finite causal response")
            if goal["condition_low"] != (goal["sigma_output_min"] < .1):
                raise ValueError("outcome threshold differs from preregistration")
            goals.append(goal)
            if goal["condition_low"]:
                low.append(goal["minmax_finite_response_ratio"])
            else:
                high.append(goal["minmax_finite_response_ratio"])
        if low:
            low_puzzle.append(sum(low)/len(low))
        if high:
            high_puzzle.append(sum(high)/len(high))
    summary = group["summary"]
    n_low = sum(g["condition_low"] for g in goals)
    n_high = len(goals) - n_low
    if summary["n_low_conditioning_goals"] != n_low or summary["n_high_conditioning_goals"] != n_high:
        raise ValueError("published conditioning counts fail independent recomputation")
    if summary["n_measured_goals"] != 512:
        raise ValueError("published measured count differs")
    if summary["n_low_conditioning_puzzles"] != len(low_puzzle) or summary["n_high_conditioning_puzzles"] != len(high_puzzle):
        raise ValueError("published independent puzzle counts differ")
    if low_puzzle:
        if abs(summary["low_puzzle_mean_finite_ratio"] -
               statistics.mean(low_puzzle)) > 1e-12:
            raise ValueError("low sigma mean discrepancy")
    if high_puzzle:
        if abs(summary["high_puzzle_mean_finite_ratio"] -
               statistics.mean(high_puzzle)) > 1e-12:
            raise ValueError("high sigma mean discrepancy")
    eligible = len(low_puzzle) >= 10 and len(high_puzzle) >= 10 and n_low >= 30 and n_high >= 30
    if eligible:
        raise ValueError("official reported all-underpowered gate unexpectedly eligible")
    if summary["status"] != "underpowered_groups" or summary["registered_conditional_gate"]:
        raise ValueError("registered scientific gate status changed")
    return {
        "component": component, "family": family, "n_puzzles": len(rows),
        "goal_interventions": len(goals), "low_sigma_goals": n_low,
        "low_sigma_puzzles": len(low_puzzle),
        "median_output_sigma_min": statistics.median(g["sigma_output_min"] for g in goals),
        "min_physical_sigma_min_among_low": min(
            (g["sigma_physical_min"] for g in goals if g["condition_low"]),default=None),
        "mean_abs_joint_margin_nonadditivity": statistics.mean(
            abs(g["native_joint_margin_nonadditivity"]) for g in goals),
        "low_group_mean_finite_ratio": statistics.mean(low_puzzle) if low_puzzle else None,
        "high_group_mean_finite_ratio": statistics.mean(high_puzzle) if high_puzzle else None,
        "finite_min_lowers_native_9logit_effect_fraction": sum(
            g["minmax_finite_response_ratio"] < 1 for g in goals)/len(goals),
        "registered_scientific_gate": "UNDERPOWERED",
    }


def audit(pilot, full):
    assert_protocol(pilot, full)
    rows = [inspect_component(full, f, c) for f in FAMILIES for c in COMPONENTS]
    for f in FAMILIES:
        co = full["source_to_target_coactivity"][f]
        if len(co) != 4:
            raise ValueError("missing transferred feature coactivity comparisons")
        for pair in co:
            if (pair["n_goals"] != 512 or pair["n_source_strict"] != 512 or
                    not 0 <= pair["n_both_mapped_target_active"] <= 512):
                raise ValueError("inconsistent source/target coactivity count")
    return {
        "schema":"SMC15/independent-raw-receipt-standard-library-audit-v1",
        "source_zip_sha256":ARCHIVE_SHA, "pilot_json_sha256":PILOT_SHA,
        "full_json_sha256":FULL_SHA, "protocol_hash_checks":"PASS",
        "original_feature_correspondence_replay":"PASS, 4 matched and 4 clue-shuffled hashes",
        "native_zero_dose_and_sham_checks":"PASS on every goal receipt",
        "n_distinct_fresh_puzzles":128,
        "n_component_goal_measurements":5120,
        "registered_scientific_gate":"UNDERPOWERED, all 10 teacher/SAE/cohort cells",
        "independent_target_coactivity_audit":"NOT_FULLY_REPLAYABLE: archived coactivity has aggregate counts but omits target top4 masks per token",
        "teacher_family":rows,
        "source_to_target_coactivity_reported_by_runner":full["source_to_target_coactivity"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--pilot",type=Path)
    parser.add_argument("--full",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    opts = parser.parse_args()
    if opts.archive and not (opts.pilot or opts.full):
        pilot, full = load_archive(opts.archive)
    elif opts.pilot and opts.full and not opts.archive:
        pilot, full = read_jsons(opts.pilot,opts.full)
    else:
        parser.error("provide --archive ZIP or both --pilot and --full JSON")
    report = audit(pilot, full)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC15_RECEIPT_AUDIT_PASS",json.dumps({
        "data_checks":report["protocol_hash_checks"],
        "scientific_gate":report["registered_scientific_gate"],
        "n_goals":report["n_component_goal_measurements"],
        "out":str(opts.out)}))


if __name__ == "__main__":
    main()
