"""NOR-14 independent receipt and original per-case arithmetic audit.

Only standard-library hash, arithmetic and JSON; this does not import
the training/evaluation implementation. Pass --full the unzipped artifact
nor14_full.json to audit every recorded native decision and comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "results" / "nor14_first_full_receipt.json"
EXPECTED_SHA = "57b4b82550dce6228356dae924264a95f38d66b3974c038b574eadcad9f7df2d"
EXPECTED_MODEL = "54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def sign_p(positive, negative):
    n = positive + negative
    if not n:
        return 1.
    return min(1., 2*sum(comb(n, k) for k in range(min(positive, negative)+1))/2**n)


def paired(left, right):
    diff = defaultdict(int)
    positive = negative = count = 0
    for a, b in zip(left, right, strict=True):
        assert a["public_features_sha256"] == b["public_features_sha256"]
        assert a["family"] == b["family"]
        if a["family"] == "late_source":
            continue
        count += 1
        d = b["future_correct"] - a["future_correct"]
        positive += d == 1
        negative += d == -1
        assert d in (-1, 0, 1)
        diff[a["public_features_sha256"]] += d
    p_groups = sum(z > 0 for z in diff.values())
    n_groups = sum(z < 0 for z in diff.values())
    return {
        "cases": count, "unique_public_inputs": len(diff),
        "paired_gains": positive, "paired_losses": negative,
        "net": positive-negative,
        "positive_unique_groups": p_groups,
        "negative_unique_groups": n_groups,
        "unadjusted_exact_group_sign_p": sign_p(p_groups, n_groups),
    }


def validate_receipt(r):
    assert r["schema"] == "nor14.first_full_hosted_receipt.v1"
    assert r["full_json_sha256"] == EXPECTED_SHA
    assert r["original_checkpoint_sha256"] == EXPECTED_MODEL
    assert r["training_source_cases"] == 96
    assert r["dev_source_cases"] == 36
    assert r["public_template_checks_shared"] == 1536
    assert r["trained_readout_slots_per_arm"] == 400
    assert r["fresh_cases"] == 288
    assert r["nonlate_distinct_input_cases"] == 240
    assert r["distinct_model_inputs"] == 241
    assert r["late_source_cases"] == 48
    assert r["late_source_distinct_model_inputs"] == 1
    assert len(r["fresh_cohort_manifests"]) == 4
    assert len(set(r["fresh_cohort_manifests"])) == 4
    assert r["development_global_frozen"]["kind"] == "endpoint_shuffled"
    assert r["arms"]["original"] == [178, 22, 794, 165]
    assert r["arms"]["endpoint_basin"] == [173, 20, 797, 163]
    assert r["arms"]["trajectory_basin"] == [177, 20, 811, 167]
    assert r["primary_endpoint_vs_original"]["gains"] == 8
    assert r["primary_endpoint_vs_original"]["losses"] == 10
    assert abs(r["primary_endpoint_vs_original"]["exact_two_sided_p"]
               - sign_p(8, 10)) < 1e-12
    assert abs(r["primary_trajectory_vs_random"]["exact_two_sided_p"]
               - sign_p(11, 7)) < 1e-12
    assert all(len(v) == 4 and v[0] + v[1] <= 288
               and v[2] <= 4*288 and v[3] <= 240
               for v in r["arms"].values())
    return r


def inspect_evidence(data, r):
    assert data["schema"] == "nor14.train_orthogonal_nonlinear_basin_vs_trajectory.v1"
    assert data["checkpoint_sha256"] == EXPECTED_MODEL
    assert data["probe_sha256"] == r["original_probe_sha256"]
    assert data["train_manifest"] == r["train_manifest"]
    assert data["source_cases_train"] == 96 and data["source_cases_dev"] == 36
    assert data["public_training_template_checks_shared"] == 1536
    assert set(data["all_training_heads_slots"].values()) == {400}
    assert data["repetitions"] == 4
    assert data["cases_per_family_per_repetition"] == 12
    assert data["development_global_frozen"]["kind"] == "endpoint_shuffled"
    assert {k:v["scale"] for k,v in data["development_per_arm_frozen"].items()
            } == r["dev_local_frozen_scales"]
    assert data["development_global_frozen"]["paid_checks"] == 102
    assert len(data["cohorts"]) == 4
    assert [v["public_input_manifest_sha256"] for v in data["cohorts"]
            ] == r["fresh_cohort_manifests"]
    assert data["scope"]["training_only_linear_projection"]
    assert data["scope"]["all_model_fits_use_public_demo_template_acceptance_only"]
    assert data["scope"]["frozen_original_gru_and_task_head"]
    assert not data["scope"]["endpoint_gate_additional_conditional_information_given_h"]

    reference = data["public_per_case_evidence"]["original"]
    assert len(reference) == 288
    families = Counter(z["family"] for z in reference)
    assert set(families) == {
        "signed", "sparse", "product", "rational", "squared", "late_source"}
    assert set(families.values()) == {48}
    all_digests = {v["public_features_sha256"] for v in reference}
    late = [z for z in reference if z["family"] == "late_source"]
    nonlate = [z for z in reference if z["family"] != "late_source"]
    assert len(all_digests) == 241
    assert len({z["public_features_sha256"] for z in nonlate}) == 240
    assert len({z["public_features_sha256"] for z in late}) == 1
    assert not {z["public_features_sha256"] for z in nonlate} & {
        z["public_features_sha256"] for z in late}

    for name, records in data["public_per_case_evidence"].items():
        assert name in r["arms"], name
        assert len(records) == len(reference)
        for idx, (a, b) in enumerate(zip(reference, records, strict=True)):
            assert (a["public_features_sha256"], a["family"]) == (
                b["public_features_sha256"], b["family"]), (name,idx)
            assert b["accepted"] in (0, 1)
            assert b["future_correct"] in (0, 1)
            assert b["false_accept"] in (0, 1)
            assert b["accepted"] == b["future_correct"]+b["false_accept"]
            assert 1 <= b["paid_checks"] <= 4
            assert len(b["verification_trace"]) == b["paid_checks"]
            assert b["top1"] == b["verification_trace"][0]
            assert b["verified_class"] == (
                b["verification_trace"][-1] if b["accepted"] else None)
        got = [
            sum(z["future_correct"] for z in records),
            sum(z["false_accept"] for z in records),
            sum(z["paid_checks"] for z in records),
            sum(z["future_correct"] for z in records
                if z["family"] != "late_source")
        ]
        assert got == r["arms"][name], (name, got)
        outcome = data["outcomes"][name]
        assert got[:3] == [
            outcome["future_correct"],
            outcome["false_accept"],
            outcome["paid_checks"]
        ]
        assert got[3] == outcome["nonlate"]["future_correct"]
        assert outcome["nonlate"]["cases"] == 240
        assert outcome["nonlate"]["false_accept"] == 0
        assert outcome["nonlate"]["unique_input_groups"] == 240
        for i, report in enumerate(data["cohorts"]):
            sub = records[i*72:(i+1)*72]
            assert len(sub) == 72
            digest = sha256("\n".join(
                x["public_features_sha256"] for x in sub).encode()).hexdigest()
            assert digest == report["public_input_manifest_sha256"]
            expected = report["result"][name]
            assert (sum(x["future_correct"] for x in sub),
                    sum(x["false_accept"] for x in sub),
                    sum(x["paid_checks"] for x in sub)) == (
                        expected["future_correct"],
                        expected["false_accept"],
                        expected["paid_checks"])

    for label, expected in data["primary_and_intervention_comparisons"].items():
        if label.endswith("_vs_original_nonlate"):
            arm = label.removesuffix("_vs_original_nonlate")
            left = data["public_per_case_evidence"]["original"]
            right = data["public_per_case_evidence"][arm]
        else:
            lead, other = label.split("_vs_", 1)
            lead = "endpoint_basin" if lead == "fixed_endpoint_basin" else lead
            left = data["public_per_case_evidence"][other]
            right = data["public_per_case_evidence"][lead]
        observed = paired(left, right)
        for field, value in observed.items():
            if field == "unadjusted_exact_group_sign_p":
                assert abs(value - expected[field]) < 1e-12, label
            else:
                assert value == expected[field], (label, field, value, expected[field])
        assert observed["cases"] == observed["unique_public_inputs"] == 240
    basin = paired(reference, data["public_per_case_evidence"]["endpoint_basin"])
    trajectory = paired(
        data["public_per_case_evidence"]["trajectory_random"],
        data["public_per_case_evidence"]["trajectory_basin"])
    assert (basin["paired_gains"], basin["paired_losses"]) == (8, 10)
    assert (trajectory["paired_gains"], trajectory["paired_losses"]) == (11, 7)
    for kind, model in data["training"].items():
        assert len(model["weight_sha256"]) == 64
        assert model["epochs"] == 150
        assert model["projection"]["kind"] == kind
        assert model["projection"]["linear_rank"] == 13
        assert model["projection"]["max_train_linear_moment"] < 1e-7
        assert len(model["projection"]["coefficients"]) == 12
    return {"model_arms_checked":len(data["public_per_case_evidence"]),
            "per_case_decisions_checked":sum(len(v) for v in data[
                "public_per_case_evidence"].values()),
            "paired_contrasts_checked":len(
                data["primary_and_intervention_comparisons"]),
            "unique_nonlate_inputs":240, "source_cases":288}


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--full", type=Path, help="unzipped original full result JSON")
    group.add_argument("--archive", type=Path, help="repository-preserved NOR-14 full ZIP")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    receipt = validate_receipt(json.loads(RECEIPT.read_text()))
    raw = None
    if args.archive:
        with ZipFile(args.archive) as z:
            assert z.namelist() == ["nor14_full.json"], z.namelist()
            assert z.testzip() is None
            raw = z.read("nor14_full.json")
    if args.full:
        raw = args.full.read_bytes()
    if raw is not None:
        assert len(raw) == receipt["full_json_utf8_bytes"]
        assert sha256(raw).hexdigest() == EXPECTED_SHA
        result = inspect_evidence(json.loads(raw), receipt)
        print("NOR-14 INDEPENDENT FULL RECEIPT AUDIT PASSED", result)
    else:
        print("NOR-14 FROZEN RECEIPT VALID; full input artifact required for per-case audit")
    if args.check:
        assert receipt["workflow_run"].startswith("https://github.com/")


if __name__ == "__main__":
    main()
