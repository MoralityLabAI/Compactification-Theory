"""NOR-31 immutable original first-run evidence auditor (no NOR-31 imports).

The binary archive contains original untouched first-created bytes of
all eight PREP/CHALLENGE/LEAK/OUTCOME files from Actions #44032053843.
The SHA receipt was committed after first complete independent replay.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import math
import zipfile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor31_first_original_receipt.json"
ARCHIVE=HERE/"results"/"nor31_original_hosted.zip"


def check():
    rec=json.loads(RECEIPT.read_bytes())
    assert rec["schema"]=="nor31.first_original_24_source_matched_root_receipt.v1"
    assert rec["registration_commit"]=="dd0e9f60cdab90ed2b2fb482dedc7496f7f50870"
    assert rec["first_initiated_actions_run"]==44032053843
    assert rec["independent_replay_job"]==133867148826
    assert rec["original_artifacts"]=={
        "precommitment":14304994094,"evaluation":14305008802,
        "leaked_and_outcome":14305103339}
    assert ARCHIVE.is_file(),"NOR31 first original evidence ZIP not yet committed"
    originals={}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert set(z.namelist())=={
            x["name"] for x in rec["original_raw"].values()}
        for key,info in rec["original_raw"].items():
            raw=z.read(info["name"])
            assert len(raw)==info["bytes"],(key,"raw byte count")
            assert sha256(raw).hexdigest()==info["sha256"],(key,"raw hash")
            originals[key]=json.loads(raw)
    source=originals["fixture"]
    cal=originals["calibration"]
    pre=originals["pre_sealed"]
    pre_manifest=originals["pre_manifest"]
    eval=originals["evaluation"]
    late=originals["late_sealed"]
    late_manifest=originals["late_manifest"]
    full=originals["outcome"]
    assert source["schema"]=="nor31.public_only_24_fresh_sources.v1"
    assert source["source_only"] and source["hidden_truth_absent"]
    assert source["attack_sets_absent"] and source["challenge_entropy_absent"]
    assert len(source["cases"])==24
    assert len({r["full_public_input_sha256"] for r in source["cases"]})==24
    assert cal["phase"]=="calibration" and eval["phase"]=="evaluation"
    assert cal["root_origin"]==eval["root_origin"]=="os_secrets_token_bytes_32"
    assert cal["root_hex_disclosed_after_freeze"]!=eval["root_hex_disclosed_after_freeze"]
    assert cal["root_commitment_sha256"]==rec["roots"]["calibration_sha256"]
    assert eval["root_commitment_sha256"]==rec["roots"]["evaluation_sha256"]
    assert len(cal["records"])==len(eval["records"])==24
    assert pre_manifest["sealed_sha256"]==rec["original_raw"]["pre_sealed"]["sha256"]
    assert pre_manifest["calibration_sha256"]==rec["original_raw"]["calibration"]["sha256"]
    assert eval["precommit_manifest_sha256"]==rec["original_raw"]["pre_manifest"]["sha256"]
    assert eval["calibration_sha256"]==rec["original_raw"]["calibration"]["sha256"]
    assert late_manifest["sealed_sha256"]==rec["original_raw"]["late_sealed"]["sha256"]
    assert late_manifest["precommit_manifest_sha256"]==rec["original_raw"]["pre_manifest"]["sha256"]
    assert len(pre["openings"])==len(late["openings"])==2304
    assert full["schema"]=="nor31.prospective_independent_calibration_vs_leaked_eval_root.v1"
    for report,receipt_key in (
        ("fixture_sha256","fixture"),("calibration_sha256","calibration"),
        ("pre_root_sealed_sha256","pre_sealed"),("pre_root_manifest_sha256","pre_manifest"),
        ("evaluation_sha256","evaluation"),("late_sealed_sha256","late_sealed"),
        ("late_manifest_sha256","late_manifest")):
        assert full[report]==rec["original_raw"][receipt_key]["sha256"]
    assert full["fixed_batch_decisions"]==rec["first_scientific_result"]["fixed_batch_decisions"]==2359296
    assert full["registered_primary_pass"]==rec["first_scientific_result"]["registered_primary_pass"]
    assert full["primary_positive_source_pairs"]==24
    assert full["original_early_commitments"]==2304
    assert full["post_eval_root_commitments"]==2304
    assert full["late_changed_attack_supports"]==rec["first_scientific_result"]["changed_supports_between_conditions"]
    assert full["per_schedule_informed_tests"]==36864
    assert full["per_schedule_informed_detections"]==0
    assert full["delta_disagreements"]==0
    assert full["legitimate_target_rejections"]==0
    assert full["late_or_informed_accepted_old_commitments"]==0
    assert full["root_phase_independence_checked"]
    assert full["original_attacks_unchanged_by_eval_root_disclosure"]
    assert full["source_count"]==len(full["source_records"])==24
    assert math.isclose(full["primary_paired_hybrid_mean_decrease"],
                        rec["first_scientific_result"]["primary_paired_mean_reduction"],abs_tol=1e-15)
    for arm,rates in rec["first_scientific_result"]["equal_source_rates"].items():
        for condition,val in rates.items():
            assert math.isclose(full["equal_source_policy_means"][arm][condition],
                                val,abs_tol=1e-15)
    assert all(full["registered_gates"].values())
    print("NOR31 ORIGINAL 8-FILE BYTE SHA AUDIT PASS; 24 SOURCES; 2,359,296 DECISIONS")
    for key,info in rec["original_raw"].items():
        print("NOR31 ORIGINAL",key,info["bytes"],info["sha256"])


if __name__=="__main__":
    check()
