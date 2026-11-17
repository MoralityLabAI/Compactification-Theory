"""NOR-32 read-only, first hosted eight-file original SHA-256 archive audit.

No imports from NOR32 source/optimizer/grader/replay. This script checks
unchanged raw bytes and the frozen original scientific decision. The
source-controlled ZIP is published only after the first independent
23,592,960-decision replay completed successfully.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json,zipfile,math

HERE=Path(__file__).resolve().parent
REC=HERE/"results"/"nor32_first_original_receipt.json"
ARCHIVE=HERE/"results"/"nor32_original_hosted.zip"


def check():
    r=json.loads(REC.read_bytes())
    assert r["schema"]=="nor32.first_original_nested_batch_receipt.v1"
    assert r["registration_first_commit"]=="44d8c506de2b2079c167cb0ce5f3a2eb620028b5"
    assert r["first_initiated_actions_run"]==44043154269
    assert r["first_independent_replay_job"]==133904863094
    assert ARCHIVE.is_file(),"Original NOR32 first run archive has not been committed"
    objs={}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert set(z.namelist())=={v["file"] for v in r["original_raw"].values()}
        for k,v in r["original_raw"].items():
            raw=z.read(v["file"])
            assert len(raw)==v["bytes"],(k,len(raw))
            assert sha256(raw).hexdigest()==v["sha256"],(k,"sha256")
            objs[k]=json.loads(raw)
    f,cal,pm,ev,lm,o=(objs[k] for k in
        ("fixture","calibration","pre_manifest","evaluation","late_manifest","outcome"))
    assert f["schema"]=="nor32.public_only_24_fresh_sources.v1"
    assert f["source_offset"]==160032000 and f["replicates"]==[26,27,28,29]
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["attack_sets_absent"] and f["challenge_entropy_absent"]
    assert len(f["cases"])==24
    assert len({x["full_public_input_sha256"] for x in f["cases"]})==24
    assert cal["phase"]=="calibration" and ev["phase"]=="evaluation"
    assert cal["root_origin"]==ev["root_origin"]=="os_secrets_token_bytes_32"
    assert cal["root_hex_disclosed_after_freeze"]!=ev["root_hex_disclosed_after_freeze"]
    assert cal["root_commitment_sha256"]==r["os_roots"]["calibration_commitment"]
    assert ev["root_commitment_sha256"]==r["os_roots"]["evaluation_commitment"]
    assert cal["source_count"]==ev["source_count"]==24
    assert cal["trial_count"]==ev["trial_count"]==2048
    assert len(cal["records"])==len(ev["records"])==24
    assert pm["schema"]=="nor32.pre_root_9216_commitments.v1"
    assert lm["schema"]=="nor32.leaked_root_9216_commitments.v1"
    assert pm["attack_count"]==lm["attack_count"]==9216
    assert pm["sealed_sha256"]==r["original_raw"]["pre_sealed"]["sha256"]
    assert lm["sealed_sha256"]==r["original_raw"]["late_sealed"]["sha256"]
    assert ev["precommit_manifest_sha256"]==r["original_raw"]["pre_manifest"]["sha256"]
    assert ev["calibration_sha256"]==r["original_raw"]["calibration"]["sha256"]
    assert lm["precommit_manifest_sha256"]==r["original_raw"]["pre_manifest"]["sha256"]
    assert lm["calibration_sha256"]==r["original_raw"]["calibration"]["sha256"]
    assert o["schema"]=="nor32.nested_depth_attack_coverage_and_detection.v1"
    for file_key,field in (
        ("fixture","fixture_sha256"),("calibration","calibration_sha256"),
        ("pre_sealed","pre_sealed_sha256"),("pre_manifest","pre_manifest_sha256"),
        ("evaluation","evaluation_sha256"),("late_sealed","late_sealed_sha256"),
        ("late_manifest","late_manifest_sha256")):
        assert o[field]==r["original_raw"][file_key]["sha256"]
    assert o["calibration_root_commitment"]==r["os_roots"]["calibration_commitment"]
    assert o["evaluation_root_commitment"]==r["os_roots"]["evaluation_commitment"]
    assert o["registered_primary_pass"]==r["first_result"]["registered_primary_pass"]
    assert len(o["source_records"])==24
    assert o["fixed_batch_decisions"]==r["first_result"]["fixed_batch_decisions"]==23592960
    assert o["early_commitment_count"]==o["late_commitment_count"]==9216
    assert o["late_changed_attack_support_count"]==6144
    assert o["informed_per_trial_controls"]==147456
    assert o["informed_per_trial_detected"]==0
    assert o["delta_disagreements"]==o["honest_target_rejections"]==0
    assert o["forged_prior_commitments_accepted"]==0
    assert o["monotone_nested_coverage"]
    assert o["long_hybrid_sources_with_four_unseen_at_512"]==12
    assert o["long_hybrid_sources_with_four_unseen_at_2048"]==0
    assert o["hybrid2048_positive_source_pairs"]==24
    assert math.isclose(o["hybrid2048_source_mean_detection_gain"],
                        r["first_result"]["hybrid2048_equal_source_detection_difference"],
                        abs_tol=1e-15)
    for depth,mean in r["first_result"]["hybrid_detection"].items():
        for label,v in mean.items():
            assert math.isclose(o["equal_source_detection"]["hybrid_50"][depth][label],
                                v,abs_tol=1e-15)
    assert all(o["registered_gates"].values())
    print("NOR32 IMMUTABLE FIRST ORIGINAL SHA-AUDIT PASS, 24 SOURCES AND 23,592,960 DECISIONS")
    for k,v in r["original_raw"].items():
        print("NOR32 ORIGINAL",k,v["bytes"],v["sha256"])


if __name__=="__main__":
    check()
