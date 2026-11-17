"""NOR-28 read-only SHA auditor for the first hosted precommit/OS-entropy run.

Uses only the standard library. It deliberately never imports experiment,
sampler, grade or the first-run independent replayer. The permanent
source-controlled ZIP contains exact unaltered bytes of FIVE original files
from the first *initiated* prospective workflow, Actions #44019242345.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import zipfile
import math

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/"results"/"nor28_original_hosted.zip"
RECEIPT=HERE/"results"/"nor28_first_original_receipt.json"


def check():
    r=json.loads(RECEIPT.read_bytes())
    assert r["schema"]=="nor28.first_immutable_os_entropy_commitment_receipt.v1"
    assert r["registration_first_commit"]=="da296c8d0cdfd8f42066896dca35bb16b7502f4d"
    assert r["first_started_actions_run"]==44019242345
    assert r["first_prep_artifact_id"]==14300925012
    assert r["first_challenge_artifact_id"]==14300611109
    assert ARCHIVE.is_file(),"NOR28 original first-run evidence archive is not committed"
    blobs={}
    with zipfile.ZipFile(ARCHIVE) as zipped:
        assert set(zipped.namelist())=={r["raw"][k]["name"] for k in
                                      ("fixture","sealed","manifest","challenges","full")}
        for key,info in r["raw"].items():
            raw=zipped.read(info["name"])
            assert len(raw)==info["bytes"],(key,"size")
            assert sha256(raw).hexdigest()==info["sha256"],(key,"sha256")
            blobs[key]=json.loads(raw)
    fixture,sealed,manifest,challenges,full=[
        blobs[k] for k in ("fixture","sealed","manifest","challenges","full")]
    assert fixture["schema"]=="nor28.public_only_12_sources.v1"
    assert fixture["source_only"] and fixture["private_truth_absent"]
    assert fixture["attacks_absent"] and fixture["challenge_entropy_absent"]
    assert len(fixture["cases"])==12
    assert sealed["schema"]=="nor28.attack_openings.v1"
    assert len(sealed["openings"])==96
    assert manifest["schema"]=="nor28.public_commitments.v1"
    assert len(manifest["commitments"])==96
    assert manifest["sealed_sha256"]==r["raw"]["sealed"]["sha256"]
    assert manifest["fixture_sha256"]==r["raw"]["fixture"]["sha256"]
    assert challenges["schema"]=="nor28.fresh_os_entropy_512_audits_per_source.v1"
    assert challenges["root_origin"]=="os_secrets_token_bytes_32"
    root=bytes.fromhex(challenges["root_hex_disclosed_after_commit"])
    assert len(root)==32
    assert sha256(b"NOR28-root-v1\x00"+root).hexdigest()==\
           r["original_root_commitment_sha256"]
    assert len(challenges["records"])==12
    assert challenges["manifest_sha256"]==r["raw"]["manifest"]["sha256"]
    assert full["schema"]=="nor28.commit_before_entropy_exact_audit.v1"
    assert full["fixture_sha256"]==r["raw"]["fixture"]["sha256"]
    assert full["manifest_sha256"]==r["raw"]["manifest"]["sha256"]
    assert full["sealed_attack_sha256"]==r["raw"]["sealed"]["sha256"]
    assert full["original_challenge_sha256"]==r["raw"]["challenges"]["sha256"]
    assert full["os_root_commitment"]==r["original_root_commitment_sha256"]
    assert full["registered_primary_pass"]==r["prospective_result"]["registered_primary_pass"]
    assert full["independent_decisions"]==147456
    assert full["attack_commitment_count"]==96
    assert full["postcommit_leakage_decision_changes"]==0
    assert full["nonzero_shift_disagreements"]==0
    assert full["forged_prior_commitment_accepted"]==0
    assert full["fully_informed_after_schedule_detected"]==0
    assert full["honest_target_rejections"]==0
    assert full["commitment_openings_accepted"]==96
    assert full["deliberate_mutations_rejected"]==96
    assert full["primary_hybrid_vs_static_source_wins"]==10
    assert len(full["source_records"])==12
    for key,stored in (
        ("primary_hybrid_calibrated_source_mean","hybrid_calibrated_detection_source_mean"),
        ("static_calibrated_source_mean","static_calibrated_detection_source_mean"),
        ("uniform_calibrated_source_mean","uniform_calibrated_detection_source_mean")):
        assert math.isclose(full[key],r["prospective_result"][stored],abs_tol=1e-15)
    print("NOR28 IMMUTABLE ORIGINAL FIVE-FILE RAW BYTE AUDIT PASS")
    for key,info in r["raw"].items():
        print("NOR28 ORIGINAL",key,info["bytes"],info["sha256"])


if __name__=="__main__":check()
