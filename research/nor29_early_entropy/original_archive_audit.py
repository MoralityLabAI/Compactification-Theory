"""NOR-29 independent permanent first-run raw-byte archive audit.

Read-only; imports NO NOR-29 applied module and never regenerates the first
root, commitments or result. Checks every original member SHA256 and the
cross-stage source, commit, schedule and result hashes.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import math
import zipfile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor29_first_original_receipt.json"
ARCHIVE=HERE/"results"/"nor29_original_hosted.zip"


def canon(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def digest_opening(op):
    return sha256(b"NOR29-attack-commit-v1\x00"+canon(op)).hexdigest()


def check():
    ref=json.loads(RECEIPT.read_bytes())
    assert ref["schema"]=="nor29.first_original_precommit_root_leakage_receipt.v1"
    assert ref["registration_first_commit"]=="2447248fa8c192991da695ed06af2d17d642dc5f"
    assert ref["first_initiated_actions_run_id"]==44025451594
    assert ARCHIVE.is_file(),"NOR29 original first-run ZIP not yet committed"
    obj={}
    with zipfile.ZipFile(ARCHIVE) as files:
        assert set(files.namelist())=={f["name"] for f in ref["files"].values()}
        for key,meta in ref["files"].items():
            raw=files.read(meta["name"])
            assert len(raw)==meta["bytes"],(key,"size",len(raw))
            assert sha256(raw).hexdigest()==meta["sha256"],(key,"SHA256")
            obj[key]=json.loads(raw)
    f,early,em,c,late,lm,out=(
        obj[key] for key in (
            "fixture","early_sealed","early_manifest","challenge",
            "late_sealed","late_manifest","outcome"))
    assert f["schema"]=="nor29.public_only_24_fresh_sources.v1"
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["attack_sets_absent"] and f["challenge_entropy_absent"]
    assert f["source_offset"]==139329000
    assert f["replicates"]==[12,13,14,15]
    assert len(f["cases"])==24
    assert len({a["full_public_input_sha256"] for a in f["cases"]})==24
    assert early["schema"]=="nor29.original_768_salted_commitments.v1"
    assert em["schema"]=="nor29.public_768_precommitments.v1"
    assert em["fixture_sha256"]==ref["files"]["fixture"]["sha256"]
    assert early["fixture_sha256"]==ref["files"]["fixture"]["sha256"]
    assert em["sealed_sha256"]==ref["files"]["early_sealed"]["sha256"]
    assert len(em["commitments"])==len(early["openings"])==768
    old_refs={(a["source_sha256"],a["replica"]):a["commitment_sha256"]
              for a in em["commitments"]}
    assert len(old_refs)==768
    for o in early["openings"]:
        assert o["attack_class"]=="blind_uniform"
        assert digest_opening(o)==old_refs[(o["source_sha256"],o["replica"])]
    assert c["schema"]=="nor29.single_fresh_os_root_24x3x512_schedules.v1"
    assert c["fixture_sha256"]==ref["files"]["fixture"]["sha256"]
    assert c["early_manifest_sha256"]==ref["files"]["early_manifest"]["sha256"]
    assert c["root_origin"]=="os_secrets_token_bytes_32"
    root=bytes.fromhex(c["root_hex_disclosed_after_freeze"])
    assert len(root)==32
    assert sha256(b"NOR29-root-v1\x00"+root).hexdigest()==\
           ref["original_sha256_root_commitment"]
    assert c["root_commitment_sha256"]==ref["original_sha256_root_commitment"]
    assert len(c["records"])==24 and c["trial_count"]==512
    assert all(len(p)==512 for z in c["records"] for p in z["schedules"].values())
    assert late["schema"]=="nor29.early_root_leaked_2304_attack_openings.v1"
    assert lm["schema"]=="nor29.early_root_leaked_2304_new_commitments.v1"
    assert lm["challenge_sha256"]==late["challenge_sha256"]==\
           ref["files"]["challenge"]["sha256"]
    assert lm["early_manifest_sha256"]==late["early_manifest_sha256"]==\
           ref["files"]["early_manifest"]["sha256"]
    assert lm["sealed_sha256"]==ref["files"]["late_sealed"]["sha256"]
    assert lm["root_disclosed_to_attacker_before_new_commitments"] is True
    assert len(lm["commitments"])==len(late["openings"])==2304
    new_refs={(a["source_sha256"],a["auditor"],a["replica"]):
              a["commitment_sha256"] for a in lm["commitments"]}
    assert len(new_refs)==2304
    for o in late["openings"]:
        key=(o["source_sha256"],o["auditor"],o["replica"])
        assert digest_opening(o)==new_refs[key]
        blind=next(x for x in early["openings"] if
                   x["source_sha256"]==key[0] and x["replica"]==key[2])
        forged=dict(blind,changed_indices=o["changed_indices"],
                    nonce_hex=o["nonce_hex"])
        assert digest_opening(forged)!=old_refs[(key[0],key[2])]
    assert out["schema"]=="nor29.batch_fixed_early_root_leakage_v1"
    expected={
        "fixture_sha256":"fixture",
        "original_early_manifest_sha256":"early_manifest",
        "original_sealed_early_sha256":"early_sealed",
        "original_challenge_sha256":"challenge",
        "late_attack_manifest_sha256":"late_manifest",
        "late_sealed_attack_sha256":"late_sealed",
    }
    for field,source in expected.items():
        assert out[field]==ref["files"][source]["sha256"]
    assert out["root_commitment_sha256"]==ref["original_sha256_root_commitment"]
    assert out["os_root_origin"]==c["root_origin"]
    assert out["registered_primary_pass"]==ref["primary"]["registered_pass"]
    assert out["source_count"]==24
    assert len(out["source_records"])==24
    assert out["original_early_attack_commitments"]==768
    assert out["postroot_new_attack_commitments"]==2304
    assert out["fixed_batch_decisions"]==2359296
    assert out["fully_informed_per_schedule_decisions"]==36864
    assert out["fully_informed_per_schedule_detected"]==0
    assert out["late_attack_opens_old_commitment"]==0
    assert out["shift_magnitude_disagreements"]==0
    assert out["source_hybrid_blind_minus_late_wins"]==ref["primary"]["source_paired_hybrid_blind_minus_late_positive"]
    assert math.isclose(out["source_hybrid_blind_minus_late_mean"],
                        ref["primary"]["source_mean_hybrid_blind_minus_late"],abs_tol=1e-15)
    for arm,part in {
        "hybrid_50":("hybrid_blind_mean","hybrid_early_root_aware_mean"),
        "uniform16":("uniform16_blind_mean","uniform16_early_root_aware_mean"),
        "static_prior":("static_prior_blind_mean","static_prior_early_root_aware_mean"),
    }.items():
        for regime,key in zip(("blind_uniform","early_root_aware"),part):
            assert math.isclose(out["equal_source_policy_means"][arm][regime],
                                ref["primary"][key],abs_tol=1e-15)
    assert all(out["registered_gates"].values())
    print("NOR29 PERMANENT FIRST-RUN ORIGINAL SEVEN-FILE RAW BYTE AUDIT PASS")
    for key,meta in ref["files"].items():
        print("NOR29 FIRST",key,meta["bytes"],meta["sha256"])


if __name__=="__main__":check()
