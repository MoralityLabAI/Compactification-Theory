"""NOR-32 permanent first scientific FAIL byte audit and exact independent replay.

Preserves the FIRST preregistered source-only fixture and full result,
including the negative registered verdict. Reads the source-controlled
ZIP and receipt, pins both original SHA256s, and independently rebuilds
all 24 RRC25 programs, six new OOD support mechanisms, 3,648 attack
sets, and 11,206,656 heldout audit decisions.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor32_ood_audits.receipt_audit import audit

EXPECTED={
    "nor32_fixture.json":(3892735,"129dba8ef9b1ae64edf58f9b71d6de761c3f4ca4fe5be4224e2495e3353518b7"),
    "nor32_full.json":(1142260,"4b475c62c7289574c9ff74b67e2634f9094b9c60f4d296269588cbe591bed0e9"),
}


def original_first_run_audit():
    receipt=json.loads((HERE/"results/nor32_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor32.original_first_preregistered_ood_support_transfer.v1"
    assert receipt["first_registration_commit"]==(
        "07f0f91211c21b3a7ed540ebcb246faffc7377e0")
    assert receipt["original_artifact_id"]==14310062534
    assert receipt["first_original_completed_grader_run"].endswith("44042332472")
    assert receipt["original_file_sizes_and_sha256"]=={
        name:{"bytes":size,"sha256":digest}
        for name,(size,digest) in EXPECTED.items()}
    assert receipt["scientific_primary_pass"] is False
    assert receipt["source_ood_blind_wins"]==4
    assert receipt["worst_ood_family_wins"]==24
    assert receipt["policyaware_ood_source_wins"]==19
    assert receipt["expected_independent_audit_decisions"]==11206656
    assert receipt["exact_witness_pairs_directly_rechecked"]==87552
    assert receipt["registered_five_gates"]=={
        "blind_ood_mean_and_source_wins":False,
        "calibrated_ood_and_legacy_template_retention":True,
        "retain_60pct_of_pure_uniform_ood":True,
        "source_witness_floor_replay_controls":True,
        "worst_family_source_wins":True,
    }
    archive=HERE/"results/nor32_original_hosted.zip"
    assert archive.is_file()
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(EXPECTED)
        assert z.testzip() is None
        raw={name:z.read(name) for name in EXPECTED}
    for name,(size,digest) in EXPECTED.items():
        assert len(raw[name])==size
        assert sha256(raw[name]).hexdigest()==digest
    source=json.loads(raw["nor32_fixture.json"])
    outcome=json.loads(raw["nor32_full.json"])
    assert source["public_only"] and source["private_truth_and_program_absent"]
    assert len(source["cases"])==len(outcome["source_records"])==24
    assert outcome["registered_primary_pass"] is False
    assert outcome["registered_five_gates"]==receipt["registered_five_gates"]
    verified=audit(raw["nor32_fixture.json"],raw["nor32_full.json"])
    assert verified["genuinely_new_source_units"]==24
    assert verified["independent_audit_world_decisions"]==11206656
    assert verified["original_exact_witness_pairs_crosschecked"]==87552
    assert verified["registered_primary_pass"] is False
    return verified


if __name__=="__main__":
    print("NOR32 ORIGINAL PREREGISTERED FAIL SHA-AUDITED AND INDEPENDENTLY REPLAYED",
          original_first_run_audit())
