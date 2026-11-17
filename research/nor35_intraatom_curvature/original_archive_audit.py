"""Permanent NOR-35 first valid source/grade SHA and independent full replay.

The earlier pre-result snapshot #44064001927 had a bottom-curvature
tie ordering that did not implement preregistered ascending-index tie
breaking and lacked the independent replayer. The original first VALID
completed run #44064368383 contains the corrected implementation, full
public source and grade bytes, and an independent 18,874,368-decision
source-locked replayer. Preserve that one exact original artifact.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor35_intraatom_curvature.receipt_audit import replay

ORIGINAL={
    "nor35_fixture.json":(4539616,
        "7a517c8a9f87b15e12362da2dd99d188933a018a233fe2d043ea4aca9c4b02ef"),
    "nor35_full.json":(706150,
        "194a0709ed5672dbb8b6706b4d40d6f8f3a28e28164dc3275005d8421ccd05a6"),
}


def audit_first_valid_original():
    receipt=json.loads((HERE/"results/nor35_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor35.original_corrected_sourceheldout_curvature.v1"
    assert receipt["registration_first_commit"]==(
        "3773c1b1522f419e54f34252b0c743018105bc79")
    assert receipt["first_valid_complete_replay_run"].endswith("44064368383")
    assert receipt["original_artifact_id"]==14319982047
    assert receipt["original_file_bytes_and_sha256"]=={
        name:{"bytes":size,"sha256":digest}
        for name,(size,digest) in ORIGINAL.items()}
    assert receipt["fresh_public_source_count"]==24
    assert receipt["historical_sources_excluded"]==108
    assert receipt["fixed_blind_attack_sets_replayed"]==1536
    assert receipt["source_arm_transformed_challenge_plans_replayed"]==294912
    assert receipt["original_audit_world_decisions_replayed"]==18874368
    assert receipt["direct_true_integer_witness_checks"]==2304
    assert receipt["frozen_mix_weights"]==[2,0,1,1,4,8]
    assert receipt["registered_primary_pass"]
    assert receipt["independent_source_atom_attack_and_fraction_replay_passed"]
    assert all(receipt["original_registered_five_gates"].values())
    assert receipt["source_primary_wins"]=={"B":24,"C":24,"I":24,"M":24}
    assert receipt["pre_grade_implementation_correction"]
    archive=HERE/"results/nor35_original_hosted.zip"
    assert archive.exists()
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(ORIGINAL)
        assert z.testzip() is None
        original={name:z.read(name) for name in ORIGINAL}
    for name,(size,h) in ORIGINAL.items():
        assert len(original[name])==size
        assert sha256(original[name]).hexdigest()==h
    full=json.loads(original["nor35_full.json"])
    assert full["registered_primary_pass"]
    assert full["source_wins"]==receipt["source_primary_wins"]
    assert full["source_mean_primary"]==receipt["source_primary_means"]
    assert full["registered_five_gates"]==receipt["original_registered_five_gates"]
    check=replay(original["nor35_fixture.json"],original["nor35_full.json"])
    assert check["source_units"]==24
    assert check["historic_source_units_excluded"]==108
    assert check["original_native_programs_replayed"]==24
    assert check["paired_attack_sets_replayed"]==1536
    assert check["all_source_arm_transformed_audit_plans_reconstructed"]==294912
    assert check["source_arm_attack_world_decisions_replayed"]==18874368
    assert check["direct_integer_witness_comparisons"]==2304
    assert check["registered_primary_pass"]
    return check


if __name__=="__main__":
    print("NOR35 FIRST VALID ORIGINAL SHA-LOCKED COMPLETE REPLAY PASSED",
          audit_first_valid_original())
