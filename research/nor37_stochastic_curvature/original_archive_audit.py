"""NOR-37 permanent original first-source/grade SHA lock and exact replay.

Original registered grader #44079786124 produced the first source-only
fixture and complete graded outcome. A one-shot publisher SHA-verified
both raw streams, independently replayed 30,277,632 decisions, and
archived the original bytes with immutable canonical receipt. This
read-only audit rechecks original byte identity and reconstructs every
original source, greedy calibration attack and heldout cross-play.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor37_stochastic_curvature.receipt_audit import replay

FIRST={
    "nor37_fixture.json":(4857464,
        "f969925a159aa3e36d7061e2e65e0a6b9e6e0f82634e2a7e52b4dac874df6710"),
    "nor37_full.json":(559101,
        "4852bd7de8012484db34404519553ae4d1e9f388203f2fdbf8b7516a534b684c"),
}
ZIP_SIZE=1826594
ZIP_SHA="185562f4d92ea09624d80599ee582d8bfead809928a213035593c9aed251cc82"


def audit_original_first():
    receipt=json.loads((HERE/"results/nor37_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor37.original_first_sourceheldout_stochastic_curvature.v1"
    assert receipt["first_registration_commit"]==(
        "a87401971de2cdefd7f36110fc6beae009c43bef")
    assert receipt["first_original_grader_run"].endswith("44079786124")
    assert receipt["first_original_artifact_id"]==14327326409
    assert receipt["first_original_artifact_name"]==(
        "nor37_first_original_source_and_stochastic_policyaware_grade")
    assert receipt["original_file_bytes_sha256"]=={
        k:{"bytes":n,"sha256":h} for k,(n,h) in FIRST.items()}
    assert receipt["new_independent_source_units"]==24
    assert receipt["excluded_historical_source_units"]==156
    assert receipt["auditor_and_attacker_policy_classes"]==10
    assert receipt["stochastic_sensitivity_levels"]==[25,50,75]
    assert receipt["blind_attack_control_families"]==4
    assert receipt["attack_restarts_per_family"]==8
    assert receipt["calibration_schedules_per_source"]==256
    assert receipt["heldout_schedules_per_source"]==512
    assert receipt["expected_replayed_crossplay_decisions"]==30277632
    assert receipt["recorded_direct_integer_witness_pairs"]==29568
    assert receipt["registered_primary_pass"]
    assert all(receipt["registered_five_gates"].values())
    assert all(x==24 for x in receipt["source_direction_wins"].values())
    assert abs(receipt["source_aware_exploration_tradeoff"]-
               .32645170603295337)<1e-15

    archive=HERE/"results/nor37_original_hosted.zip"
    assert archive.is_file()
    blob=archive.read_bytes()
    assert len(blob)==ZIP_SIZE
    assert sha256(blob).hexdigest()==ZIP_SHA
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(FIRST)
        assert z.testzip() is None
        contents={name:z.read(name) for name in FIRST}
    for name,(length,hash_value) in FIRST.items():
        assert len(contents[name])==length
        assert sha256(contents[name]).hexdigest()==hash_value
    fixture=json.loads(contents["nor37_fixture.json"])
    full=json.loads(contents["nor37_full.json"])
    assert fixture["schema"]=="nor37.new_public_only_reps56to59.v1"
    assert full["schema"]=="nor37.stochastic_intraatom_curvature_policyaware.v1"
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert len(fixture["cases"])==len(full["source_records"])==24
    assert full["public_fixture_sha256"]==FIRST["nor37_fixture.json"][1]
    assert full["registered_primary_pass"]
    assert full["registered_five_gates"]==receipt["registered_five_gates"]
    assert full["source_direction_wins"]==receipt["source_direction_wins"]
    assert full["source_means"]==receipt["source_means"]
    assert full["retained_ratio_stochastic_to_deterministic_top_gain"]==(
        receipt["source_aware_exploration_tradeoff"])

    validated=replay(contents["nor37_fixture.json"],contents["nor37_full.json"])
    assert validated["new_source_units_reconstructed"]==24
    assert validated["historical_source_units_excluded"]==156
    assert validated["calibrated_greedy_attack_sets_replayed"]==1920
    assert validated["fixed_blind_attack_sets_replayed"]==768
    assert validated["component_calibration_and_evaluation_plans_independently_built"]==663552
    assert validated["source_policy_attack_world_decisions"]==30277632
    assert validated["registered_primary_pass"]
    return validated


if __name__=="__main__":
    print("NOR37 ORIGINAL FIRST BYTE-PINNED SOURCE/GRADE FULL REPLAY PASSED",
          audit_original_first())
