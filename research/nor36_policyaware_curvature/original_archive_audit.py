"""Permanent SHA-locked NOR-36 first original source/grade evidence audit.

The first registered grader #44069984005 created original source-only
fixture and complete result. Its first independent job used the wrong
field name for the fixture's source_only flag. The independent reader
was corrected without changing the original bytes; successful replay
#44071008844 and original artifact archive #44071915589 verified the
same pinned first result. This read-only checker verifies exact SHA and
replays every original world, attack and source from the permanent ZIP.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor36_policyaware_curvature.receipt_audit import replay

ORIGINAL={
    "nor36_fixture.json":(4224502,
        "881bf776e8e77eff6b8a5406789d07a4ced2eed9394a62b27d02d835c99355a4"),
    "nor36_full.json":(341232,
        "ffc9ebaf5b8e16c93e01c2c71cc387e4c8a455505f5af8cb3c9cad42ea9a44b5")
}
ORIGINAL_ZIP_SHA="8a17aea5bd358969599125a61b6c31596287d101df2fe2cffb43ec286f23593d"


def audit_original_first_run():
    receipt=json.loads((HERE/"results/nor36_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor36.original_first_registered_policyaware_curvature.v1"
    assert receipt["first_registration_commit"]==(
        "ede05db28b814d5000ac855fecb3b762876c6f9e")
    assert receipt["first_completed_grader_run"].endswith("44069984005")
    assert receipt["first_original_artifact_id"]==14322985791
    assert receipt["first_original_artifact_name"]==(
        "nor36_original_frozen_public_sources_and_full_policyaware_outcome")
    assert receipt["original_source_and_grade_bytes_sha256"]=={
        name:{"bytes":n,"sha256":h} for name,(n,h) in ORIGINAL.items()}
    assert receipt["original_source_units"]==24
    assert receipt["historical_source_units_excluded"]==132
    assert receipt["frozen_mixture_weights"]==[2,0,1,1,4,8]
    assert receipt["attacker_classes"]==6 and receipt["greedy_restarts_per_class"]==8
    assert receipt["blind_control_families"]==4 and receipt["blind_restarts_per_family"]==8
    assert receipt["calibration_schedule_count"]==256
    assert receipt["evaluation_schedule_count"]==512
    assert receipt["independent_policy_world_decisions"]==13762560
    assert receipt["registered_scientific_primary_pass"]
    assert all(receipt["original_registered_five_gates"].values())
    assert receipt["registered_source_direction_wins"]=={
        "self_curv_weaker":24,
        "fixed_curv_attacks_expose_weakness":24,
        "top_h50_gain":24,
        "mix_self_curv_weaker":24,
        "top_mix_gain":24
    }
    archive=HERE/"results/nor36_original_hosted.zip"
    assert archive.is_file()
    blob=archive.read_bytes()
    assert len(blob)==1472991
    assert sha256(blob).hexdigest()==ORIGINAL_ZIP_SHA
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(ORIGINAL)
        assert z.testzip() is None
        original={name:z.read(name) for name in ORIGINAL}
    for name,(length,h) in ORIGINAL.items():
        assert len(original[name])==length
        assert sha256(original[name]).hexdigest()==h
    f=json.loads(original["nor36_fixture.json"])
    o=json.loads(original["nor36_full.json"])
    assert f["source_only"] and f["planted_program_and_private_target_absent"]
    assert f["attacks_and_challenge_schedules_absent"]
    assert len(f["cases"])==len(o["source_records"])==24
    assert o["public_only_fixture_sha256"]==ORIGINAL["nor36_fixture.json"][1]
    assert o["registered_primary_pass"]
    assert o["registered_five_gates"]==receipt["original_registered_five_gates"]
    assert o["source_direction_wins"]==receipt["registered_source_direction_wins"]
    assert o["source_equal_mean_primary"]==receipt["registered_equal_source_mean_differences"]
    result=replay(original["nor36_fixture.json"],original["nor36_full.json"])
    assert result["new_public_source_units"]==24
    assert result["excluded_historical_source_units"]==132
    assert result["replayed_greedy_attacks"]==1152
    assert result["replayed_blind_attack_sets"]==768
    assert result["independent_crossplay_policy_world_decisions"]==13762560
    assert result["registered_primary_pass"]
    return result


if __name__=="__main__":
    print("NOR36 FIRST ORIGINAL SOURCE/GRADE SHA-LOCKED FULL REPLAY PASSED",
          audit_original_first_run())
