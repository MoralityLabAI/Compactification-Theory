"""NOR-33 immutable first source/grade SHA-256 and independent replay gate.

Keeps first prospective 24-source matched-overlap PASS as the immutable
evidence object, including all source-level data, 6720 attack sets,
20,643,840 comparisons, and 20,160 direct exact witness pairs.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor33_overlap_matched_transfer.receipt_audit import audit

EXPECTED={
    "nor33_fixture.json":(4547200,
        "16f3b397aea17adbd1462666142a13e933c8da67a1286e1e6418d32b22090db8"),
    "nor33_full.json":(1761233,
        "88d37a957f7d3be42b76213b351df921a08556654a90a2ed21df1b66788d6a62")
}


def original_first_run_audit():
    r=json.loads((HERE/"results/nor33_first_frozen_receipt.json").read_text())
    assert r["schema"]=="nor33.original_first_source_locked_overlap_matched.v1"
    assert r["preregistered_first_commit"]==(
        "9e14d1132744272c6c674e3799cea47cdb70b031")
    assert r["first_completed_all_green_run"].endswith("44046554983")
    assert r["first_original_artifact_id"]==14311872570
    assert r["original_files"]=={
        f:{"bytes":size,"sha256":digest} for f,(size,digest) in EXPECTED.items()}
    assert r["source_public_units"]==24 and r["previous_source_units_excluded"]==60
    assert r["matched_support_cells"]==840 and r["attack_sets"]==6720
    assert r["auditor_world_decisions_independently_replayed"]==20643840
    assert r["exact_witness_pair_checks"]==20160
    assert r["registered_primary_pass"]
    assert r["independent_source_attack_schedule_and_count_replay_passed"]
    assert all(r["registered_five_gates"].values())
    assert r["source_wins"]=={
        "gradient":24,"low":24,"high":24,"curvature_residual":24}
    zip_path=HERE/"results/nor33_original_hosted.zip"
    assert zip_path.exists()
    with ZipFile(zip_path) as z:
        assert sorted(z.namelist())==sorted(EXPECTED)
        assert z.testzip() is None
        raw={name:z.read(name) for name in EXPECTED}
    for name,(n,h) in EXPECTED.items():
        assert len(raw[name])==n
        assert sha256(raw[name]).hexdigest()==h
    original=json.loads(raw["nor33_full.json"])
    assert original["registered_primary_pass"]
    assert original["registered_five_gates"]==r["registered_five_gates"]
    assert original["source_wins"]==r["source_wins"]
    assert original["equal_source_means"]==r["equal_source_means"]
    report=audit(raw["nor33_fixture.json"],raw["nor33_full.json"])
    assert report["fresh_source_count"]==24
    assert report["historical_sources_excluded"]==60
    assert report["matched_support_cells_replayed"]==840
    assert report["four_coordinate_attack_sets_replayed"]==6720
    assert report["auditor_world_decisions_replayed"]==20643840
    assert report["registered_primary_pass"]
    return report


if __name__=="__main__":
    print("NOR33 ORIGINAL SHA-LOCKED FIRST SOURCE/RESULT REPLAY PASSED",
          original_first_run_audit())
