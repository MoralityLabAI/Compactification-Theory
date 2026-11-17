"""NOR-40 permanent first-original-source/grade byte lock.

Run after the dedicated publisher has committed its deterministic ZIP and
receipt. The ordinary check is fast and read-only. --full runs the expensive
independently implemented source, attack and heldout cross-play replayer.

Never regenerate a fresh source or rescore a frozen primary gate in this file.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"
FIRST_REGISTRATION="dbc1adbe0e30b9c0e36421cf07cebcf376d6a1ec"
FIRST_RUN_ID=44110593049
FIRST_ARTIFACT_ID=14343000877
FIRST={
    "nor40_fixture.json":(
        9387973,
        "b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5"),
    "nor40_full.json":(
        3492789,
        "42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092"),
}
FIRST_ZIP_BYTES=3624637
FIRST_ZIP_SHA256="1ef743e51066c902c82beb96be5b3d6b99bb501f65b8298749a5808649000a9c"


def audit(full=False):
    manifest=json.loads((RESULTS/"nor40_first_frozen_receipt.json").read_bytes())
    assert manifest["schema"]=="nor40.first_original_strong_adversary_archive.v1"
    assert manifest["first_registration_commit"]==FIRST_REGISTRATION
    assert manifest["first_original_grade_run"]==FIRST_RUN_ID
    assert manifest["first_original_artifact_id"]==FIRST_ARTIFACT_ID
    assert manifest["source_only_first_fixture_frozen_before_grade"]
    assert manifest["original_file_bytes_sha256"]=={
        n:{"bytes":size,"sha256":h} for n,(size,h) in FIRST.items()}
    assert manifest["new_source_units"]==48
    assert manifest["historical_source_units_excluded"]==276
    archive=RESULTS/"nor40_original_first_hosted.zip"
    buf=archive.read_bytes()
    assert len(buf)==manifest["archive_zip_bytes"]==FIRST_ZIP_BYTES
    assert sha256(buf).hexdigest()==manifest["archive_zip_sha256"]==FIRST_ZIP_SHA256
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(FIRST)
        assert z.testzip() is None
        contents={name:z.read(name) for name in FIRST}
    for name,(size,digest) in FIRST.items():
        assert len(contents[name])==size
        assert sha256(contents[name]).hexdigest()==digest
    source=json.loads(contents["nor40_fixture.json"])
    graded=json.loads(contents["nor40_full.json"])
    assert source["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    assert graded["schema"]=="nor40.strong_calibration_policyaware.v1"
    assert source["registered_first_commit"]==graded["registration_first_commit"]==FIRST_REGISTRATION
    assert source["public_only"] and source["planted_and_private_target_absent"]
    assert source["attack_witness_and_audit_absent"]
    assert source["excluded_historical_sources"]==graded["historical_sources_excluded"]==276
    assert source["replicates"]==list(range(76,84))
    assert len(source["cases"])==len(graded["source_records"])==48
    assert graded["public_fixture_sha256"]==FIRST["nor40_fixture.json"][1]
    assert graded["registered_five_gates"]==manifest["original_five_gates"]
    assert graded["registered_primary_pass"]==manifest["original_science_pass"]
    assert graded["source_means"]==manifest["source_means"]
    assert graded["source_direction_wins"]==manifest["source_direction_wins"]
    assert graded["registered_primary_pass"]==all(graded["registered_five_gates"].values())
    assert len(graded["registered_five_gates"])==5
    assert graded["calibration_schedules_per_source"]==2048
    assert graded["heldout_schedules_per_source"]==512
    assert graded["restricted_pool_size"]==16
    assert graded["restricted_sets_per_policy"]==1820
    assert graded["attack_sets_per_policy"]==8
    assert all(row["split"]=="TEST" and
               not set(row)&{"program","future","target","attack","witness","audit"}
               for row in source["cases"])
    ids=[r["full_public_input_sha256"] for r in source["cases"]]
    assert len(ids)==len(set(ids))==48
    result={
        "schema":"nor40.first_immutable_original_evidence_audit.v1",
        "registration":FIRST_REGISTRATION,
        "first_run":FIRST_RUN_ID,
        "source_units":48,
        "source_sha256":FIRST["nor40_fixture.json"][1],
        "grade_sha256":FIRST["nor40_full.json"][1],
        "zip_sha256":FIRST_ZIP_SHA256,
        "original_science_pass":graded["registered_primary_pass"],
        "original_gates":graded["registered_five_gates"],
        "full_independent_replay_requested":bool(full),
    }
    if full:
        from research.nor40_strong_adversaries.receipt_audit import replay
        result["independent_replay"]=replay(
            contents["nor40_fixture.json"],contents["nor40_full.json"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",action="store_true")
    args=p.parse_args()
    print(json.dumps(audit(args.full),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
