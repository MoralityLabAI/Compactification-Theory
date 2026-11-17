"""NOR-39 permanent original-source/grade SHA archive verifier.

Read-only. No fresh NOR39 sources are constructed. The default pass checks
the literal first raw fixture and grade bytes, their permanent ZIP manifest,
chronology metadata, historical 228-source disjointness and original verdict.
Use --full to independently regenerate all 68,419,584+ cross-play decisions
from the archived originals through the separately implemented replayer.

Strict pre-data chronology requires the original workflow record as well:
this code verifies the archive's recorded first-run provenance and bytes,
not a machine-wide absence of unlogged pre-data reads.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import sys
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RESULTS=HERE/"results"
REGISTERED="4a5f3750d240d46c031e0cb432fb7b1a884cfbe9"
FIRST={
    "nor39_fixture.json":(
        8457812,
        "3dd23e71663d3c179ecacd437892c650469a6980230e71c353840f3b8f5a290c"),
    "nor39_full.json":(
        930059,
        "995b6e110c829fd9a8ebb036e4e08ef64cfccee8298bff0def255c53f35c17c6"),
}
ZIP_BYTES=3014180
ZIP_SHA="492b54bec77975d6c0c36f810fba38cc086a11c8e47b7d487767b94bb0efde10"


def audit(full_replay: bool=False):
    manifest=json.loads((RESULTS/"nor39_first_frozen_receipt.json").read_text())
    assert manifest["schema"]=="nor39.original_first_grade_archive.v1"
    assert manifest["registration"]==REGISTERED
    assert manifest["original_grade_run"]==44102073143
    assert manifest["artifact_id"]==14336748652
    assert manifest["original_bytes_sha256"]=={
        name:{"bytes":n,"sha256":h} for name,(n,h) in FIRST.items()}
    assert manifest["source_units"]==48
    assert manifest["historical_source_units_excluded"]==228
    assert manifest["procedural_source_sequence_clean"] is True
    zpath=RESULTS/"nor39_original_first_hosted.zip"
    buf=zpath.read_bytes()
    assert len(buf)==manifest["zip_archive_bytes"]==ZIP_BYTES
    assert sha256(buf).hexdigest()==manifest["zip_archive_sha256"]==ZIP_SHA
    with ZipFile(zpath) as z:
        assert sorted(z.namelist())==sorted(FIRST)
        assert z.testzip() is None
        originals={name:z.read(name) for name in FIRST}
    for name,(n,h) in FIRST.items():
        assert len(originals[name])==n
        assert sha256(originals[name]).hexdigest()==h
    fixture=json.loads(originals["nor39_fixture.json"])
    grade=json.loads(originals["nor39_full.json"])
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert fixture["schema"]=="nor39.clean_global_uniform_public_only_reps68to75.v1"
    assert grade["schema"]=="nor39.clean_global_uniform_policyaware.v1"
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert fixture["replicates"]==list(range(68,76))
    assert fixture["excluded_historical_sources"]==grade["historical_sources_excluded"]==228
    assert len(fixture["cases"])==len(grade["source_records"])==48
    assert grade["public_fixture_sha256"]==FIRST["nor39_fixture.json"][1]
    assert grade["registered_primary_pass"] is True
    assert len(grade["registered_five_gates"])==5
    assert all(grade["registered_five_gates"].values())
    assert grade["registered_five_gates"]==manifest["scorer_gates"]
    assert grade["source_means"]==manifest["source_means"]
    assert grade["source_direction_wins"]==manifest["source_direction_wins"]
    assert all(not set(r).intersection(
        ("program","future","target","attack","witness","audit"))
        for r in fixture["cases"])
    # Historical witness archives provide the independent old-source roster.
    # Imports occur after raw SHA validation and never generate a NOR39 source.
    from research.nor39_clean_global_uniform import receipt_audit as independent
    history=independent.historical_source_ids()
    ids=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(history)==len(set(history))==228
    assert len(set(ids))==48 and set(ids).isdisjoint(set(history))
    result={
        "schema":"nor39.original_first_immutable_archive_audit.v1",
        "registration":REGISTERED,"source_identities":len(ids),
        "historical_excluded":228,"source_sha256":FIRST["nor39_fixture.json"][1],
        "grade_sha256":FIRST["nor39_full.json"][1],
        "zip_sha256":ZIP_SHA,
        "scorer_original_pass":grade["registered_primary_pass"],
        "registered_gates":grade["registered_five_gates"],
        "archived_process_claim_clean":manifest["procedural_source_sequence_clean"],
        "full_independent_replay":bool(full_replay)}
    if full_replay:
        result["independent_replay"]=independent.replay(
            originals["nor39_fixture.json"],originals["nor39_full.json"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",action="store_true")
    args=p.parse_args()
    print(json.dumps(audit(full_replay=args.full),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
