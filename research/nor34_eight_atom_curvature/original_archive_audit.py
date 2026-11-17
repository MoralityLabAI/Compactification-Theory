"""NOR-34 permanent SHA-locked FIRST prospective source and result replay.

Verifies immutable complete-public source and full outcome bytes from
original green run #44054626746, then independently reconstructs
24 original native RRC25 programs, 384 three-way paired attack cohorts,
all exact conditional probabilities and 3,538,944 heldout decisions.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor34_eight_atom_curvature.receipt_audit import audit

ORIGINAL={
    "nor34_fixture.json":(5011440,
        "1cbf58a851df898a6e46fe4859d25ac5cf5b68a2e2b3e81d8e7afeede8dacb00"),
    "nor34_full.json":(337379,
        "4507769666f5560d33b8a0c99b4b5c85c5c2710b88e0021e77a85679106f6bd3"),
}


def audit_first_original():
    receipt=json.loads((HERE/"results/nor34_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor34.original_first_preregistered_eight_atom_curvature.v1"
    assert receipt["first_registration_commit"]==(
        "ef6d7b4d0374d0310a640258d94eaf474fb66c4b")
    assert receipt["original_green_run"].endswith("44054626746")
    assert receipt["original_artifact_id"]==14314703934
    assert receipt["file_bytes_sha256"]=={
        name:{"bytes":size,"sha256":h} for name,(size,h) in ORIGINAL.items()}
    assert receipt["new_public_source_units"]==24
    assert receipt["historical_public_source_units_excluded"]==84
    assert receipt["original_paired_attack_triplets"]==384
    assert receipt["all_original_policy_world_decisions_replayed"]==3538944
    assert receipt["direct_authentic_integer_witness_pairs_checked"]==27648
    assert receipt["frozen_mix_quanta"]==[2,0,1,1,4,8]
    assert receipt["coarse_residual_positive_source_count"]==24
    assert receipt["fine_residual_attenuation_source_count"]==24
    assert receipt["independent_exact_replay_passed"]
    assert receipt["original_registered_primary_pass"]
    assert all(receipt["original_preregistered_five_gates"].values())

    with ZipFile(HERE/"results/nor34_original_hosted.zip") as z:
        assert sorted(z.namelist())==sorted(ORIGINAL)
        assert z.testzip() is None
        blobs={name:z.read(name) for name in ORIGINAL}
    for name,(size,digest) in ORIGINAL.items():
        assert len(blobs[name])==size
        assert sha256(blobs[name]).hexdigest()==digest
    original=json.loads(blobs["nor34_full.json"])
    assert original["registered_primary_pass"]
    assert original["preregistered_five_gates"]==receipt["original_preregistered_five_gates"]
    assert original["source_mean_residual_fraction_explained"]==receipt["original_fraction_explained"]
    assert original["source_theoretical_coarse_residual_mean"]==receipt["original_mean_coarse_residual"]
    assert original["source_theoretical_fine_residual_mean"]==receipt["original_mean_fine_residual"]
    replayed=audit(blobs["nor34_fixture.json"],blobs["nor34_full.json"])
    assert replayed["new_source_count"]==24
    assert replayed["historical_sources_excluded"]==84
    assert replayed["paired_attack_triplets_replayed"]==384
    assert replayed["full_audit_world_decisions_replayed"]==3538944
    assert replayed["authentic_integer_witness_pair_checks"]==27648
    assert replayed["source_coarse_wins"]==24
    assert replayed["source_fine_attenuation_wins"]==24
    assert replayed["registered_primary_pass"]
    return replayed


if __name__=="__main__":
    print("NOR34 FIRST ORIGINAL SHA-LOCKED EIGHT-ATOM CURVATURE REPLAY VERIFIED",
          audit_first_original())
