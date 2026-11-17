"""NOR-30 permanent read-only SHA and original source/attack replay audit.

Checks first hosted three byte streams against the archived source-locked
receipt; then recreates all original 12 TEST source programs and every
cross-play count with a separately written trial-ID-set replayer.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor30_minimax_audits.receipt_audit import replay_original

FILES=("nor30_train_receipt.json","nor30_fixture.json","nor30_full.json")
EXPECTED={
    "nor30_train_receipt.json":(3113,"df9414c3ee3132a90fe8e6cb5fd231332e480fe9e87ea5b3c3bef86045330c84"),
    "nor30_fixture.json":(2284577,"9961afd81cac78b32fc5280fe54291f7db4dca0cf85cb6624560ddac420e1878"),
    "nor30_full.json":(337563,"4189235f7cb5c14f3c7d1028abda264d893c7b5e5d2eda0c4c1de091f052b9c8"),
}


def audit_original():
    receipt=json.loads((HERE/"results/nor30_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor30.original_sourceheldout_minimax_first_run.v1"
    assert receipt["registration_first_commit"]==(
        "118dc142b1bd9143218f9868fea248c53ddc62dc")
    assert receipt["original_artifact_id"]==14303208219
    assert receipt["original_complete_green_run"].endswith("44027909426")
    assert receipt["first_run_primary_pass"]
    assert receipt["constrained_minimax_weights"]==[3,0,0,13,0,0,0]
    assert receipt["new_test_public_sources"]==12
    assert receipt["historical_training_public_sources"]==12
    assert receipt["distributional_degeneracy"]
    archive=HERE/"results/nor30_original_hosted.zip"
    assert archive.is_file()
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(FILES)
        assert z.testzip() is None
        originals={name:z.read(name) for name in FILES}
    for name,(count,digest) in EXPECTED.items():
        raw=originals[name]
        assert len(raw)==count
        assert sha256(raw).hexdigest()==digest
        assert receipt["original_file_bytes_and_sha256"][name]=={
            "bytes":count,"sha256":digest}
    tr=json.loads(originals[FILES[0]])
    fu=json.loads(originals[FILES[2]])
    assert fu["training_receipt_sha256"]==EXPECTED[FILES[0]][1]
    assert fu["fixture_sha256"]==EXPECTED[FILES[1]][1]
    assert tr["historical_matrix_sha256"]==receipt["historical_7x7_integer_game_sha256"]
    assert fu["primary_wins_against_hybrid50_self_response"]==12
    assert fu["primary_wins_against_hybrid50_worst"]==12
    with TemporaryDirectory(prefix="nor30-original-") as d:
        paths={}
        for name,raw in originals.items():
            p=Path(d)/name
            p.write_bytes(raw)
            paths[name]=p
        audited=replay_original(paths[FILES[0]],paths[FILES[1]],paths[FILES[2]])
    assert audited["historical_train_source_count"]==12
    assert audited["prospective_test_source_count"]==12
    assert audited["full_policy_decisions_replayed"]==4816896
    assert audited["candidate_training_grid_replayed"]==74613
    assert audited["primary_pass"]
    return audited


if __name__=="__main__":
    print("NOR30 ORIGINAL FIRST-RUN SOURCE-LOCKED ARCHIVE AUDIT PASSED",
          audit_original())
