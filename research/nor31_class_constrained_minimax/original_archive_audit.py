"""Permanent NOR-31 original TRAIN / fresh TEST byte-lock and full replay.

Original hosted artifact from GitHub Actions #44032550237 was preserved
before interpretations. Every CI execution checks all three original byte
identities and independently reconstructs historical optimizer, fresh
source programs, attack sets and 4,128,768 cross-play decisions.
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
from research.nor31_class_constrained_minimax.receipt_audit import replay_all

EXPECTED={
    "nor31_train_receipt.json":(5300,"ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0"),
    "nor31_fixture.json":(2245544,"b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85"),
    "nor31_full.json":(317452,"f978a2c388715056e7999e01d4ddc30f6c24a778b0e6e3e741a070902223430c")
}


def audit():
    receipt=json.loads((HERE/"results/nor31_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor31.original_sourceheldout_sixclass_first_run.v1"
    assert receipt["preregistered_first_commit"]==(
        "7e37e8c57a68e7224a67626d2495e9bcf922a424")
    assert receipt["original_artifact_id"]==14306125618
    assert receipt["original_fully_green_run"].endswith("44032550237")
    assert receipt["original_file_sizes_and_sha256"]=={
        k:{"bytes":v[0],"sha256":v[1]} for k,v in EXPECTED.items()}
    assert receipt["independent_replay_passed"] and receipt["registered_primary_pass"]
    assert receipt["trained_sixclass_weights"]==[2,0,1,1,4,8]
    assert receipt["source_self_response_wins"]==12
    assert receipt["source_worstcase_wins"]==12
    assert receipt["source_template_retention_75pct_count"]==10
    assert receipt["train_grid_enumerated"]==20349
    assert receipt["train_grid_feasible"]==9424
    assert all(receipt["registered_five_gates"].values())

    archive=HERE/"results/nor31_original_hosted.zip"
    assert archive.is_file()
    with ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(EXPECTED)
        assert z.testzip() is None
        original={k:z.read(k) for k in EXPECTED}
    for name,raw in original.items():
        count,digest=EXPECTED[name]
        assert len(raw)==count
        assert sha256(raw).hexdigest()==digest

    out=json.loads(original["nor31_full.json"])
    tr=json.loads(original["nor31_train_receipt.json"])
    source=json.loads(original["nor31_fixture.json"])
    assert tr["fitted_constrained"]["weights"]==receipt["trained_sixclass_weights"]
    assert out["train_receipt_sha256"]==EXPECTED["nor31_train_receipt.json"][1]
    assert out["test_public_fixture_sha256"]==EXPECTED["nor31_fixture.json"][1]
    assert out["registered_primary_pass"]
    assert len(source["cases"])==len(out["source_records"])==12
    assert out["primary_template_75pct_source_count"]==10

    with TemporaryDirectory(prefix="nor31-original-") as tmp:
        paths={}
        for name,raw in original.items():
            dest=Path(tmp)/name
            dest.write_bytes(raw)
            paths[name]=dest
        replayed=replay_all(paths["nor31_train_receipt.json"],
                            paths["nor31_fixture.json"],paths["nor31_full.json"])
    assert replayed["historical_training_source_units"]==24
    assert replayed["fresh_test_source_units"]==12
    assert replayed["replayed_heldout_policy_world_decisions"]==4128768
    assert replayed["registered_primary_pass"]
    return replayed


if __name__=="__main__":
    print("NOR31 ORIGINAL SHA-LOCKED TRAIN AND TEST FULL REPLAY PASSED",audit())
