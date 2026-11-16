"""NOR-24 immutable original source and heldout result byte audit."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor24_adaptive_audits.receipt_audit import audit

def audit_archived_original():
    receipt=json.loads((HERE/"results/nor24_first_frozen_receipt.json").read_text())
    assert receipt["schema"]=="nor24.first_preregistered_train_only_posterior_audits.v1"
    assert receipt["preregistered_source_commit"]==(
        "780c822c34314c0dead2cd1331280ca2f8978332")
    assert receipt["registered_primary_pass"] and receipt["independent_replay_pass"]
    with ZipFile(HERE/"results/nor24_original_hosted.zip") as z:
        assert sorted(z.namelist())==["nor24_fixture.json","nor24_full.json"]
        assert z.testzip() is None
        fixture=z.read("nor24_fixture.json")
        full=z.read("nor24_full.json")
    assert len(fixture)==receipt["fixture_bytes"]==4300923
    assert len(full)==receipt["result_bytes"]==737700
    assert sha256(fixture).hexdigest()==receipt["fixture_sha256"]
    assert sha256(full).hexdigest()==receipt["result_sha256"]
    assert receipt["fixture_sha256"]==(
        "88285e2400566c633d211071ace59784987d6458d743db478321dc90f33ef001")
    assert receipt["result_sha256"]==(
        "ed54eabd70637a42c34caf66d04bcadd50f890a1ced46cea0132f8767b476807")
    result=audit(fixture,full)
    assert result["original_sources_replayed"]==24
    assert result["train_episodes"]==768
    assert result["test_clean_schedules_replayed"]==12288
    assert result["independent_policy_world_budget_checks"]==2764800
    assert result["registered_primary_pass"]
    return result

if __name__=="__main__":
    print("NOR24 ORIGINAL FIRST RECEIPT VERIFIED",audit_archived_original())
