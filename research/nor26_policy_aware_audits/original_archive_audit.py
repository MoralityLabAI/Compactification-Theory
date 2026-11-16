"""NOR-26 first raw hosted evidence SHA lock.

Read-only audit of the exact byte streams in the permanently committed ZIP,
originally produced by GitHub Actions #43895351638, before any interpretation
or retrospective summary. Does not import the NOR-26 experiment implementation.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parent
RECEIPT=ROOT/"results"/"nor26_first_original_receipt.json"
ARCHIVE=ROOT/"results"/"nor26_original_hosted.zip"


def check():
    ref=json.loads(RECEIPT.read_text())
    assert ref["preregistration_sha"]=="51ee62518948505754236959c77930594daea011"
    assert ref["first_hosted_actions_run"]==43895351638
    assert ref["original_hosted_artifact_id"]==14237404539
    assert ARCHIVE.is_file(), "original NOR26 artifact ZIP has not yet been committed"
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert set(archive.namelist())=={"nor26_fixture.json","nor26_full.json"}
        original={}
        for key in ("fixture","outcome"):
            metadata=ref[key]
            raw=archive.read(metadata["file"])
            assert len(raw)==metadata["bytes"],(key,"bytes",len(raw))
            assert sha256(raw).hexdigest()==metadata["sha256"],(key,"SHA256")
            original[key]=json.loads(raw)
    src=original["fixture"]
    z=original["outcome"]
    assert src["source_only"] and src["hidden_truth_absent"] and src["attack_sets_absent"]
    assert src["schema"]=="nor26.fresh_public_only_12_sources.v1"
    assert len(src["cases"])==12
    assert len({c["full_public_input_sha256"] for c in src["cases"]})==12
    assert z["schema"]=="nor26.policy_aware_blind_attack_crossplay.v1"
    assert len(z["source_records"])==12
    assert z["fixture_sha256"]==ref["fixture"]["sha256"]
    assert z["registered_primary_pass"]==ref["result"]["registered_primary_pass"]
    assert z["primary_hybrid50_vs_static_wins"]==ref["result"]["wins_hybrid50_over_static"]
    assert z["primary_hybrid50_selfplay_mean"]==ref["result"]["source_mean_hybrid50_diagonal"]
    assert all(len(row["crossplay"])==49 and
               row["informed_detected"]==row["legitimate_failures"]==row["delta_disagreements"]==0
               for row in z["source_records"])
    assert sum(len(row["crossplay"]) for row in z["source_records"])==588
    print("NOR26 first original bytes and source-locked result PASS")
    print("source sha256",ref["fixture"]["sha256"])
    print("outcome sha256",ref["outcome"]["sha256"])
    print("source units",len(z["source_records"]),"registered outcome",z["registered_primary_pass"])


if __name__=="__main__":
    check()
