"""FC-10 held-out REAL Lean goal-state test generator.

Committed AFTER train-only source 631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0.
Train CI must pin that earlier commit and must assert this module is absent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import core,train_data


def test_rows()->list[dict]:
    old=train_data.source_rows()
    rows=core.source_rows("test")
    if len(rows)!=48 or any(x["split"]!="test" for x in rows):
        raise AssertionError("frozen test count mismatch")
    if {x["family"] for x in rows}&{x["family"] for x in old}:
        raise AssertionError("train/dev grammar families leaked into test")
    old_fp={x["signature_sha256"] for x in old}
    new_fp=[x["signature_sha256"] for x in rows]
    if len(set(new_fp))!=len(rows) or old_fp.intersection(new_fp):
        raise AssertionError("source theorem signature repetition")
    return rows


def emit(out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    data=test_rows()
    manifest=out/"test.jsonl"
    source=out/"Test.lean"
    digest=core.write_jsonl(manifest,data)
    source.write_text(core.lean_document(data),encoding="utf8")
    receipt={
      "lean_version":core.LEAN_VERSION,
      "test_cases":len(data),
      "families":list(core.TEST_FAMILIES),
      "no_train_or_dev_family_overlap":True,
      "full_action_registry":list(core.TACTICS),
      "test_metadata_sha256":digest,
      "reference_test_lean_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
      "train_only_source_commit":"631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0",
    }
    (out/"test_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


def extract(root:Path,log:Path)->dict:
    tasks=[json.loads(x) for x in (root/"test.jsonl").read_text().splitlines()]
    states=core.parse_actual_goal_states(log.read_text(encoding="utf8"),tasks)
    digest=core.write_jsonl(root/"test_states.jsonl",states)
    return {
      "test_actual_lean_states":len(states),
      "real_goal_states_sha256":digest,
      "full_goal_turnstiles":sum(x["goal_state"].count("⊢") for x in states),
      "first_test_goal_preview":states[0]["goal_state"],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--extract-log",type=Path)
    a=p.parse_args()
    result=extract(a.out_dir,a.extract_log) if a.extract_log else emit(a.out_dir)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
