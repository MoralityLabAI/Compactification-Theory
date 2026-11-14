"""FC-10 train/dev-only real Lean tactic-state source and log extractor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from . import core


def source_rows()->list[dict]:
    rows=core.source_rows("train")+core.source_rows("dev")
    fingerprints=[r["signature_sha256"] for r in rows]
    if len(set(fingerprints))!=len(rows):
        raise AssertionError("duplicate signature across train and dev")
    if {r["family"] for r in rows if r["split"]=="train"} & {
       r["family"] for r in rows if r["split"]=="dev"}:
        raise AssertionError("train/dev family overlap")
    return rows


def emit(out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    records=source_rows()
    manifest=out/"train_dev.jsonl"
    source=out/"TrainDev.lean"
    digest=core.write_jsonl(manifest,records)
    source.write_text(core.lean_document(records),encoding="utf8")
    receipt={
        "source":"FC10 native Lean goal-state source",
        "lean_version":core.LEAN_VERSION,
        "total":len(records),"train":132,"dev":24,
        "families_train":list(core.TRAIN_FAMILIES),
        "families_dev":list(core.DEV_FAMILIES),
        "registered_tactics":list(core.TACTICS),
        "metadata_jsonl_sha256":digest,
        "reference_lean_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_generator_absent":True,
    }
    (out/"train_receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def extract(manifest:Path,log:Path,out:Path)->dict:
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    goal_states=core.parse_actual_goal_states(log.read_text(encoding="utf8"),rows)
    digest=core.write_jsonl(out,goal_states)
    result={
       "compiler_produced_goal_states":len(goal_states),
       "train":sum(x["split"]=="train" for x in goal_states),
       "dev":sum(x["split"]=="dev" for x in goal_states),
       "lean_goal_turnstiles":sum(x["goal_state"].count("⊢") for x in goal_states),
       "real_goal_states_sha256":digest,
       "first_goal_preview":goal_states[0]["goal_state"],
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--extract-log",type=Path)
    args=p.parse_args()
    if args.extract_log:
        response=extract(args.out_dir/"train_dev.jsonl",args.extract_log,
                         args.out_dir/"train_states.jsonl")
    else:
        response=emit(args.out_dir)
    print(json.dumps(response,sort_keys=True,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
