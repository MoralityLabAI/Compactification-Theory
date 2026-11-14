"""FC-11 train-only Lean tactic effect matrix over original MathCan goals.

All eight preregistered action scripts are actually attempted against the
original source context of every TRAIN goal state. Scratch probes may use
sorry to expose a successor state; they are NEVER positive certificates.
The evaluator only counts a theorem after separate sorry-free compilation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from . import source_core as core
from . import scratch

FROZEN_TRAIN_GOALS_SHA="47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d"
FROZEN_TRAIN_SOURCE_REV="2e7e749d4c7f8610a4083f4f8551d084102ea55a"


def make_probes(source:Path,states_file:Path,out:Path)->dict:
    core.source_lean_version(source)
    if hashlib.sha256(states_file.read_bytes()).hexdigest()!=FROZEN_TRAIN_GOALS_SHA:
        raise AssertionError("train states differ from frozen source CI")
    rows=[json.loads(x) for x in states_file.read_text(encoding="utf8").splitlines()]
    if len(rows)!=31 or len({x["stage_id"] for x in rows})!=len(rows):
        raise AssertionError("frozen source-stage count changed")
    out.mkdir(parents=True,exist_ok=True)
    source_cache={}
    attempts=[]
    step_counts=Counter()
    start=time.perf_counter()
    for i,row in enumerate(rows):
        if row["goal_count"]!=1:
            raise AssertionError("registered training goal requires one open Lean goal")
        path=row["source_file"]
        if path not in source_cache:
            source_cache[path]=(source/path).read_text(encoding="utf8")
        original=source_cache[path]
        for action in core.TACTICS:
            result=scratch.action_effect(original,row["theorem"],action,
                   row["stage_index"],row["actual_lean_goal_state"])
            step_counts[action]+=1
            attempts.append({
                "stage_id":row["stage_id"],
                "source_file":path,
                "state_sha256":row["actual_state_sha256"],
                "source_original_commit":core.SOURCE_COMMIT,
                "action":action,
                "action_index":core.TACTICS.index(action),
                "reference_command_not_a_label":row["source_command"],
                "accepted_scratch_probe":result["accepted"],
                "progress":result["progress"],
                "closed_in_scratch_probe":result["no_goal_left_in_probe"],
                "remaining_goals":result["real_remaining_goal_count"],
                "candidate_source_sha256":result["source_sha256"],
                "actual_lean_wall_ms":result["lean_wall_ms"],
                "compiler_error_excerpt":result["compiler_error_excerpt"],
            })
        print("FC11_TRAIN_PROBED",json.dumps({
            "stage_id":row["stage_id"],
            "action_classes_attempted":len(core.TACTICS),
            "valid":sum(x["accepted_scratch_probe"] for x in attempts[-len(core.TACTICS):]),
            "progress":sum(x["progress"] for x in attempts[-len(core.TACTICS):]),
            "closed":sum(x["closed_in_scratch_probe"] for x in attempts[-len(core.TACTICS):])
        },sort_keys=True),flush=True)
    digest=core.write_rows(out/"action_probes.jsonl",attempts)
    report={
      "status":"train-only scratch tactic effect exploration, untrusted",
      "source_commit":core.SOURCE_COMMIT,
      "train_only_revision":FROZEN_TRAIN_SOURCE_REV,
      "real_train_states_sha256":FROZEN_TRAIN_GOALS_SHA,
      "source_stage_count":len(rows),
      "action_classes":list(core.TACTICS),
      "total_actual_compiler_tactic_probes":len(attempts),
      "all_tactic_classes_represented":all(step_counts[a]==len(rows)
                                           for a in core.TACTICS),
      "scratch_probes_not_kernel_certificates":True,
      "accepted_tactic_attempts":sum(p["accepted_scratch_probe"] for p in attempts),
      "progressive_attempts":sum(p["progress"] for p in attempts),
      "closed_attempts":sum(p["closed_in_scratch_probe"] for p in attempts),
      "source_stage_coverage":dict(step_counts),
      "action_probe_jsonl_sha256":digest,
      "probe_elapsed_s":round(time.perf_counter()-start,3),
    }
    (out/"train_probe_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("FC11_PROBE_REPORT",json.dumps(report,sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    make_probes(a.source,a.states,a.out)


if __name__=="__main__":
    main()
