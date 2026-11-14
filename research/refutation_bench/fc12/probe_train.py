"""FC-12 train-only real Lean effect matrix of all context-typed tactic families.

Scratch uses sorry only to expose actual successor goal states. No scratch
result is an accepted theorem proof. The code never reads a heldout module.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from . import core,scratch
from .train_data import TRAIN_MODULES

REQUIRED_POSITIVE=("exact_local","rw_local","rcases_local","induction_list")
EXPECTED_GOALS_SHA="403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0"


def candidate_action_test(entry:dict,source:Path,family:str,script:str)->dict:
    original=(source/entry["source_file"]).read_text(encoding="utf8")
    start=time.perf_counter()
    try:
        actual=scratch.compile_candidate(original,entry["theorem"],[script],
                   original_stage=entry["stage_index"],probe=True,
                   timeout_seconds=18)
        accepted=bool(actual["accepted"])
        remaining="\n".join(actual["real_remaining_goal_states"])
        # A changed state or closed goal is a candidate's *training* signal.
        progressed=bool(accepted and (
            actual["no_goal_left_in_probe"] or
            remaining.strip()!=entry["actual_lean_goal_state"].strip()))
        closed=bool(accepted and actual["no_goal_left_in_probe"])
        return {
            "stage_id":entry["stage_id"],"state_sha256":entry["actual_state_sha256"],
            "family":family,"script":script,
            "source_file":entry["source_file"],"theorem":entry["theorem"],
            "source_original_commit":core.SOURCE_COMMIT,
            "scratch_only":True,"accepted":accepted,"progress":progressed,
            "closed_in_scratch":closed,"remaining_goal_count":actual["real_remaining_goal_count"],
            "compiler_returncode":actual["compiler_returncode"],
            "candidate_source_sha256":actual["source_sha256"],
            "lean_wall_ms":actual["lean_wall_ms"],
            "error_excerpt":actual["compiler_error_excerpt"],
        }
    except Exception as exc:
        # An exception is a failed action with the exception preserved in the
        # audit, rather than a training-positive invented from original proofs.
        return {
           "stage_id":entry["stage_id"],"state_sha256":entry["actual_state_sha256"],
           "family":family,"script":script,
           "source_file":entry["source_file"],"theorem":entry["theorem"],
           "source_original_commit":core.SOURCE_COMMIT,
           "scratch_only":True,"accepted":False,"progress":False,
           "closed_in_scratch":False,"remaining_goal_count":0,
           "compiler_returncode":None,"candidate_source_sha256":None,
           "lean_wall_ms":round((time.perf_counter()-start)*1000,6),
           "error_excerpt":type(exc).__name__+": "+str(exc)[:320],
        }


def run(source:Path,goalfile:Path,out:Path,*,workers:int=4)->dict:
    if (source/"lean-toolchain").read_text().strip()!=core.TOOLCHAIN:
        raise AssertionError("original pinned Lean version differs")
    if hashlib.sha256(goalfile.read_bytes()).hexdigest()!=EXPECTED_GOALS_SHA:
        raise AssertionError("train-source compiler goal SHA changed")
    rows=core.states_from_compiler(goalfile.parent)
    if len(rows)!=51:
        raise AssertionError("FC12 real train state count changed")
    if any(x["source_file"] not in TRAIN_MODULES for x in rows):
        raise AssertionError("heldout source family entered train labels")
    out.mkdir(parents=True,exist_ok=True)
    candidates=[]
    per_state=Counter()
    for row in rows:
        if row["goal_count"]!=1:
            continue
        generated=core.generate(row["actual_lean_goal_state"])
        for family in core.FAMILIES:
            per_state[family]+=int(bool(generated[family]))
            for script in generated[family]:
                if not scratch.valid_script(script):
                    raise AssertionError("unregistered script generated")
                candidates.append((row,source,family,script))
    began=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(lambda args:candidate_action_test(*args),candidates))
    if len(results)!=len(candidates):
        raise AssertionError("lost compiler tactic outcomes")
    digest=core.canonical_rows(out/"train_candidate_outcomes.jsonl",results)
    positives=Counter()
    closings=Counter()
    per_family=Counter()
    for r in results:
        per_family[r["family"]]+=1
        positives[r["family"]]+=int(r["progress"])
        closings[r["family"]]+=int(r["closed_in_scratch"])
    missing=[f for f in REQUIRED_POSITIVE if positives[f]<1]
    result={
       "experiment":"FC12_original_compiler_typed_tactic_train_probe",
       "source_original_commit":core.SOURCE_COMMIT,
       "source_lean_toolchain":core.TOOLCHAIN,
       "source_goals_sha256":EXPECTED_GOALS_SHA,
       "real_training_goal_count":len(rows),
       "candidate_count":len(results),
       "candidate_tactic_all_from_real_goal_context":True,
       "candidate_max_per_family":core.MAX_ARGUMENTS,
       "all_candidate_outcomes_compiler_probed":True,
       "progress_by_family":dict(positives),
       "closing_by_family":dict(closings),
       "attempts_by_family":dict(per_family),
       "states_with_candidate_by_family":dict(per_state),
       "mandatory_positive_action_families":list(REQUIRED_POSITIVE),
       "missing_mandatory_positive_families":missing,
       "formal_proofs_claimed_from_scratch":0,
       "original_training_source_files":sorted(TRAIN_MODULES),
       "compiler_results_sha256":digest,
       "elapsed_s":round(time.perf_counter()-began,3),
    }
    (out/"FC12_probe_receipt.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("FC12_TRAIN_PROBE",json.dumps(result,sort_keys=True),flush=True)
    if missing:
        raise RuntimeError("no genuine positive training examples for: "+", ".join(missing))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    run(args.source,args.states,args.out,workers=args.workers)


if __name__=="__main__":
    main()
