"""Post-hoc, zero-neural goal-ancestry control for the frozen FC-06 test corpus.

Added AFTER the first registered FC-06 model result. It cannot be used to
claim a confirmatory learning result. This exposes a structural shortcut in
the test grammar: only backbone rules conclude backbone atoms, so a plain
syntactic reverse dependency slice identifies precisely the useful rules.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

from . import test_data
from ..fc05 import proofs


def goal_ancestors(task:dict)->set[str]:
    """Compute the syntactic backward dependency set using input rules only.

    This is a graph traversal, not a theorem prover or a source-label lookup.
    """
    ancestry={task["goal"]}
    for _ in range(len(task["atoms"])):
        old=len(ancestry)
        for rule in task["rules"]:
            if rule["head"] in ancestry:
                ancestry.update(rule["premises"])
        if len(ancestry)==old:
            return ancestry
    raise AssertionError("finite goal dependency closure did not stabilize")


def priority_trace(task:dict,budget:int=5)->list[int]:
    """Apply only legal rules whose head is syntactically goal-relevant."""
    ancestry=goal_ancestors(task)
    known=set(task["facts"])
    trace=[]
    while task["goal"] not in known and len(trace)<budget:
        legal=[i for i,rule in enumerate(task["rules"])
               if rule["head"] not in known
               and set(rule["premises"])<=known
               and rule["head"] in ancestry]
        if not legal:
            break
        # Tie-breaking is deterministic and never uses labels or proof traces.
        choice=min(legal)
        trace.append(choice)
        known.add(task["rules"][choice]["head"])
    return trace


def evaluate_records(tasks:list[dict])->dict:
    rows=[]
    errors=Counter()
    for t in tasks:
        trace=priority_trace(t)
        certificate=proofs.verify_trace(t,trace,budget=5)
        success=certificate["verified"]
        if success and t["label"]!="established":
            raise AssertionError("false positive proof escaped exact checker")
        if not success and t["label"]=="established":
            errors[(t["split"],t["family_id"])]+=1
        rows.append({"case_id":t["case_id"],"split":t["split"],
                     "family":t["family_id"],"source_label":t["label"],
                     "trace":trace,"steps":len(trace),
                     "verified_positive":success,
                     "checker_reason":certificate["reason"],
                     "goal_ancestors_count":len(goal_ancestors(t))})
    return {
        "status":"posthoc_exploratory_analysis",
        "method":"syntactic_backward_goal_ancestry_only",
        "model_parameters":0,"model_recurrent_cell_calls":0,
        "test_positive":sum(t["label"]=="established" for t in tasks),
        "test_negative":sum(t["label"]=="refuted" for t in tasks),
        "certified_positive":sum(r["verified_positive"] for r in rows),
        "false_certified_positive":0,
        "missed_positive":sum(errors.values()),
        "errors_by_family":{f"{a}/{b}":n for (a,b),n in sorted(errors.items())},
        "counts_by_split":{
            split:{"valid":sum(t["label"]=="established" for t in tasks if t["split"]==split),
                   "certified":sum(r["verified_positive"] for r in rows if r["split"]==split)}
            for split in ("test_same","test_deep")},
        "rows":rows
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    rows=test_data.generate_tests()
    report=evaluate_records(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
