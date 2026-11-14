"""FC-09 TEST-ONLY structurally held-out conjunctive Horn graphs.

This module was added AFTER train-only source commit
315119ead7264fbfbf5e58ee1b7b5c8b512d475d.
No training job may import it. The preregistered static symbolic control
scores are recorded here before any model looks at FC-09 test examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from . import train_data as train
from . import controls
from ..formal_curriculum import generator as fc02

SEED=20261113
FAMILIES=("tight_alternative_and","shared_bridge",
          "many_headed_goal","recursive_lures")
SPLITS={
    "test_same":(192,3,4,728747),
    "test_deep":(192,5,5,839963),
}


def _parents(rng:random.Random,family:str,index:int,
             decoys:list[int],depth:int)->list[int]:
    previous=decoys[:index]
    spine=list(range(depth))
    if family=="tight_alternative_and":
        if index<2:
            return [rng.choice(spine)]
        return sorted(set([previous[-1],rng.choice(spine)]))
    if family=="shared_bridge":
        if index<2:
            return [0]
        if index%2:
            return sorted(set([previous[0],previous[-1]]))
        return sorted(set([previous[0],rng.choice(spine)]))
    if family=="many_headed_goal":
        if index%3==0:
            return [0]
        if index==1:
            return [rng.choice(spine)]
        if index%2:
            return [previous[-1]]
        return sorted(set([previous[-1],rng.choice(spine)]))
    if family=="recursive_lures":
        if index<2:
            return [rng.choice(spine)]
        if index%3==0:
            return sorted(set([previous[-2],rng.choice(spine)]))
        if index%2:
            return [previous[-1]]
        return sorted(set([previous[-1],previous[0]]))
    raise ValueError("unregistered new test family "+family)


def generate_tests()->list[dict]:
    training=train.generate_train_dev()
    older=set(train.previous_fingerprints())
    seen=older|{x["fingerprint"] for x in training}
    if len(seen)!=len(older)+len(training):
        raise AssertionError("older/train source overlap")
    rows=[]
    for split,(total,lo,hi,offset) in SPLITS.items():
        groups=Counter()
        attempts=0
        while sum(groups.values())<total:
            attempts+=1
            if attempts>total*1200:
                raise RuntimeError("unable to generate test graph family "+split)
            family=FAMILIES[len(rows)%len(FAMILIES)]
            index=groups[family]
            valid=(index%2==0)
            depth=lo+(index//2)%(hi-lo+1)
            seed=SEED*100000+offset*10000+attempts
            identifier=f"FC09_{split}_{len(rows):04d}"
            graph=train.construct(seed,depth,valid,family,_parents)
            task=train.certify(graph,split,family,depth,valid,identifier)
            if task["fingerprint"] in seen:
                continue
            seen.add(task["fingerprint"])
            rows.append(task)
            groups[family]+=1
        if any(groups[family]!=total//len(FAMILIES) for family in FAMILIES):
            raise AssertionError("test family counts drifted")
    audit_tests(rows,training)
    return rows


def audit_tests(rows:list[dict],training:list[dict]|None=None)->dict:
    if training is None:
        training=train.generate_train_dev()
    earlier=set(train.previous_fingerprints())
    earlier.update(x["fingerprint"] for x in training)
    seen=set()
    counts=Counter()
    for task in rows:
        split,family,label=task["split"],task["family_id"],task["label"]
        if split not in SPLITS or family not in FAMILIES:
            raise AssertionError("unregistered test split/family")
        low,high=SPLITS[split][1:3]
        if not low<=task["depth"]<=high:
            raise AssertionError("held-out proof depth differs")
        finger=fc02._wl_fingerprint(task)
        if finger in earlier or finger in seen or finger!=task["fingerprint"]:
            raise AssertionError("test graph duplicated/tampered")
        seen.add(finger)
        raw={name:task[name] for name in ("atoms","facts","rules","goal")}
        record=train.certify(raw,split,family,task["depth"],
                             label=="established",task["case_id"])
        if record!=task:
            raise AssertionError("source test exact label or trace changed")
        counts[split,family,label]+=1
    for split,(total,_,__,___) in SPLITS.items():
        for family in FAMILIES:
            for label in ("established","refuted"):
                if counts[split,family,label]!=total//(2*len(FAMILIES)):
                    raise AssertionError("test label imbalance")
    control_scores=controls.first_source_controls(rows)
    return {
       "schema_version":1,
       "case_count":len(rows),
       "test_same":sum(x["split"]=="test_same" for x in rows),
       "test_deep":sum(x["split"]=="test_deep" for x in rows),
       "positives":sum(x["label"]=="established" for x in rows),
       "negatives":sum(x["label"]=="refuted" for x in rows),
       "new_families":list(FAMILIES),
       "distinct_WL":len(seen),"earlier_WL_overlap":0,
       "five_action_cap":train.BUDGET,
       "no_unary_goal_paths":True,
       "all_valid_goal_rules_are_conjunctive":True,
       "pre_model_symbolic_controls":control_scores,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    data=generate_tests()
    report=audit_tests(data)
    fc02.write_jsonl(args.out,data)
    report["dataset_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(data),encoding="utf8")
        report["lean_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
