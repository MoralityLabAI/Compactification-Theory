"""FC-07 virgin family-held-out alternative-producer Horn tests.

Written AFTER frozen train-only commit f470a97502b853addf2b0f67c9f6225001074a90.
The model training job checks out the older source revision where this
file does not exist, and never evaluates these cases for model selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from . import train_data as train
from ..formal_curriculum import generator as fc02
from ..fc06.posthoc_backward_slice import priority_trace
from ..fc05 import proofs

ROOT_SEED=20261113
TEST_FAMILIES=("reciprocal_forks","delayed_subgoal",
               "false_shortcuts","nested_switches")
TEST_SPECS={
    "test_same":(192,2,4,75323),
    "test_deep":(192,4,5,86453),
}
GUARD=train.GUARD


def _test_decoy_parents(rng:random.Random,family:str,
                        index:int,decoys:list[int],depth:int)->list[int]:
    prior=decoys[:index]
    spine=list(range(depth))
    if family=="reciprocal_forks":
        if index<2:
            return [rng.choice(spine)]
        if index%2:
            return sorted(set([prior[-2],rng.choice(spine)]))
        return [prior[-1]]
    if family=="delayed_subgoal":
        if index==0:
            return [max(0,depth-2)]
        if index%3==0:
            return [max(0,depth-1)]
        return sorted(set([prior[-1],max(0,depth-2)]))
    if family=="false_shortcuts":
        if index==0 or index%3==0:
            return [0]
        if index%2:
            return [prior[-1]]
        return sorted(set([prior[-1],rng.choice(spine)]))
    if family=="nested_switches":
        if index==0:
            return [0]
        if index==1:
            return [rng.choice(spine)]
        return sorted(set([prior[-1],prior[max(0,index//2-1)]]))
    raise ValueError("unsupported test family: "+family)


def _make(seed:int,depth:int,valid:bool,family:str)->dict:
    if not 2<=depth<=5:
        raise ValueError("FC07 test depth outside frozen range")
    rng=random.Random(seed)
    decoys=list(range(depth+1,GUARD))
    rules=[{"premises":[i-1],"head":i} for i in range(1,depth+1)]
    for j,atom in enumerate(decoys):
        rules.append({"premises":_test_decoy_parents(rng,family,j,decoys,depth),
                      "head":atom})
    for j,atom in enumerate(decoys):
        target=1+rng.randrange(depth)
        if family=="delayed_subgoal":
            target=depth if j%2 else max(1,depth-1)
        extra=[GUARD] if family=="false_shortcuts" and j%2==0 else []
        rules.append({"premises":sorted(set([target-1,atom,*extra])),
                      "head":target})
    rules.append({"premises":[GUARD],"head":GUARD})
    room=20-len(rules)
    extra=min(room,1+rng.randrange(3))
    for _ in range(extra):
        target=rng.choice(decoys)
        if family=="reciprocal_forks":
            source=rng.choice(decoys)
        elif family=="delayed_subgoal":
            source=rng.choice([*decoys,GUARD])
        elif family=="false_shortcuts":
            source=rng.choice([GUARD,*decoys])
        else:
            source=rng.choice([*decoys,GUARD])
        parents=[source]
        if rng.random()<.45:
            parents.append(rng.choice(list(range(depth))))
        rules.append({"premises":sorted(set(parents)),"head":target})

    if not valid:
        broken=1+rng.randrange(depth)
        for rule in rules:
            if rule["head"]==broken:
                rule["premises"]=sorted(set(rule["premises"]+[GUARD]))
    if len(rules)>20:
        raise AssertionError("more than 20 implication rules")
    names=[f"P{i}" for i in range(12)]
    perm=list(range(12))
    rng.shuffle(perm)
    alias={i:names[perm[i]] for i in range(12)}
    graph={
        "atoms":sorted(names),
        "facts":[alias[0]],
        "goal":alias[depth],
        "rules":[{"premises":sorted(alias[x] for x in rule["premises"]),
                  "head":alias[rule["head"]]} for rule in rules],
    }
    rng.shuffle(graph["rules"])
    return graph


def generate_tests()->list[dict]:
    earlier=train.generate_train_dev()
    prior=set(train.previous_fingerprints())
    seen=set(prior)
    seen.update(x["fingerprint"] for x in earlier)
    if len(seen)!=len(prior)+len(earlier):
        raise AssertionError("train or earlier fingerprint overlap")
    records=[]
    for split,(total,minimum,maximum,offset) in TEST_SPECS.items():
        accepted=Counter()
        attempts=0
        while sum(accepted.values())<total:
            attempts+=1
            if attempts>total*800:
                raise RuntimeError("insufficient unseen graph structures")
            family=TEST_FAMILIES[len(records)%len(TEST_FAMILIES)]
            i=accepted[family]
            valid=(i%2==0)
            depth=minimum+(i//2)%(maximum-minimum+1)
            seed=ROOT_SEED*100000+offset*10000+attempts
            name=f"FC07_{split}_{len(records):04d}"
            graph=_make(seed,depth,valid,family)
            item=train.certify(graph,split,family,depth,valid,name)
            if item["fingerprint"] in seen:
                continue
            seen.add(item["fingerprint"])
            records.append(item)
            accepted[family]+=1
        if any(accepted[f]!=total//len(TEST_FAMILIES) for f in TEST_FAMILIES):
            raise AssertionError("family balance changed")
    audit_tests(records,train_records=earlier)
    return records


def audit_tests(records:list[dict],*,train_records:list[dict]|None=None)->dict:
    if train_records is None:
        train_records=train.generate_train_dev()
    previous=set(train.previous_fingerprints())
    previous.update(x["fingerprint"] for x in train_records)
    seen=set()
    balanced=Counter()
    for item in records:
        split,family,label=item["split"],item["family_id"],item["label"]
        if split not in TEST_SPECS or family not in TEST_FAMILIES:
            raise AssertionError("unregistered test graph family")
        lo,hi=TEST_SPECS[split][1:3]
        if not lo<=item["depth"]<=hi:
            raise AssertionError("test depth invalid")
        if item["fingerprint"] in seen or item["fingerprint"] in previous:
            raise AssertionError("training, prior, or test WL overlap")
        seen.add(item["fingerprint"])
        graph={k:item[k] for k in ("atoms","facts","rules","goal")}
        expected=train.certify(graph,split,family,item["depth"],
                               label=="established",item["case_id"])
        if expected!=item:
            raise AssertionError("test annotation tampered")
        balanced[split,family,label]+=1
    for split,(total,_,__,___) in TEST_SPECS.items():
        for family in TEST_FAMILIES:
            for label in ("established","refuted"):
                if balanced[split,family,label]!=total//(2*len(TEST_FAMILIES)):
                    raise AssertionError("source split imbalance")
    controls={}
    for split in TEST_SPECS:
        cases=[x for x in records if x["split"]==split and x["label"]=="established"]
        ok=sum(proofs.verify_trace(x,priority_trace(x,budget=5),budget=5)["verified"]
               for x in cases)
        controls[split]={"valid":len(cases),"backward_slice_success":ok}
    return {
        "schema_version":1,
        "cases":len(records),
        "splits":{s:sum(x["split"]==s for x in records) for s in TEST_SPECS},
        "positive":sum(x["label"]=="established" for x in records),
        "negative":sum(x["label"]=="refuted" for x in records),
        "distinct_WL":len(seen),"prior_overlap":0,
        "backward_slice_first_observed":controls,
        "max_proof_actions":5,"reachable_novelty_on_valid":10,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    cases=generate_tests()
    report=audit_tests(cases)
    fc02.write_jsonl(args.out,cases)
    report["dataset_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(cases),encoding="utf8")
        report["lean_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
