"""FC-08 test-only conjunctive Horn families, added AFTER train source freeze.

Train-only source commit was 51d0082f27e4c6214ec0cebb4886df44a4962321
and predates this file. Do not load FC-08 test data during training.
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
from ..fc05 import proofs
from ..formal_curriculum import generator as fc02

ROOT_SEED=20261113
TEST_FAMILIES=("paired_requirements","nested_and_switch",
               "goal_misdirection","multi_support_cycle")
TEST_SPECS={
    "test_same":(192,3,4,303941),
    "test_deep":(192,5,5,405179),
}


def test_decoy_parents(rng:random.Random,family:str,index:int,
                       decoys:list[int],depth:int)->list[int]:
    earlier=decoys[:index]
    spine=list(range(depth))
    if family=="paired_requirements":
        if index<2:
            return [rng.choice(spine)]
        if index%2==0:
            return sorted({earlier[-2],earlier[-1]})
        return sorted({earlier[-1],rng.choice(spine)})
    if family=="nested_and_switch":
        if index==0:
            return [0]
        if index==1:
            return [rng.choice(spine)]
        if index%2:
            return [earlier[index//2]]
        return sorted({earlier[-1],earlier[max(0,index//2-1)]})
    if family=="goal_misdirection":
        if index<3:
            return [rng.choice(spine)]
        if index%3==0:
            return [0]
        if index%2:
            return sorted({earlier[-1],rng.choice(spine)})
        return [earlier[-2]]
    if family=="multi_support_cycle":
        if index<2:
            return [rng.choice(spine)]
        if index%3==0:
            return [earlier[-2]]
        if index%3==1:
            return sorted({earlier[-1],rng.choice(spine)})
        return sorted({earlier[-1],earlier[-2]})
    raise ValueError("unregistered test family: "+family)


def generate_tests()->list[dict]:
    training=train.generate_train_dev()
    prior=set(train.previous_fingerprints())
    seen=set(prior)
    seen.update(row["fingerprint"] for row in training)
    if len(seen)!=len(prior)+len(training):
        raise AssertionError("training or earlier data shared a WL fingerprint")
    records=[]
    for split,(count,dmin,dmax,offset) in TEST_SPECS.items():
        groups=Counter()
        attempts=0
        while sum(groups.values())<count:
            attempts+=1
            if attempts>count*1200:
                raise RuntimeError("insufficient novel FC08 test graphs in "+split)
            family=TEST_FAMILIES[len(records)%len(TEST_FAMILIES)]
            k=groups[family]
            good=k%2==0
            depth=dmin+(k//2)%(dmax-dmin+1)
            seed=ROOT_SEED*100000+offset*10000+attempts
            case_id=f"FC08_{split}_{len(records):04d}"
            raw=train.construct(seed,depth,good,family,test_decoy_parents)
            entry=train.certify(raw,split,family,depth,good,case_id)
            if entry["fingerprint"] in seen:
                continue
            seen.add(entry["fingerprint"])
            records.append(entry)
            groups[family]+=1
        if any(groups[family]!=count//len(TEST_FAMILIES)
               for family in TEST_FAMILIES):
            raise AssertionError("family count imbalance")
    audit_tests(records,training)
    return records


def audit_tests(records:list[dict],training:list[dict]|None=None)->dict:
    if training is None:
        training=train.generate_train_dev()
    prior=set(train.previous_fingerprints())
    prior.update(x["fingerprint"] for x in training)
    seen=set()
    balance=Counter()
    for row in records:
        split,fam,label=row["split"],row["family_id"],row["label"]
        if split not in TEST_SPECS or fam not in TEST_FAMILIES:
            raise AssertionError("wrong test family")
        dmin,dmax=TEST_SPECS[split][1:3]
        if not dmin<=row["depth"]<=dmax:
            raise AssertionError("wrong test proof depth")
        fp=fc02._wl_fingerprint(row)
        if fp in seen or fp in prior or fp!=row["fingerprint"]:
            raise AssertionError("test WL overlap")
        seen.add(fp)
        raw={k:row[k] for k in ("atoms","facts","rules","goal")}
        correct=train.certify(raw,split,fam,row["depth"],
                              label=="established",row["case_id"])
        if correct!=row:
            raise AssertionError("test certificate modified")
        balance[split,fam,label]+=1
    for split,(count,_,__,___) in TEST_SPECS.items():
        for fam in TEST_FAMILIES:
            for label in ("established","refuted"):
                if balance[split,fam,label]!=count//(2*len(TEST_FAMILIES)):
                    raise AssertionError("test label/family imbalance")
    return {
      "schema_version":1,"count":len(records),
      "test_same":sum(x["split"]=="test_same" for x in records),
      "test_deep":sum(x["split"]=="test_deep" for x in records),
      "positive":sum(x["label"]=="established" for x in records),
      "negative":sum(x["label"]=="refuted" for x in records),
      "distinct_WL":len(seen),"prior_WL_overlap":0,
      "action_budget":5,
      "all_valid_proofs_require_and":True,
      "first_observed_source_only_controls":
        controls.first_observed_receipt(records,proofs.verify_trace),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    rows=generate_tests()
    report=audit_tests(rows)
    fc02.write_jsonl(args.out,rows)
    report["dataset_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        report["lean_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
