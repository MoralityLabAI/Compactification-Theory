"""FC-06 untouched family-held-out tests, added AFTER the train-only source pin.

The training job is required to checkout FC-06 source at
1bf514325788b633432eb9f3e49627bd923a216e, which does NOT contain
this module. The new test graph grammars are disjoint from train/dev.
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

ROOT_SEED=20261113
TEST_FAMILIES=("zigzag_lures","nested_fork","late_spoke","alternating_gate")
TEST_SPECS={
    "test_same":(192,2,4,33629),
    "test_deep":(192,4,5,44839),
}


def _decoy_premises(rng:random.Random,family:str,
                    idx:int,decoys:list[int],depth:int)->list[int]:
    earlier=decoys[:idx]
    early_spine=list(range(depth))
    if family=="zigzag_lures":
        if idx<2:
            return [0]
        if idx%2:
            return sorted(set([earlier[-2],rng.choice(early_spine)]))
        return [earlier[-2]]
    if family=="nested_fork":
        if idx<3:
            return [0]
        return [earlier[max(0,(idx//2)-1)]]
    if family=="late_spoke":
        if idx<max(2,len(decoys)//2):
            return [0]
        return [rng.choice(early_spine)]
    if family=="alternating_gate":
        if idx==0:
            return [0]
        if idx%2:
            return sorted(set([earlier[-1],rng.choice(early_spine)]))
        return [0]
    raise ValueError("unknown FC-06 test grammar")


def _make(seed:int,depth:int,valid:bool,family:str)->dict:
    if not 2<=depth<=5:
        raise ValueError("FC-06 test depth limit")
    rng=random.Random(seed)
    guard=train.GUARD
    decoys=list(range(depth+1,guard))
    rules=[{"premises":[k-1],"head":k} for k in range(1,depth+1)]
    if not valid:
        broken=rng.randrange(depth)
        rules[broken]["premises"]=sorted({*rules[broken]["premises"],guard})

    for i,atom in enumerate(decoys):
        parents=_decoy_premises(rng,family,i,decoys,depth)
        rules.append({"premises":sorted(set(parents)),"head":atom})
    # Extra reachable decoy rules enrich topology without creating
    # alternate goal proofs or repairing the blocked backbone.
    extra=min(len(decoys),rng.randint(3,7))
    for _ in range(extra):
        i=rng.randrange(len(decoys))
        atom=decoys[i]
        prior=decoys[:i]
        support=[0,*range(depth),*prior]
        chosen=rng.choice(support)
        parents=[chosen]
        if prior and rng.random()<.40:
            parents.append(rng.choice(prior))
        rule={"premises":sorted(set(parents)),"head":atom}
        if rule not in rules:
            rules.append(rule)
    rules.append({"premises":[guard],"head":guard})
    if len(rules)>20:
        raise AssertionError("test rule cap")

    names=[f"P{i}" for i in range(train.ATOM_COUNT)]
    perm=list(range(train.ATOM_COUNT))
    rng.shuffle(perm)
    alias={i:names[perm[i]] for i in range(train.ATOM_COUNT)}
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
    train_rows=train.generate_train_dev()
    seen=set(train.older_fingerprints())
    seen.update(x["fingerprint"] for x in train_rows)
    if len(seen)!=len(train.older_fingerprints())+len(train_rows):
        raise AssertionError("old/new structural overlap")

    output=[]
    for split,(total,dmin,dmax,offset) in TEST_SPECS.items():
        accepted=Counter()
        attempts=0
        while sum(accepted.values())<total:
            attempts+=1
            if attempts>total*800:
                raise RuntimeError("unable to generate unique "+split)
            family=TEST_FAMILIES[sum(accepted.values())%len(TEST_FAMILIES)]
            index=accepted[family]
            valid=(index%2==0)
            depth=dmin+(index//2)%(dmax-dmin+1)
            seed=ROOT_SEED*100000+offset*10000+attempts
            case_id=f"FC06_{split}_{len(output):04d}"
            graph=_make(seed,depth,valid,family)
            row=train.certify(graph,split=split,family=family,
                              depth=depth,valid=valid,case_id=case_id)
            if row["fingerprint"] in seen:
                continue
            seen.add(row["fingerprint"])
            output.append(row)
            accepted[family]+=1
        if any(accepted[f]!=total//len(TEST_FAMILIES) for f in TEST_FAMILIES):
            raise AssertionError("imbalanced test family")
    audit_tests(output,train_rows=train_rows)
    return output


def audit_tests(rows:list[dict],*,train_rows:list[dict]|None=None)->dict:
    if train_rows is None:
        train_rows=train.generate_train_dev()
    prior=set(train.older_fingerprints())
    prior.update(r["fingerprint"] for r in train_rows)
    seen=set()
    counts=Counter()
    for row in rows:
        s=row["split"];fam=row["family_id"];label=row["label"]
        if s not in TEST_SPECS or fam not in TEST_FAMILIES:
            raise AssertionError("unexpected test split/family")
        dmin,dmax=TEST_SPECS[s][1:3]
        if not dmin<=row["depth"]<=dmax:
            raise AssertionError("incorrect test depth")
        if row["fingerprint"] in seen or row["fingerprint"] in prior:
            raise AssertionError("heldout data overlap")
        seen.add(row["fingerprint"])
        raw={k:row[k] for k in ("atoms","facts","rules","goal")}
        expected=train.certify(raw,split=s,family=fam,depth=row["depth"],
                              valid=(label=="established"),case_id=row["case_id"])
        if expected!=row:
            raise AssertionError("test annotation mismatch")
        counts[s,fam,label]+=1

    for split,(n,_,__,___) in TEST_SPECS.items():
        for fam in TEST_FAMILIES:
            for label in ("established","refuted"):
                if counts[split,fam,label]!=n//(2*len(TEST_FAMILIES)):
                    raise AssertionError("test family imbalance")
    return {"schema_version":1,"case_count":len(rows),
            "source_train_count":len(train_rows),
            "test_same":sum(x["split"]=="test_same" for x in rows),
            "test_deep":sum(x["split"]=="test_deep" for x in rows),
            "positives":sum(x["label"]=="established" for x in rows),
            "negatives":sum(x["label"]=="refuted" for x in rows),
            "distinct_WL_graphs":len(seen),
            "overlap_count":0,"max_proof_actions":train.ACTION_BUDGET,
            "positive_max_reachable_novelties":10}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    a=p.parse_args()
    records=generate_tests()
    report=audit_tests(records)
    fc02.write_jsonl(a.out,records)
    report["dataset_sha256"]=hashlib.sha256(a.out.read_bytes()).hexdigest()
    if a.lean:
        a.lean.parent.mkdir(parents=True,exist_ok=True)
        a.lean.write_text(fc02.lean_corpus(records),encoding="utf8")
        report["lean_sha256"]=hashlib.sha256(a.lean.read_bytes()).hexdigest()
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
