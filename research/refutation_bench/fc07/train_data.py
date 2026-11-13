"""FC-07 train/dev ONLY: alternative goal producers and seductive decoy ancestry.

The preregistration was committed before this file as
d3216371ae480b7e1b42cb913a33847ed5ad47c2. This source must be pinned
before any FC-07 test generator is added. Training may never open tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from ..fc06 import train_data as fc06
from ..fc06 import test_data as fc06_test
from ..formal_curriculum import generator as fc02
from ..fc06.posthoc_backward_slice import goal_ancestors

ROOT_SEED=20261113
ACTION_BUDGET=5
TRAIN_FAMILIES=("braided_ancestors","shared_subgoals","cycle_traps")
DEV_FAMILIES=("gated_forks",)
TRAIN_SPECS={
    "train":(768,2,4,53107,TRAIN_FAMILIES),
    "dev":(192,2,4,64211,DEV_FAMILIES),
}
GUARD=11


def _decoy_input(rng:random.Random,family:str,i:int,
                 decoys:list[int],depth:int)->list[int]:
    earlier=decoys[:i]
    spine=list(range(depth))
    if family=="braided_ancestors":
        if i==0:
            return [rng.choice(spine)]
        if i%2:
            return sorted(set([earlier[-1],rng.choice(spine)]))
        return [earlier[-1]]
    if family=="shared_subgoals":
        if i==0:
            return [0]
        if i<3:
            return sorted(set([decoys[0],rng.choice(spine)]))
        return sorted(set([earlier[-1],decoys[0]]))
    if family=="cycle_traps":
        if i==0 or i%3==0:
            return [rng.choice(spine)]
        if i%3==1:
            return [earlier[-1]]
        return sorted(set([earlier[-1],rng.choice(spine)]))
    if family=="gated_forks":
        if i==0:
            return [0]
        if i==1:
            return [rng.choice(spine)]
        return sorted(set([earlier[-1],earlier[max(0,i//2-1)]]))
    raise ValueError("train grammar absent: "+family)


def _core(seed:int,depth:int,valid:bool,family:str)->dict:
    if not 2<=depth<=5:
        raise ValueError("invalid FC-07 backbone depth")
    rng=random.Random(seed)
    decoys=list(range(depth+1,GUARD))
    rules=[{"premises":[i-1],"head":i} for i in range(1,depth+1)]
    # Every distractor is constructively reachable, but the shortest
    # backbone does not need to derive any of these nodes.
    for j,atom in enumerate(decoys):
        rules.append({"premises":_decoy_input(rng,family,j,decoys,depth),
                      "head":atom})

    # Decoy heads feed alternatives into the *real goal ancestry*. Every
    # decoy now lies inside the naive backward syntactic slice, although
    # the direct backbone is a shorter proof.
    for j,atom in enumerate(decoys):
        target=1+rng.randrange(depth)
        rules.append({"premises":sorted([target-1,atom]),"head":target})

    # Guard is unreachable. Backedges and shared decoy branches present
    # plausible but unnecessary alternatives, with bounded rule count.
    rules.append({"premises":[GUARD],"head":GUARD})
    room=20-len(rules)
    extras=min(room,1+rng.randrange(3))
    for _ in range(extras):
        target=rng.choice(decoys)
        source=rng.choice([*decoys,GUARD])
        parent=[source]
        if rng.random()<.42:
            parent.append(rng.choice(list(range(depth))))
        rules.append({"premises":sorted(set(parent)),"head":target})

    if not valid:
        # All producers of a chosen goal-backbone atom are now guarded.
        blocked=1+rng.randrange(depth)
        for rule in rules:
            if rule["head"]==blocked:
                rule["premises"]=sorted(set(rule["premises"]+[GUARD]))
    if len(rules)>20:
        raise AssertionError("rule count exceeded 20")

    names=[f"P{i}" for i in range(12)]
    permutation=list(range(12))
    rng.shuffle(permutation)
    alias={i:names[permutation[i]] for i in range(12)}
    out={
        "atoms":sorted(names),
        "facts":[alias[0]],
        "goal":alias[depth],
        "rules":[{"premises":sorted(alias[x] for x in rule["premises"]),
                  "head":alias[rule["head"]]} for rule in rules],
    }
    rng.shuffle(out["rules"])
    return out


def certify(raw:dict,split:str,family:str,depth:int,valid:bool,
            case_id:str)->dict:
    """Re-use independent exact truth-table, BFS and Lean-source contracts."""
    task=fc06.certify(raw,split=split,family=family,depth=depth,
                       valid=valid,case_id=case_id)
    ancestors=goal_ancestors(task)
    if not set(task["reachable"])<=ancestors:
        # Negative cases may have unreachable spine tails, while their
        # decoys should still be goal-relevant; only the true closure is
        # subject to the strict test below.
        if valid:
            raise AssertionError("reachable distractor fell outside ancestry")
    if valid:
        if task["min_steps"]!=depth or len(task["reachable"])-len(task["facts"])!=10:
            raise AssertionError("strict proof budget or novel count shifted")
        if not set(task["atoms"])-set(task["facts"])-{task["goal"]}:
            raise AssertionError("invalid atom structure")
        if len(ancestors)<11:
            raise AssertionError("goal ancestry lacks reachable decoys")
    return task


@lru_cache(maxsize=1)
def previous_fingerprints()->frozenset[str]:
    prior=fc02.generate()
    from ..fc03 import generator as fc03
    from ..fc04 import generator as fc04
    from ..fc05 import generator as fc05
    prior+=fc03.generate_tests()+fc04.generate_tests()+fc05.generate_tests()
    prior+=fc06.generate_train_dev()+fc06_test.generate_tests()
    fingerprints=[x["fingerprint"] for x in prior]
    if len(set(fingerprints))!=len(fingerprints):
        raise AssertionError("previous source contains repeated WL fingerprints")
    return frozenset(fingerprints)


def generate_train_dev()->list[dict]:
    seen=set(previous_fingerprints())
    result=[]
    for split,(count,lo,hi,offset,families) in TRAIN_SPECS.items():
        counts=Counter()
        attempts=0
        while sum(counts.values())<count:
            attempts+=1
            if attempts>count*800:
                raise RuntimeError("insufficient distinct graphs for "+split)
            family=families[len(result)%len(families)]
            k=counts[family]
            valid=(k%2==0)
            depth=lo+((k//2)%(hi-lo+1))
            seed=ROOT_SEED*100000+offset*10000+attempts
            case_id=f"FC07_{split}_{len(result):04d}"
            raw=_core(seed,depth,valid,family)
            record=certify(raw,split,family,depth,valid,case_id)
            if record["fingerprint"] in seen:
                continue
            seen.add(record["fingerprint"])
            result.append(record)
            counts[family]+=1
        if any(counts[f]!=count//len(families) for f in families):
            raise AssertionError("train/dev family skew")
    audit_train_dev(result)
    return result


def audit_train_dev(records:list[dict])->dict:
    old=set(previous_fingerprints())
    unique=set()
    balance=Counter()
    for record in records:
        split=record["split"]
        fam=record["family_id"]
        if split not in TRAIN_SPECS or fam not in TRAIN_SPECS[split][4]:
            raise AssertionError("unregistered train family")
        if record["fingerprint"] in unique or record["fingerprint"] in old:
            raise AssertionError("WL overlap")
        unique.add(record["fingerprint"])
        raw={k:record[k] for k in ("atoms","facts","rules","goal")}
        expected=certify(raw,split,fam,record["depth"],
                         record["label"]=="established",record["case_id"])
        if expected!=record:
            raise AssertionError("annotation tampered")
        balance[split,fam,record["label"]]+=1
    for split,(count,_,__,___,families) in TRAIN_SPECS.items():
        for fam in families:
            for label in ("established","refuted"):
                if balance[split,fam,label]!=count//(2*len(families)):
                    raise AssertionError("balance changed")
    return {
      "schema_version":1,"count":len(records),
      "train_count":sum(r["split"]=="train" for r in records),
      "dev_count":sum(r["split"]=="dev" for r in records),
      "positive":sum(r["label"]=="established" for r in records),
      "negative":sum(r["label"]=="refuted" for r in records),
      "novel_atoms_on_valid":10,
      "proof_action_budget":ACTION_BUDGET,
      "fingerprints_distinct":len(unique),"previous_overlap":0,
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    a=p.parse_args()
    records=generate_train_dev()
    info=audit_train_dev(records)
    fc02.write_jsonl(a.out,records)
    info["dataset_sha256"]=hashlib.sha256(a.out.read_bytes()).hexdigest()
    if a.lean:
        a.lean.parent.mkdir(parents=True,exist_ok=True)
        a.lean.write_text(fc02.lean_corpus(records),encoding="utf8")
        info["lean_sha256"]=hashlib.sha256(a.lean.read_bytes()).hexdigest()
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(info,sort_keys=True,indent=2)+"\n")
    print(json.dumps(info,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
