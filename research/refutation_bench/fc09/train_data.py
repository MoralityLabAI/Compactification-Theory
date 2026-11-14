"""FC-09 TRAIN/DEV-ONLY conjunctive Horn curriculum; later tests are absent.

Preregistration was frozen at 37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4
before generating cases, training a model, or evaluating test metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from ..fc08 import train_data as fc08
from ..fc08 import test_data as fc08_test
from ..fc06.posthoc_backward_slice import goal_ancestors
from ..fc07.posthoc_unary import unary_shortest_trace
from ..formal_curriculum import generator as fc02

ROOT_SEED=20261113
BUDGET=5
GUARD=11
TRAIN_FAMILIES=("convergent_and","crossed_dependencies","cyclic_shared")
DEV_FAMILIES=("nested_conjunction",)
SPLITS={
    "train":(768,3,4,506311,TRAIN_FAMILIES),
    "dev":(192,3,4,617533,DEV_FAMILIES),
}


def _parents(rng:random.Random,family:str,index:int,decoys:list[int],
             depth:int)->list[int]:
    previous=decoys[:index]
    spine=list(range(depth))
    if family=="convergent_and":
        if index<2:
            return [rng.choice(spine)]
        return sorted(set([previous[-1],rng.choice(spine)]))
    if family=="crossed_dependencies":
        if index<2:
            return [rng.choice(spine)]
        if index%2==0:
            return [previous[index//2-1]]
        return sorted(set([previous[-1],previous[0]]))
    if family=="cyclic_shared":
        if index==0 or index%3==0:
            return [rng.choice(spine)]
        if index%3==1:
            return [previous[-1]]
        return sorted(set([previous[-1],rng.choice(spine)]))
    if family=="nested_conjunction":
        if index<2:
            return [rng.choice(spine)]
        if index%2:
            return sorted(set([previous[-1],previous[0]]))
        return sorted(set([previous[-2],rng.choice(spine)]))
    raise ValueError("test family not authorized in train-only source: "+family)


def construct(seed:int,depth:int,valid:bool,family:str,parent_fn)->dict:
    if depth not in (3,4,5):
        raise ValueError("FC-09 canonical depth outside protocol")
    rng=random.Random(seed)
    rules=fc08.canonical_rules(depth)
    decoys=list(range(depth+1,GUARD))
    for i,atom in enumerate(decoys):
        parents=parent_fn(rng,family,i,decoys,depth)
        rules.append({"premises":sorted(set(parents)),"head":atom})

    # Alternative AND/OR routes use useful backbone atoms plus reachable
    # distractors. None can derive the goal by a unary-only path.
    for j,atom in enumerate(decoys):
        # Every alternative requires a goal-side critical prerequisite.
        # In depth 5, atom 4 alone is cheaply available from fact 0,
        # so require BOTH 3 and 4 plus the decoy to prevent a 3-step bypass.
        premises=({2,atom} if depth==3 else
                  {3,atom} if depth==4 else {3,4,atom})
        rules.append({"premises":sorted(premises),"head":depth})

    rules.append({"premises":[GUARD],"head":GUARD})
    room=20-len(rules)
    # Competing producers of INNER goal ancestors create shared subgoals
    # and cyclic shortcuts which appear promising but cost novelty actions.
    for step in range(room):
        target=rng.randint(2,depth-1)
        decoy=rng.choice(decoys)
        premises={target-1,decoy}
        if family in ("cyclic_shared","nested_conjunction") and step%2==0:
            premises.add(rng.choice(list(range(target))))
        rule={"premises":sorted(premises),"head":target}
        if rule not in rules:
            rules.append(rule)

    if not valid:
        # Guard every possible GOAL producer. The independent truth-table
        # verifier must still confirm the claimed failure.
        for rule in rules:
            if rule["head"]==depth:
                rule["premises"]=sorted(set(rule["premises"]+[GUARD]))

    if len(rules)>20:
        raise AssertionError("too many Horn rules")
    if not all(len(r["premises"])>=2 for r in rules if r["head"]==depth):
        raise AssertionError("unary goal producer survived")

    names=[f"P{i}" for i in range(12)]
    permutation=list(range(12))
    rng.shuffle(permutation)
    alias={i:names[permutation[i]] for i in range(12)}
    graph={
      "atoms":sorted(names),
      "facts":[alias[0]],
      "goal":alias[depth],
      "rules":[{"premises":sorted(alias[x] for x in r["premises"]),
                "head":alias[r["head"]]} for r in rules],
    }
    rng.shuffle(graph["rules"])
    return graph


def certify(raw:dict,split:str,family:str,depth:int,valid:bool,case_id:str)->dict:
    record=fc08.certify(raw,split,family,depth,valid,case_id)
    if not all(len(rule["premises"])>=2 for rule in raw["rules"]
               if rule["head"]==raw["goal"]):
        raise AssertionError("FC-09 unary goal shortcut")
    if unary_shortest_trace(raw,budget=BUDGET):
        raise AssertionError("a unary-only goal derivation exists")
    if valid:
        if record["min_steps"]!=depth:
            raise AssertionError("minimum proof length changed")
        if len(record["reachable"])-len(record["facts"])!=10:
            raise AssertionError("positive closure is not ten novel conclusions")
        if len(goal_ancestors(record))<11:
            raise AssertionError("decoy graph disappeared from goal ancestry")
    return record


@lru_cache(maxsize=1)
def previous_fingerprints()->frozenset[str]:
    old=set(fc08.previous_fingerprints())
    extra=fc08.generate_train_dev()+fc08_test.generate_tests()
    ids=[x["fingerprint"] for x in extra]
    if len(set(ids))!=len(ids) or old.intersection(ids):
        raise AssertionError("FC-08 prior data has duplicate structural fingerprints")
    return frozenset(old.union(ids))


def generate_train_dev()->list[dict]:
    seen=set(previous_fingerprints())
    rows=[]
    for split,(total,low,high,offset,families) in SPLITS.items():
        count=Counter()
        attempts=0
        while sum(count.values())<total:
            attempts+=1
            if attempts>total*950:
                raise RuntimeError("failed to sample enough unique "+split)
            family=families[len(rows)%len(families)]
            k=count[family]
            valid=k%2==0
            depth=low+(k//2)%(high-low+1)
            seed=ROOT_SEED*100000+offset*10000+attempts
            identifier=f"FC09_{split}_{len(rows):04d}"
            sample=certify(construct(seed,depth,valid,family,_parents),
                           split,family,depth,valid,identifier)
            if sample["fingerprint"] in seen:
                continue
            seen.add(sample["fingerprint"])
            rows.append(sample)
            count[family]+=1
        if any(count[f]!=total//len(families) for f in families):
            raise AssertionError("train/dev family distribution changed")
    audit_train_dev(rows)
    return rows


def audit_train_dev(rows:list[dict],prior:set[str]|None=None)->dict:
    if prior is None:
        prior=set(previous_fingerprints())
    fingerprints=set()
    groups=Counter()
    for task in rows:
        split,family,label=task["split"],task["family_id"],task["label"]
        if split not in SPLITS or family not in SPLITS[split][4]:
            raise AssertionError("unregistered family")
        fingerprint=fc02._wl_fingerprint(task)
        if fingerprint!=task["fingerprint"] or fingerprint in prior or fingerprint in fingerprints:
            raise AssertionError("fingerprint overlap or tampering")
        fingerprints.add(fingerprint)
        raw={k:task[k] for k in ("atoms","facts","rules","goal")}
        replay=certify(raw,split,family,task["depth"],label=="established",task["case_id"])
        if replay!=task:
            raise AssertionError("source record tampered")
        groups[split,family,label]+=1
    for split,(n,_,__,___,families) in SPLITS.items():
        for family in families:
            for label in ("established","refuted"):
                if groups[split,family,label]!=n//(2*len(families)):
                    raise AssertionError("label balance violated")
    return {
      "schema_version":1,
      "n":len(rows),
      "train_count":sum(x["split"]=="train" for x in rows),
      "dev_count":sum(x["split"]=="dev" for x in rows),
      "positives":sum(x["label"]=="established" for x in rows),
      "negatives":sum(x["label"]=="refuted" for x in rows),
      "distinct_structural_fingerprints":len(fingerprints),
      "prior_fingerprint_overlap":0,
      "five_action_cap":BUDGET,
      "every_valid_goal_requires_conjunction":True,
      "no_unary_goal_paths":True,
      "positive_reachable_novelty":10,
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    data=generate_train_dev()
    receipt=audit_train_dev(data)
    fc02.write_jsonl(args.out,data)
    receipt["dataset_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(data),encoding="utf8")
        receipt["lean_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
