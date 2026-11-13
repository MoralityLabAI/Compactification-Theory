"""FC-08 TRAIN/DEV-ONLY generator: conjunctive, shared-premise Horn proof search.

Preregistration committed as b6c8ffbef6606e89737bbfc32d73b8732b765e8d
before this module. The later test generator is a separate Git commit and
must be absent from the training-job checkout. Every label and minimum proof
is independently checked by the existing Horn and Boolean routines.
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
from ..fc07 import train_data as fc07
from ..fc07 import test_data as fc07_test
from ..fc06.posthoc_backward_slice import goal_ancestors
from ..fc07.posthoc_unary import unary_shortest_trace
from ..fc05 import proofs
from ..formal_curriculum import generator as fc02

SEED=20261113
RULE_BUDGET=5
ATOM_COUNT=12
GUARD=11
TRAIN_FAMILIES=("braided_and","shared_fork_and","cycle_and")
DEV_FAMILIES=("cross_and",)
SPLITS={
    "train":(768,3,4,101311,TRAIN_FAMILIES),
    "dev":(192,3,4,202627,DEV_FAMILIES),
}


def canonical_bytes(x:dict)->bytes:
    return fc02.canonical_bytes(x)


def canonical_rules(depth:int)->list[dict]:
    """A provable goal requires irreducible conjunctions and d new facts."""
    if depth==3:
        return [
            {"premises":[0],"head":1},
            {"premises":[0],"head":2},
            {"premises":[1,2],"head":3},
        ]
    if depth==4:
        return [
            {"premises":[0],"head":1},
            {"premises":[0],"head":2},
            {"premises":[1,2],"head":3},
            {"premises":[2,3],"head":4},
        ]
    if depth==5:
        return [
            {"premises":[0],"head":1},
            {"premises":[0],"head":2},
            {"premises":[1,2],"head":3},
            {"premises":[0],"head":4},
            {"premises":[3,4],"head":5},
        ]
    raise ValueError("FC-08 proof depth must be 3,4,5")


def train_decoy_parents(rng:random.Random,family:str,index:int,
                        decoys:list[int],depth:int)->list[int]:
    before=decoys[:index]
    spine=list(range(depth))
    if family=="braided_and":
        if index<2:
            return [rng.choice(spine)]
        if index%2==0:
            return sorted({before[-1],rng.choice(spine)})
        return [before[-2]]
    if family=="shared_fork_and":
        if index==0:
            return [0]
        if index==1:
            return [rng.choice(spine)]
        if index%3==0:
            return sorted({before[-1],before[0]})
        return [before[index//2]]
    if family=="cycle_and":
        if index<2 or index%3==0:
            return [rng.choice(spine)]
        if index%3==1:
            return [before[-1]]
        return sorted({before[-2],rng.choice(spine)})
    if family=="cross_and":
        if index<2:
            return [rng.choice(spine)]
        if index%2:
            return sorted({before[-1],before[0]})
        return sorted({before[-2],rng.choice(spine)})
    raise ValueError("test grammar is absent from training: "+family)


def construct(seed:int,depth:int,valid:bool,family:str,
              parent_fn)->dict:
    """Construct canonically, rename atoms, then permute all source rules."""
    rng=random.Random(seed)
    rules=canonical_rules(depth)
    decoys=list(range(depth+1,GUARD))
    for j,atom in enumerate(decoys):
        antecedents=parent_fn(rng,family,j,decoys,depth)
        rules.append({"premises":sorted(set(antecedents)),"head":atom})

    # Every decoy participates in an alternative producer of the real goal.
    # The alternatives have multiple antecedents, no true unary shortcut,
    # and at least the canonical minimum proof cost. Their conjunctions
    # can share useful premises or require separately derived subgoals.
    for j,atom in enumerate(decoys):
        if depth==3:
            anchor=rng.choice([1,2])
        elif depth==4:
            anchor=3
        else:
            anchor=3
        antecedents={anchor,atom}
        if rng.random()<.30:
            antecedents.add(rng.choice(list(range(1,depth))))
        rules.append({"premises":sorted(antecedents),"head":depth})

    # Guard is unreachable. Extra alternative decoy producers create
    # independent cycles and cross-links without repairing the goal.
    rules.append({"premises":[GUARD],"head":GUARD})
    spare=20-len(rules)
    for _ in range(min(spare,rng.randint(1,3))):
        target=rng.choice(decoys)
        prior=[x for x in decoys if x!=target]
        source=rng.choice([GUARD,*prior])
        antecedents={source}
        if rng.random()<.55:
            antecedents.add(rng.choice(list(range(depth))))
        alternative={"premises":sorted(antecedents),"head":target}
        if alternative not in rules:
            rules.append(alternative)

    if not valid:
        # Block every producer of the goal, preserving a complete
        # countermodel in which the unreachable guard is false.
        for rule in rules:
            if rule["head"]==depth:
                rule["premises"]=sorted(set(rule["premises"]+[GUARD]))

    if not all(len(r["premises"])>=2 for r in rules if r["head"]==depth):
        raise AssertionError("a unary goal shortcut survived construction")
    if len(rules)>20 or len(rules)<depth:
        raise AssertionError("source rule cap violated")

    names=[f"P{i}" for i in range(ATOM_COUNT)]
    perm=list(range(ATOM_COUNT))
    rng.shuffle(perm)
    alias={i:names[perm[i]] for i in range(ATOM_COUNT)}
    item={
      "atoms":sorted(names),
      "facts":[alias[0]],
      "goal":alias[depth],
      "rules":[{"premises":sorted(alias[x] for x in rule["premises"]),
                "head":alias[rule["head"]]} for rule in rules],
    }
    rng.shuffle(item["rules"])
    return item


def certify(raw:dict,split:str,family:str,depth:int,valid:bool,case_id:str)->dict:
    """Independent finite truth tables, exact shortest proof and structure."""
    record=fc06.certify(raw,split=split,family=family,depth=depth,
                        valid=valid,case_id=case_id)
    if len(raw["atoms"])!=ATOM_COUNT:
        raise AssertionError("not exactly 12 propositions")
    if any(len(rule["premises"])<2 for rule in raw["rules"]
           if rule["head"]==raw["goal"]):
        raise AssertionError("goal has a unary producer")
    if unary_shortest_trace(raw,budget=RULE_BUDGET):
        raise AssertionError("unexpected unary-only goal proof")
    ancestors=goal_ancestors(raw)
    if valid:
        if len(ancestors)<11:
            raise AssertionError("a reachable distractor stayed outside goal ancestry")
        if record["min_steps"]!=depth:
            raise AssertionError("shortest proof length changed")
        if len(record["reachable"])-len(record["facts"])!=10:
            raise AssertionError("not ten reachable novelty conclusions")
        if not proofs.verify_trace(raw,record["shortest_trace"],budget=RULE_BUDGET)["verified"]:
            raise AssertionError("positive reference trace failed independent replay")
    return record


@lru_cache(maxsize=1)
def previous_fingerprints()->frozenset[str]:
    prior=fc07.previous_fingerprints()
    train07=fc07.generate_train_dev()
    test07=fc07_test.generate_tests()
    items=list(prior)+[x["fingerprint"] for x in train07+test07]
    if len(set(items))!=len(items):
        raise AssertionError("earlier FC-02...FC-07 corpus has structural duplicates")
    return frozenset(items)


def generate_train_dev()->list[dict]:
    seen=set(previous_fingerprints())
    output=[]
    for split,(count,low,high,offset,families) in SPLITS.items():
        per=Counter()
        attempts=0
        while sum(per.values())<count:
            attempts+=1
            if attempts>count*950:
                raise RuntimeError("exhausted unique source graphs for "+split)
            family=families[len(output)%len(families)]
            index=per[family]
            valid=index%2==0
            depth=low+(index//2)%(high-low+1)
            seed=SEED*100000+offset*10000+attempts
            ident=f"FC08_{split}_{len(output):04d}"
            raw=construct(seed,depth,valid,family,train_decoy_parents)
            entry=certify(raw,split,family,depth,valid,ident)
            if entry["fingerprint"] in seen:
                continue
            seen.add(entry["fingerprint"])
            output.append(entry)
            per[family]+=1
        if any(per[family]!=count//len(families) for family in families):
            raise AssertionError("family count unbalanced")
    audit_train_dev(output)
    return output


def audit_train_dev(rows:list[dict],prior:set[str]|None=None)->dict:
    if prior is None:
        prior=set(previous_fingerprints())
    seen=set()
    groups=Counter()
    for entry in rows:
        split,family,label=entry["split"],entry["family_id"],entry["label"]
        if split not in SPLITS or family not in SPLITS[split][4]:
            raise AssertionError("unregistered train/dev family")
        fp=fc02._wl_fingerprint(entry)
        if fp!=entry["fingerprint"] or fp in prior or fp in seen:
            raise AssertionError("WL family collision or tampered fingerprint")
        seen.add(fp)
        graph={key:entry[key] for key in ("atoms","facts","rules","goal")}
        replay=certify(graph,split,family,entry["depth"],
                       label=="established",entry["case_id"])
        if entry!=replay:
            raise AssertionError("training annotation changed")
        groups[split,family,label]+=1
    for split,(count,_,__,___,families) in SPLITS.items():
        for family in families:
            for label in ("established","refuted"):
                if groups[split,family,label]!=count//(2*len(families)):
                    raise AssertionError("split distribution changed")
    return {"n":len(rows),"train":sum(x["split"]=="train" for x in rows),
            "dev":sum(x["split"]=="dev" for x in rows),
            "positive":sum(x["label"]=="established" for x in rows),
            "negative":sum(x["label"]=="refuted" for x in rows),
            "unique_fingerprints":len(seen),"past_overlap":0,
            "min_actions":[3,4],"rule_action_budget":RULE_BUDGET,
            "all_valid_goals_rely_on_conjunction":True,
            "all_valid_decoys_in_goal_ancestry":True,
            "positive_reachable_novelty":10}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    rows=generate_train_dev()
    receipt=audit_train_dev(rows)
    fc02.write_jsonl(args.out,rows)
    receipt["dataset_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        receipt["lean_source_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
