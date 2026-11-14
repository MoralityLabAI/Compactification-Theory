"""FC-06 training-only source: strict five-action proof-choice with distractors.

This module contains only training/validation grammar. FC-06 unseen test
families live in a distinct, later-committed module. The training CI checks
out the source at the commit before any test grammar was added.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from ..formal_curriculum import generator as fc02
from ..certificates import find_countermodel, verify_countermodel
from ..fc05 import proofs

ROOT_SEED = 20261113
ACTION_BUDGET = 5
ATOM_COUNT = 12
GUARD = 11
TRAIN_FAMILIES = ("star_decoys", "ladder_decoys", "fork_decoys")
DEV_FAMILIES = ("cross_decoys",)
TRAIN_SPECS = {"train":(768,2,4,11203,TRAIN_FAMILIES),
               "dev":(192,2,4,22411,DEV_FAMILIES)}


def canonical_bytes(x):
    return fc02.canonical_bytes(x)


def shortest_trace(task:dict, budget:int=ACTION_BUDGET)->list[int]|None:
    """Independent minimum-cardinality BFS over monotone Horn proof states."""
    known0=frozenset(task["facts"])
    queue=deque([(known0,())])
    visited={known0}
    while queue:
        known,trace=queue.popleft()
        if task["goal"] in known:
            return list(trace)
        if len(trace)>=budget:
            continue
        for i,rule in enumerate(task["rules"]):
            if rule["head"] in known or not set(rule["premises"])<=known:
                continue
            new=known | frozenset([rule["head"]])
            if new not in visited:
                visited.add(new)
                queue.append((new,trace+(i,)))
    return None


def _decoy_parents(rng:random.Random,family:str,
                   index:int,decoys:list[int],depth:int)->list[int]:
    """All chosen parents precede the decoy in a constructive topological order.

    Backbone dependencies may include prior to the goal but never the guard
    or the goal itself, so all distractors are reachable on positive tasks.
    """
    prior=decoys[:index]
    spine=list(range(depth))
    if family=="star_decoys":
        return [0]
    if family=="ladder_decoys":
        return [prior[-1]] if prior else [0]
    if family=="fork_decoys":
        if index<2:
            return [0]
        return [prior[index%2]] if index%3 else [rng.choice(spine)]
    if family=="cross_decoys":
        if index==0:
            return [0]
        if index==1:
            return [rng.choice(spine)]
        if index%2==0:
            return sorted(set([prior[-2],rng.choice(spine)]))
        return [prior[-1]]
    raise ValueError("test grammar unavailable in training source: "+family)


def _make(seed:int,depth:int,valid:bool,family:str)->dict:
    if depth not in (2,3,4):
        raise ValueError("train depth outside protocol")
    rng=random.Random(seed)
    decoys=list(range(depth+1,GUARD))
    rules=[{"premises":[i-1],"head":i} for i in range(1,depth+1)]
    if not valid:
        broken=rng.randrange(depth)
        rules[broken]["premises"]=sorted(set(rules[broken]["premises"]+[GUARD]))

    for i,atom in enumerate(decoys):
        pre=_decoy_parents(rng,family,i,decoys,depth)
        rules.append({"premises":sorted(set(pre)),"head":atom})

    # Source-extra rule graph motifs vary within family. Each extra rule
    # adds a *second* derivation of an already reachable distractor;
    # it cannot repair a blocked goal chain. Cap so <= 20 rules.
    extras=min(len(decoys),rng.randint(3,7))
    for _ in range(extras):
        j=rng.randrange(len(decoys))
        atom=decoys[j]
        prior=decoys[:j]
        choice=rng.choice([0,*list(range(depth)),*prior])
        pre=[choice]
        if prior and rng.random()<.4:
            pre.append(rng.choice(prior))
        candidate={"premises":sorted(set(pre)),"head":atom}
        if candidate not in rules:
            rules.append(candidate)

    rules.append({"premises":[GUARD],"head":GUARD})
    if len(rules)>20:
        raise AssertionError("too many rules")
    # Only graph topology, not proposition labels, encodes dependencies.
    names=[f"P{i}" for i in range(ATOM_COUNT)]
    permutation=list(range(ATOM_COUNT))
    rng.shuffle(permutation)
    alias={i:names[permutation[i]] for i in range(ATOM_COUNT)}
    result={
        "atoms":sorted(names),
        "facts":[alias[0]],
        "goal":alias[depth],
        "rules":[{"premises":sorted(alias[x] for x in rule["premises"]),
                  "head":alias[rule["head"]]} for rule in rules],
    }
    rng.shuffle(result["rules"])
    return result


def certify(raw:dict,*,split:str,family:str,depth:int,
            valid:bool,case_id:str)->dict:
    """Two independent exact decision procedures and a bounded positive trace."""
    if len(raw["atoms"])!=ATOM_COUNT or len(raw["rules"])>20:
        raise AssertionError("shape outside prereg")
    p=fc02.formal_problem(raw)
    closure=fc02.least_closure(raw["facts"],raw["rules"])
    truth=(raw["goal"] in closure)
    if truth!=valid:
        raise AssertionError("constructed task has wrong entailment label")
    truth_table=find_countermodel(p["premises"],p["conclusion"])
    if (truth_table is None)!=valid:
        raise AssertionError("Boolean checker and forward closure differ")
    minimum=shortest_trace(raw)
    certificate=None
    if valid:
        if minimum is None or len(minimum)!=depth:
            raise AssertionError("positive minimum proof length changed")
        if not proofs.verify_trace(raw,minimum,budget=ACTION_BUDGET)["verified"]:
            raise AssertionError("minimal positive trace was not certified")
        # All 12 atoms except guard are provable; one is a given fact.
        if len(closure)!=11 or len(closure-set(raw["facts"]))!=10:
            raise AssertionError("positive decoy saturation violated")
        if not len(minimum)<=ACTION_BUDGET<10:
            raise AssertionError("proof budget is not binding")
    else:
        if minimum is not None:
            raise AssertionError("a negative task unexpectedly has a proof")
        assignment={a:(a in closure) for a in raw["atoms"]}
        if not verify_countermodel(p["premises"],p["conclusion"],
                                   assignment)["verified"]:
            raise AssertionError("invalid negative countermodel")
        certificate={"kind":"countermodel","assignment":assignment}
    return {
        **raw,"case_id":case_id,"family_id":family,"split":split,
        "depth":depth,"schema_version":1,
        "formal":p,"label":"established" if valid else "refuted",
        "reachable":sorted(closure),"min_steps":len(minimum) if minimum is not None else None,
        "shortest_trace":minimum,"countermodel":certificate,
        "fingerprint":fc02._wl_fingerprint(raw),
    }


@lru_cache(maxsize=1)
def older_fingerprints()->frozenset[str]:
    from ..fc03 import generator as fc03
    from ..fc04 import generator as fc04
    from ..fc05 import generator as fc05
    prior=fc02.generate()+fc03.generate_tests()+fc04.generate_tests()+fc05.generate_tests()
    items=[x["fingerprint"] for x in prior]
    if len(set(items))!=len(items):
        raise AssertionError("duplicate older dataset")
    return frozenset(items)


def generate_train_dev()->list[dict]:
    seen=set(older_fingerprints())
    records=[]
    for split,(count,depth_min,depth_max,offset,families) in TRAIN_SPECS.items():
        per_family=Counter()
        attempts=0
        while sum(per_family.values())<count:
            attempts+=1
            if attempts>count*600:
                raise RuntimeError("failed to sample unique "+split)
            family=families[len(records)%len(families)]
            index=per_family[family]
            valid=(index%2==0)
            depth=depth_min+(index//2)%(depth_max-depth_min+1)
            seed=ROOT_SEED*100000+offset*10000+attempts
            ident=f"FC06_{split}_{len(records):04d}"
            raw=_make(seed,depth,valid,family)
            sample=certify(raw,split=split,family=family,depth=depth,valid=valid,case_id=ident)
            if sample["fingerprint"] in seen:
                continue
            seen.add(sample["fingerprint"])
            records.append(sample)
            per_family[family]+=1
        if any(per_family[x]!=count//len(families) for x in families):
            raise AssertionError("family balance violated")
    audit_train_dev(records,prior=set(older_fingerprints()))
    return records


def audit_train_dev(records:list[dict],prior:set[str]|None=None)->dict:
    if prior is None:
        prior=set(older_fingerprints())
    seen=set()
    counts=Counter()
    for x in records:
        split,family,label=x["split"],x["family_id"],x["label"]
        if split not in TRAIN_SPECS or family not in TRAIN_SPECS[split][4]:
            raise AssertionError("unregistered split or family")
        if x["fingerprint"] in seen or x["fingerprint"] in prior:
            raise AssertionError("duplicate structural graph")
        seen.add(x["fingerprint"])
        original={k:x[k] for k in ("atoms","facts","rules","goal")}
        replay=certify(original,split=split,family=family,depth=x["depth"],
                       valid=(label=="established"),case_id=x["case_id"])
        if replay!=x:
            raise AssertionError("source annotation modified")
        counts[split,family,label]+=1
    for split,(count,_,__,___,families) in TRAIN_SPECS.items():
        for family in families:
            for label in ("established","refuted"):
                if counts[split,family,label]!=count//(2*len(families)):
                    raise AssertionError("count/label imbalance")
    return {"schema_version":1,"records":len(records),
            "train_count":sum(x["split"]=="train" for x in records),
            "dev_count":sum(x["split"]=="dev" for x in records),
            "positive":sum(x["label"]=="established" for x in records),
            "negative":sum(x["label"]=="refuted" for x in records),
            "distinct_WL_graphs":len(seen),"prior_WL_overlap":0,
            "budget":ACTION_BUDGET,"positive_novel_closure":10}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--lean",type=Path)
    a=p.parse_args()
    rows=generate_train_dev()
    receipt=audit_train_dev(rows)
    fc02.write_jsonl(a.out,rows)
    receipt["dataset_sha256"]=hashlib.sha256(a.out.read_bytes()).hexdigest()
    if a.lean:
        a.lean.parent.mkdir(parents=True,exist_ok=True)
        a.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        receipt["lean_sha256"]=hashlib.sha256(a.lean.read_bytes()).hexdigest()
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
