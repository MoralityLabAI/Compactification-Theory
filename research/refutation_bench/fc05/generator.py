"""FC-05 virgin, preregistered Horn families with both certificate polarities.

Protocol frozen in commit e6ab5716e3316441adad131b087a50d1ca0f2734
before the new generator, neural model training or any test inspection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from ..formal_curriculum import generator as fc02
from ..fc03 import generator as fc03
from ..fc04 import generator as fc04
from ..certificates import find_countermodel, verify_countermodel

SEED=20261113
FAMILIES=("split_relay","cross_gate","detour_join","layered_cycle")
SPLITS={"test_same":(192,2,4,81011),"test_deep":(192,6,8,92033)}
PREREG="e6ab5716e3316441adad131b087a50d1ca0f2734"


def _draw(rng:random.Random,family:str,depth:int,good:bool)->dict:
    n=depth+4
    h1,h2,guard=depth+1,depth+2,depth+3
    facts=[0]
    rules=[{"premises":[i-1],"head":i} for i in range(1,depth+1)]
    tail=rules[depth-1]
    if family=="split_relay":
        rules.extend([
            {"premises":[0],"head":h1},
            {"premises":[h1] if good else [h1,guard],"head":h2}
        ])
        tail["premises"]=sorted({depth-1,h1,h2})
    elif family=="cross_gate":
        rules.extend([
            {"premises":[max(0,depth-2)],"head":h1},
            {"premises":[max(0,depth-3),h1] if good else [guard,h1],"head":h2}
        ])
        tail["premises"]=sorted({depth-1,h2})
    elif family=="detour_join":
        rules.extend([
            {"premises":[0],"head":h1},
            {"premises":[h1],"head":h2},
            {"premises":[h2,max(0,depth-2)] if good else [h2,guard],
             "head":depth}
        ])
        if not good:
            tail["premises"]=sorted({depth-1,guard})
        elif rng.random()<.5:
            tail["premises"]=sorted({depth-1,guard})
    elif family=="layered_cycle":
        rules.extend([
            {"premises":[max(0,depth-2)] if good else [guard],"head":h1},
            {"premises":[h1],"head":h2},
            {"premises":[h2],"head":h1}
        ])
        tail["premises"]=sorted({depth-1,h2})
    else:
        raise ValueError(family)

    # Guard is syntactically present yet never derived. Additional decoy
    # rules are exclusively within the auxiliary/guard sector on negatives.
    rules.append({"premises":[guard],"head":guard})
    for _ in range(rng.randrange(2,5)):
        target=rng.choice([h1,h2])
        if good:
            options=[h1,h2,guard,rng.randrange(depth)]
        else:
            # A false conclusion can never be repaired by these decoys.
            options=[guard,h2] if target==h2 else [guard,h1,h2]
            if family=="detour_join" and target==h1:
                options=[guard,h1]
        premise=rng.choice(options)
        if rng.random()<.35:
            premises=sorted({premise,guard})
        else:
            premises=[premise]
        rules.append({"premises":premises,"head":target})
    if len(rules)>20 or n>12:
        raise AssertionError("fragment cap exceeded")

    names=[f"P{i}" for i in range(n)]
    permutation=list(range(n))
    rng.shuffle(permutation)
    alias={i:names[permutation[i]] for i in range(n)}
    item={
        "atoms":sorted(names),
        "facts":[alias[x] for x in facts],
        "goal":alias[depth],
        "rules":[{"premises":sorted(alias[p] for p in rule["premises"]),
                  "head":alias[rule["head"]]} for rule in rules]
    }
    rng.shuffle(item["rules"])
    return item


def _construct(seed:int,split:str,family:str,depth:int,good:bool,case_id:str)->dict:
    task=_draw(random.Random(seed),family,depth,good)
    p=fc02.formal_problem(task)
    closure=fc02.least_closure(task["facts"],task["rules"])
    actual=task["goal"] in closure
    if actual!=good:
        raise AssertionError(f"incorrect generated truth for {case_id}, {family}")
    other=find_countermodel(p["premises"],p["conclusion"])
    if (other is None)!=good:
        raise AssertionError("truth table contradicts closure")
    cert=None
    if good:
        fc02.proof_trace(task)
    else:
        values={a:(a in closure) for a in task["atoms"]}
        if not verify_countermodel(p["premises"],p["conclusion"],values)["verified"]:
            raise AssertionError("negative assignment fails independent verifier")
        cert={"kind":"countermodel","assignment":values}
    touched=set(task["facts"])|{task["goal"]}
    for rule in task["rules"]:
        touched.add(rule["head"])
        touched.update(rule["premises"])
    if touched!=set(task["atoms"]):
        raise AssertionError("unused proposition")
    return {**task,"case_id":case_id,"schema_version":1,"split":split,
            "family_id":family,"depth":depth,"formal":p,
            "label":"established" if good else "refuted",
            "reachable":sorted(closure),"countermodel":cert,
            "fingerprint":fc02._wl_fingerprint(task)}


def previous_records():
    return fc02.generate()+fc03.generate_tests()+fc04.generate_tests()


def generate_tests()->list[dict]:
    previous=previous_records()
    observed={x["fingerprint"] for x in previous}
    if len(observed)!=len(previous):
        raise AssertionError("previous source has WL overlaps")
    rows=[]
    for split,(count,dmin,dmax,offset) in SPLITS.items():
        accepted=Counter()
        attempts=0
        while sum(accepted.values())<count:
            attempts+=1
            if attempts>count*500:
                raise RuntimeError(f"unique graph exhaustion at {split}")
            fam=FAMILIES[sum(accepted.values())%len(FAMILIES)]
            k=accepted[fam]
            good=(k%2==0)
            depth=dmin+((k//2)%(dmax-dmin+1))
            seed=SEED*100_000+offset*10_000+attempts
            name=f"FC05_{split}_{len(rows):04d}"
            task=_construct(seed,split,fam,depth,good,name)
            if task["fingerprint"] in observed:
                continue
            observed.add(task["fingerprint"])
            accepted[fam]+=1
            rows.append(task)
        if any(accepted[x]!=count//len(FAMILIES) for x in FAMILIES):
            raise AssertionError("family count skew")
    audit(rows,previous)
    return rows


def audit(rows:list[dict],previous:list[dict]|None=None)->dict:
    if previous is None:
        previous=previous_records()
    before={x["fingerprint"] for x in previous}
    seen=set()
    tally=Counter()
    for task in rows:
        split,fam,verdict=task["split"],task["family_id"],task["label"]
        if split not in SPLITS or fam not in FAMILIES or verdict not in ("established","refuted"):
            raise AssertionError("unexpected case family")
        if not SPLITS[split][1]<=task["depth"]<=SPLITS[split][2]:
            raise AssertionError("out-of-scope depth")
        fp=fc02._wl_fingerprint(task)
        if fp in seen or fp in before or fp!=task["fingerprint"]:
            raise AssertionError("source family fingerprint overlap")
        seen.add(fp)
        tally[split,fam,verdict]+=1
        known=fc02.least_closure(task["facts"],task["rules"])
        if sorted(known)!=task["reachable"]:
            raise AssertionError("closure mismatch")
        formula=fc02.formal_problem(task)
        if formula!=task["formal"]:
            raise AssertionError("formula mismatch")
        other=find_countermodel(formula["premises"],formula["conclusion"])
        if task["label"]=="established":
            if other is not None or task["countermodel"] is not None:
                raise AssertionError("incorrect positive")
            fc02.proof_trace(task)
        else:
            if other is None or task["countermodel"] is None:
                raise AssertionError("missing negative certificate")
            if not verify_countermodel(formula["premises"],formula["conclusion"],
                task["countermodel"]["assignment"])["verified"]:
                raise AssertionError("bad negative certificate")
    for split,(count,_,__,___) in SPLITS.items():
        for fam in FAMILIES:
            for label in ("established","refuted"):
                if tally[split,fam,label]!=count//(2*len(FAMILIES)):
                    raise AssertionError(f"not balanced: {split} {fam} {label}")
    return {"protocol":PREREG,"count":len(rows),
            "splits":{s:sum(x["split"]==s for x in rows) for s in SPLITS},
            "positive_lean_proofs":sum(x["label"]=="established" for x in rows),
            "negative_checked_witnesses":sum(x["label"]=="refuted" for x in rows),
            "source_family_overlap":0,"schema_version":1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    rows=generate_tests()
    report=audit(rows)
    fc02.write_jsonl(args.out,rows)
    report["cases_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        report["lean_source_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
