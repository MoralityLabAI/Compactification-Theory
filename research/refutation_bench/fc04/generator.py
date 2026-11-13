"""FC-04 preregistered, exactly checked, entirely new structural test grammar.

The preregistration was committed as ce7190129617d8387a7acf4ceed3e8d0570cfce4
before implementation or test data inspection. This module is test-only.
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
from ..certificates import find_countermodel, verify_countermodel

SEED=20261113
FAMILIES=("relay_deadlock","asymmetric_fork","stacked_join","gated_detour")
SPLITS={"test_same":(192,2,4,62003),"test_deep":(192,6,8,73009)}
PRE_REGISTRATION="ce7190129617d8387a7acf4ceed3e8d0570cfce4"


def _construct(rng:random.Random,family:str,depth:int,positive:bool)->dict:
    n=depth+4
    h1,h2,guard=depth+1,depth+2,depth+3
    facts=[0]
    rules=[{"premises":[i-1],"head":i} for i in range(1,depth+1)]
    if family=="relay_deadlock":
        rules.append({"premises":[0] if positive else [guard],"head":h1})
        rules.append({"premises":[h1],"head":h2})
        rules[depth-1]["premises"]=sorted([depth-1,h2])
    elif family=="asymmetric_fork":
        rules.append({"premises":[0],"head":h1})
        rules.append({"premises":[h1,max(0,depth-2)] + ([] if positive else [guard]),"head":h2})
        rules[depth-1]["premises"]=[depth-1,h2]
    elif family=="stacked_join":
        rules.append({"premises":[1] if positive else [1,guard],"head":h1})
        rules.append({"premises":[h1,max(0,depth-2)],"head":h2})
        rules[depth-1]["premises"]=[depth-1,h2]
    elif family=="gated_detour":
        rules.append({"premises":[0],"head":h1})
        rules.append({"premises":[h1],"head":h2})
        rules.append({"premises":[h2] if positive else [h2,guard],"head":depth})
        if not positive:
            rules[depth-1]["premises"]=[depth-1,guard]
        elif rng.random()<.5:
            # One route can be blocked, while the other remains complete.
            rules[depth-1]["premises"]=[depth-1,guard]
    else:
        raise ValueError("unknown FC-04 family")

    # The guard has no external path; cycles only offer illusory support.
    rules.append({"premises":[guard],"head":guard})
    rules.append({"premises":[h2],"head":h2})
    for _ in range(rng.randint(1,4)):
        target=rng.choice([h1,h2])
        if positive:
            source=rng.choice([guard,h1,h2,rng.randrange(depth)])
        elif family=="asymmetric_fork" and target==h1:
            source=rng.choice([h1,guard])
        elif family=="asymmetric_fork" and target==h2:
            source=rng.choice([h2,guard])
        elif family=="stacked_join":
            source=rng.choice([guard,h2])
        elif family=="relay_deadlock":
            source=rng.choice([guard,h1,h2])
        else:
            source=rng.choice([guard,h1,h2,rng.randrange(depth)])
        ps=[source]
        if rng.random()<.4:
            ps.append(rng.choice([h1,h2,guard]))
        rules.append({"premises":sorted(set(ps)),"head":target})
    assert n<=12 and len(rules)<=20

    names=[f"P{i}" for i in range(n)]
    perm=list(range(n));rng.shuffle(perm)
    alias={i:names[perm[i]] for i in range(n)}
    data={"atoms":sorted(names),"facts":[alias[f] for f in facts],"goal":alias[depth],
          "rules":[{"premises":sorted(alias[x] for x in r["premises"]),
                    "head":alias[r["head"]]} for r in rules]}
    rng.shuffle(data["rules"])
    return data


def _annotate(seed:int,split:str,family:str,depth:int,valid:bool,case_id:str)->dict:
    raw=_construct(random.Random(seed),family,depth,valid)
    formula=fc02.formal_problem(raw)
    closure=fc02.least_closure(raw["facts"],raw["rules"])
    actual=(raw["goal"] in closure)
    if actual!=valid:
        raise AssertionError(f"construction unexpectedly changed verdict: {case_id} {family}")
    alternate=find_countermodel(formula["premises"],formula["conclusion"])
    if (alternate is None)!=valid:
        raise AssertionError("truth tables and forward chaining disagree")
    cert=None
    if valid:
        fc02.proof_trace(raw)
    else:
        values={atom:(atom in closure) for atom in raw["atoms"]}
        checked=verify_countermodel(formula["premises"],formula["conclusion"],values)
        if not checked["verified"]:
            raise AssertionError("bad least-model witness")
        cert={"kind":"countermodel","assignment":values}
    referenced=set(raw["facts"]) | {raw["goal"]}
    for rule in raw["rules"]:
        referenced.add(rule["head"])
        referenced.update(rule["premises"])
    if referenced!=set(raw["atoms"]):
        raise AssertionError("unreferenced atom")
    return {**raw,"case_id":case_id,"schema_version":1,"split":split,
            "family_id":family,"depth":depth,"formal":formula,
            "label":"established" if valid else "refuted",
            "reachable":sorted(closure),"countermodel":cert,
            "fingerprint":fc02._wl_fingerprint(raw)}


def generate_tests()->list[dict]:
    old=fc02.generate()+fc03.generate_tests()
    seen={x["fingerprint"] for x in old}
    if len(seen)!=len(old):
        raise AssertionError("previous corpus duplicates")
    rows=[]
    for split,(total,dmin,dmax,offset) in SPLITS.items():
        accepted=Counter()
        attempts=0
        while sum(accepted.values())<total:
            attempts+=1
            if attempts>total*500:
                raise RuntimeError(f"too few unique {split} graphs")
            family=FAMILIES[sum(accepted.values())%len(FAMILIES)]
            k=accepted[family]
            valid=(k%2==0)
            depth=dmin+(k//2)%(dmax-dmin+1)
            random_seed=SEED*100000+offset*10000+attempts
            case_id=f"FC04_{split}_{len(rows):04d}"
            task=_annotate(random_seed,split,family,depth,valid,case_id)
            if task["fingerprint"] in seen:
                continue
            seen.add(task["fingerprint"])
            rows.append(task)
            accepted[family]+=1
        if any(accepted[f]!=total//len(FAMILIES) for f in FAMILIES):
            raise AssertionError("unbalanced family")
    audit_tests(rows,old)
    return rows


def audit_tests(rows:list[dict],older:list[dict]|None=None)->dict:
    if older is None:
        older=fc02.generate()+fc03.generate_tests()
    old_fp={x["fingerprint"] for x in older}
    seen=set()
    counts=Counter()
    for x in rows:
        split=x["split"];fam=x["family_id"];label=x["label"]
        if split not in SPLITS or fam not in FAMILIES or label not in ("established","refuted"):
            raise AssertionError("unexpected family, split or verdict")
        lo,hi=SPLITS[split][1:3]
        if not lo<=x["depth"]<=hi:
            raise AssertionError("depth outside registered span")
        fp=fc02._wl_fingerprint(x)
        if fp!=x["fingerprint"] or fp in seen or fp in old_fp:
            raise AssertionError("repeated structure")
        seen.add(fp)
        counts[(split,fam,label)]+=1
        known=fc02.least_closure(x["facts"],x["rules"])
        if sorted(known)!=x["reachable"]:
            raise AssertionError("closure changed")
        p=fc02.formal_problem(x)
        if p!=x["formal"]:
            raise AssertionError("formula changed")
        other=find_countermodel(p["premises"],p["conclusion"])
        if x["label"]=="established":
            if other is not None or x["countermodel"] is not None:
                raise AssertionError("positive has counterexample")
            fc02.proof_trace(x)
        else:
            if other is None or x["countermodel"] is None:
                raise AssertionError("negative missing witness")
            if not verify_countermodel(p["premises"],p["conclusion"],
                                      x["countermodel"]["assignment"])["verified"]:
                raise AssertionError("unverified source witness")
    for split,(total,_,__,___) in SPLITS.items():
        for family in FAMILIES:
            for label in ("established","refuted"):
                if counts[split,family,label]!=total//(2*len(FAMILIES)):
                    raise AssertionError("unbalanced split")
    return {"protocol":PRE_REGISTRATION,
            "count":len(rows),
            "split_counts":{s:sum(1 for x in rows if x["split"]==s) for s in SPLITS},
            "source_family_count":len(FAMILIES),
            "positive_lean_theorems":sum(x["label"]=="established" for x in rows),
            "negative_checked_countermodels":sum(x["label"]=="refuted" for x in rows),
            "structural_overlap_count":0,"schema_version":1}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    a=p.parse_args()
    rows=generate_tests()
    result=audit_tests(rows)
    fc02.write_jsonl(a.out,rows)
    result["cases_sha256"]=hashlib.sha256(a.out.read_bytes()).hexdigest()
    if a.lean:
        a.lean.parent.mkdir(parents=True,exist_ok=True)
        a.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        result["lean_source_sha256"]=hashlib.sha256(a.lean.read_bytes()).hexdigest()
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
