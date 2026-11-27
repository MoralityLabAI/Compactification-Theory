"""Oracle-owned, synthetic root-action curriculum for research prioritization.

No provider output, model answers, private roots, or training admission. The
public inputs and referee labels are written separately. Values are conditional
on the finite model; optimal continuation is an evaluator resource.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
from .core import digest, require, write_json
from .research_cases import compile_research, fraction
from .research_oracle import adaptive_oracle, replay_policy
from .registry import load_registry

def as_fraction(v):
    return fraction(v.numerator,v.denominator)

def condition(d, indices, budget, skip):
    return {"hypotheses":[d["hypotheses"][i] for i in indices],
            "tests":[{"id":t["id"],"cost":t["cost"],"bits":[t["bits"][i] for i in indices]}
                     for t in d["tests"] if t["id"] not in skip],
            "budget":budget,"correct_reward":d["correct_reward"],"wrong_reward":d["wrong_reward"]}

def stop_policy(d):
    candidates=[(replay_policy(d,{"decision":k}),k) for k in ("A","B","defer")]
    return max(candidates,key=lambda row:(row[0],row[1]))

def options(d):
    h=d["hypotheses"]
    total=sum(x["weight"] for x in h)
    result=[]
    for k in ("A","B","defer"):
        p={"decision":k}
        v=replay_policy(d,p)
        result.append({"id":"stop:"+k,"kind":"stop","cost":0,"outcome_masses":[total],
                       "value":v,"myopic":v,"policy":p,"myopic_policy":p})
    for t in d["tests"]:
        if t["cost"]>d["budget"]:
            continue
        parts=[[i for i,b in enumerate(t["bits"]) if b==bit] for bit in (0,1)]
        if not all(parts):
            continue
        masses=[sum(h[i]["weight"] for i in part) for part in parts]
        sub=[condition(d,part,d["budget"]-t["cost"],{t["id"]}) for part in parts]
        best=[adaptive_oracle(s) for s in sub]
        stops=[stop_policy(s) for s in sub]
        p={"test":t["id"],"zero":best[0][1],"one":best[1][1]}
        myopic={"test":t["id"],"zero":{"decision":stops[0][1]},"one":{"decision":stops[1][1]}}
        v=sum((Fraction(m,total)*b[0] for m,b in zip(masses,best)),Fraction(0))-t["cost"]
        mv=sum((Fraction(m,total)*b[0] for m,b in zip(masses,stops)),Fraction(0))-t["cost"]
        require(replay_policy(d,p)==v and replay_policy(d,myopic)==mv,"forced-action replay mismatch")
        result.append({"id":"test:"+t["id"],"kind":"test","cost":t["cost"],
                       "outcome_masses":masses,"value":v,"myopic":mv,
                       "policy":p,"myopic_policy":myopic})
    return result

def root_record(packet):
    d=packet["public_case"]["data"]
    opts=options(d)
    optimum,_=adaptive_oracle(d)
    require(max(o["value"] for o in opts)==optimum,"forced-action decomposition mismatch")
    stop=max((o for o in opts if o["kind"]=="stop"),key=lambda o:(o["value"],o["id"]))
    myopic=max(opts,key=lambda o:(o["myopic"],o["kind"]=="stop",o["id"]))
    tests=[o for o in opts if o["kind"]=="test"]
    # A balanced binary split maximizes Shannon entropy for a deterministic bit.
    # This baseline then stops, so its realized value includes no privileged continuation.
    info=max(tests,key=lambda o:(min(o["outcome_masses"]),-o["cost"],o["id"])) if tests else stop
    public={"root_id":packet["research"]["content_sha256"],"source":packet["source"],
            "packet_id":packet["packet_id"],"family":"adaptive_evidence","difficulty":packet["difficulty"],
            "public_case":packet["public_case"],
            "actions":[{k:o[k] for k in ("id","kind","cost","outcome_masses")} for o in opts]}
    labels={"root_id":public["root_id"],"public_sha256":digest(public),
            "optimal_value":as_fraction(optimum),
            "actions":[{"id":o["id"],"optimal_continuation_value":as_fraction(o["value"]),
                         "regret":as_fraction(optimum-o["value"]),
                         "one_step_value":as_fraction(o["myopic"]),
                         "optimal_policy":o["policy"]} for o in opts],
            "baselines":{"stop":{"id":stop["id"],"value":as_fraction(stop["value"])},
                         "one_step":{"id":myopic["id"],"value":as_fraction(myopic["myopic"])},
                         "information_then_stop":{"id":info["id"],"value":as_fraction(info["myopic"])}}}
    return public,labels

SPLITS=(("train",2026112801),("calibration",2026112802),("public_eval",2026112803))

def build(out,*,counts=(48,16,32)):
    require(len(counts)==3 and all(type(n) is int and 1<=n<=256 for n in counts),"invalid split sizes")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    (root/"public").mkdir()
    (root/"referee").mkdir()
    adapters=load_registry()
    seen=set()
    manifest={"schema_version":"distill.research_curriculum.v1","source_kind":"synthetic_oracle_only",
              "training_admission":False,"splits":[],
              "scope":"public development; distinct generated content, not private confirmation or model performance"}
    for (split,seed),count in zip(SPLITS,counts):
        public=[]
        labels=[]
        ordinal=0
        while len(public)<count:
            require(ordinal<count*128,"cross-split generation exhausted")
            difficulty=("standard","frontier")[len(public)%2]
            p=compile_research("adaptive_evidence",seed=seed,ordinal=ordinal,difficulty=difficulty,adapters=adapters)
            ordinal+=1
            h=p["research"]["content_sha256"]
            if h in seen:
                continue
            seen.add(h)
            x,y=root_record(p)
            public.append(x)
            labels.append(y)
        actions=sum(len(row["actions"]) for row in public)
        means={}
        for name in ("stop","one_step","information_then_stop"):
            regrets=[Fraction(y["optimal_value"])-Fraction(y["baselines"][name]["value"]) for y in labels]
            require(all(r>=0 for r in regrets),"baseline beats oracle")
            means[name]={"mean_regret":as_fraction(sum(regrets,Fraction(0))/count),
                         "oracle_ties":sum(r==0 for r in regrets)}
        pdoc={"schema_version":"distill.research_public.v1","split":split,"rows":public}
        rdoc={"schema_version":"distill.research_referee.v1","split":split,"rows":labels,
              "custody":"evaluator_only","training_admission":False}
        write_json(root/"public"/(split+".json"),pdoc)
        write_json(root/"referee"/(split+".json"),rdoc)
        manifest["splits"].append({"split":split,"seed":seed,"roots":count,"actions":actions,
                                   "public_sha256":digest(pdoc),"referee_sha256":digest(rdoc),
                                   "baseline_regret":means})
    manifest["manifest_id"]=digest(manifest)
    write_json(root/"manifest.json",manifest)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--train",type=int,default=48)
    p.add_argument("--calibration",type=int,default=16)
    p.add_argument("--public-eval",type=int,default=32)
    a=p.parse_args()
    import json
    print(json.dumps(build(a.out,counts=(a.train,a.calibration,a.public_eval)),sort_keys=True,indent=2))

if __name__=="__main__":
    main()
