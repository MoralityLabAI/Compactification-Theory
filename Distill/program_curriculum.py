"""Generate public research-program states and evaluator-only action labels.

No model output is used. Roots are grouped by public spec hash. The shifted
family is reserved for public structural stress evaluation.
"""
from __future__ import annotations
import argparse
import json
from fractions import Fraction
from pathlib import Path
from .core import digest, require, write_json
from .instances import Rng
from .research_program import Model, Program, new_spec
from .program_oracle import Oracle, Baseline, evaluate_root, frac

SPLITS=(("train",2026112811,"shared_control"),("calibration",2026112812,"shared_control"),
        ("public_eval",2026112813,"shared_control"),("shifted_eval",2026112814,"shifted_control"))

def select_states(spec,oracle,per_stage=2):
    """Sample reachable observable histories, not hidden-world labels."""
    require(type(per_stage) is int and 1<=per_stage<=4,"invalid states per stage")
    r=Rng(["distill.program.state_sample.v1",digest(spec)])
    # Include contrasting law regimes, then additional reproducible worlds.
    indices=[0,63]
    while len(indices)<6:
        i=r.below(64)
        if i not in indices:
            indices.append(i)
    observed={}
    local=Baseline(oracle.model,"myopic")
    for i in indices:
        for policy in (oracle.choose,local.choose):
            env=Program(spec,i)
            while env.state.stage<4:
                obs=env.observe()
                observed[obs["state_id"]]=(obs,env.state)
                env.step(policy(env.state))
    chosen=[]
    for stage in range(4):
        rows=[x for x in observed.values() if x[1].stage==stage]
        # Retain an early observation and a state rich in evidence/transport
        # availability when possible. Selection is deterministic, not IID.
        rows.sort(key=lambda x:(len(x[0]["ledger"]),x[0]["state_id"]))
        require(rows,"missing stage in trajectory")
        selected=[rows[0]]
        if per_stage>1:
            rich=sorted(rows[1:],key=lambda x:(
                -int("transport" in oracle.model.legal(x[1])),
                -int(x[1].cert>=0),-len(x[0]["ledger"]),x[0]["state_id"]))
            selected+=rich[:per_stage-1]
        chosen+=selected
    return chosen

def root_records(spec,*,per_stage=2):
    m=Model(spec)
    oracle=Oracle(m)
    public=[]
    referee=[]
    root_id=digest(spec)
    for obs,state in select_states(spec,oracle,per_stage):
        v=oracle.expected(state)
        rows=[]
        for action in m.legal(state):
            q=oracle.action_value(state,action)
            require(q<=v,"action value exceeds optimum")
            rows.append({"action":action,"continuation_value":frac(q),"regret":frac(v-q)})
        require(any(Fraction(row["regret"])==0 for row in rows),"no optimal action")
        public.append({"root_id":root_id,"state_id":obs["state_id"],"observation":obs})
        referee.append({"root_id":root_id,"state_id":obs["state_id"],
                        "public_sha256":digest(public[-1]),"optimal_value":frac(v),
                        "optimal_actions":sorted(r["action"] for r in rows if Fraction(r["regret"])==0),
                        "actions":rows})
    require(len({x["state_id"] for x in public})==len(public),"duplicate observable state")
    work=oracle.work()
    oracle.close()
    return public,referee,work

def build(out,*,counts=(12,4,8,8),per_stage=2,benchmark=True):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid split sizes")
    require(type(benchmark) is bool,"invalid benchmark flag")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    seen=set()
    states=set()
    manifest={"schema_version":"distill.longitudinal_curriculum.v1","source_kind":"synthetic_oracle_only",
              "training_admission":False,"splits":[],
              "scope":"public development and shifted-structure stress; labels conditional on finite model; no learned policy"}
    reports=[]
    for (split,seed,family),count in zip(SPLITS,counts):
        pub=root/"public"/split
        ref=root/"referee"/split
        pub.mkdir(parents=True)
        ref.mkdir(parents=True)
        entries=[]
        ordinal=0
        while len(entries)<count:
            require(ordinal<count*128,"root deduplication exhausted")
            spec=new_spec(seed,ordinal,family,("standard","frontier")[len(entries)%2])
            ordinal+=1
            rid=digest(spec)
            if rid in seen:
                continue
            seen.add(rid)
            x,y,work=root_records(spec,per_stage=per_stage)
            require(not states.intersection(row["state_id"] for row in x),"state leakage across roots")
            states.update(row["state_id"] for row in x)
            pdoc={"schema_version":"distill.longitudinal_public.v1","split":split,"root_id":rid,"rows":x}
            rdoc={"schema_version":"distill.longitudinal_referee.v1","split":split,"root_id":rid,"rows":y,
                  "custody":"evaluator_only","training_admission":False}
            write_json(pub/(rid+".json"),pdoc)
            write_json(ref/(rid+".json"),rdoc)
            entries.append({"root_id":rid,"states":len(x),"actions":sum(len(r["actions"]) for r in y),
                            "public_sha256":digest(pdoc),"referee_sha256":digest(rdoc),"planner_work":work})
            if benchmark and split in ("public_eval","shifted_eval"):
                reports.append({"split":split,"root_id":rid,"result":evaluate_root(spec)})
        manifest["splits"].append({"split":split,"seed":seed,"family":family,"roots":len(entries),
                                  "states":sum(e["states"] for e in entries),
                                  "actions":sum(e["actions"] for e in entries),"entries":entries})
    summary={}
    for split in ("public_eval","shifted_eval"):
        rows=[x["result"] for x in reports if x["split"]==split]
        if rows:
            def mean(values):
                values=tuple(Fraction(x) for x in values)
                return frac(sum(values,Fraction(0))/len(values))
            summary[split]={"roots":len(rows),"transport_value":mean(r["transport_value"] for r in rows),
                "positive_transport_roots":sum(Fraction(r["transport_value"])>0 for r in rows),
                "policies":{name:{key:mean(r["policies"][name][key] for r in rows)
                                 for key in ("utility","regret","credits","certified","transports","false_recommendations","deferred")}
                            for name in rows[0]["policies"]},
                "world_transitions":sum(p["world_transitions"] for r in rows for p in r["policies"].values())}
    manifest["benchmark_summary"]=summary
    if reports:
        receipt={"schema_version":"distill.program_benchmark.v1","rows":reports,"summary":summary,
                 "scope":"exact expected utility, all worlds; planner work counted separately"}
        write_json(root/"benchmark.json",receipt)
        manifest["benchmark_sha256"]=digest(receipt)
    manifest["manifest_id"]=digest(manifest)
    write_json(root/"manifest.json",manifest)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--train",type=int,default=12)
    p.add_argument("--calibration",type=int,default=4)
    p.add_argument("--public-eval",type=int,default=8)
    p.add_argument("--shifted-eval",type=int,default=8)
    p.add_argument("--per-stage",type=int,default=2)
    p.add_argument("--skip-benchmark",action="store_true")
    a=p.parse_args()
    m=build(a.out,counts=(a.train,a.calibration,a.public_eval,a.shifted_eval),per_stage=a.per_stage,
            benchmark=not a.skip_benchmark)
    print(json.dumps({"manifest_id":m["manifest_id"],"splits":[{k:v for k,v in s.items() if k!="entries"} for s in m["splits"]],
                      "benchmark_summary":m["benchmark_summary"],"training_admission":False},sort_keys=True,indent=2))

if __name__=="__main__":
    main()
