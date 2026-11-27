"""Frozen public-development DISTILL-08 residual experiment.

Fit a provisional residual model on incumbent/oracle states, collect its new
reachable states on the same training roots, then refit. Calibration freezes
the gate by full-trajectory priced utility, separately for each work cap.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import time
from .core import digest, require, write_json
from .research_program import Model, new_spec
from .program_oracle import Oracle, actual_transition, frac
from .program_student import Meter
from .program_residual import Residual, learn_gate, labeled_rows, select, FEATURE_VERSION, DIM
from .student_campaign import PRICE, CPU_ENVELOPE_NS

SPLITS=(("train",2026112841,"shared_control"),("calibration",2026112842,"shared_control"),
        ("ordinary",2026112843,"shared_control"),("shifted",2026112844,"shifted_control"))
CAPS=(8192,32768)
GRID=tuple({"gate":g,"margin":t} for g in (.25,.75,1.5) for t in (.25,.75,1.5))
FIXED={"gate":.25,"margin":.25}
ARMS=("reuse_first","budget_rule","always","fixed_gate","fixed_shuffled","fixed_forced_base","calibrated")

def evaluate(spec,arm,cap,model=None,gate=None,config=None):
    env=Model(spec)
    total=env.mass(env.full_mask)
    sums=defaultdict(int)
    transitions=0
    cpu=0
    max_cpu=0
    overruns=0
    max_work=0
    for i,w in enumerate(env.worlds):
        s=env.initial()
        meter=Meter(cap)
        c=defaultdict(int)
        elapsed=0
        while s.stage<4:
            require(c["actions"]<32,"nonterminating policy")
            start=time.process_time_ns()
            a,info=select(env,s,meter,arm,model,gate,config)
            elapsed+=time.process_time_ns()-start
            require(a in env.legal(s),"illegal policy action")
            bit,t,reward=actual_transition(env,s,a,i)
            require([(b,x) for b,x in env.branches(s,a) if x.mask>>i&1]==[(bit,t)],"world replay mismatch")
            c["utility"]+=reward
            c["credits"]+=env.cost(a)
            c["certified"]+=int(a.startswith("certify:"))
            c["transports"]+=int(a=="transport")
            c["false_recommendations"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            c["deferred"]+=int(a=="defer")
            c["actions"]+=1
            for k,v in info.items():
                c[k]+=v
            s=t
            transitions+=1
        require(meter.used<=cap and c["credits"]<=spec["budget"],"cap violated")
        c["work"]=meter.used
        c["priced_numerator"]=c["utility"]*PRICE-meter.used
        for k,v in c.items():
            sums[k]+=w.mass*v
        max_work=max(max_work,meter.used)
        cpu+=elapsed
        max_cpu=max(max_cpu,elapsed)
        overruns+=int(elapsed>CPU_ENVELOPE_NS)
    keys=("utility","credits","certified","transports","false_recommendations","deferred",
          "actions","score_calls","deviations","fallbacks","gate_skips","work")
    result={k:frac(Fraction(sums[k],total)) for k in keys}
    result["priced_utility"]=frac(Fraction(sums["priced_numerator"],total*PRICE))
    result.update(max_work=max_work,world_transitions=transitions,policy_cpu_ns=cpu,
                  max_world_policy_cpu_ns=max_cpu,cpu_envelope_overruns=overruns)
    return result

def mean(vals):
    vals=list(vals)
    require(bool(vals),"empty mean")
    return sum(vals,Fraction(0))/len(vals)

def calibration(specs,cap,model,gate):
    baseline=[evaluate(s,"reuse_first",cap) for s in specs]
    candidates=[]
    for config in GRID:
        runs=[evaluate(s,"gated",cap,model,gate,config) for s in specs]
        diff=[Fraction(r["priced_utility"])-Fraction(b["priced_utility"]) for r,b in zip(runs,baseline)]
        false=mean(Fraction(r["false_recommendations"])-Fraction(b["false_recommendations"]) for r,b in zip(runs,baseline))
        candidates.append({"config":config,"mean_delta":frac(mean(diff)),
                           "wins":sum(d>0 for d in diff),"ties":sum(d==0 for d in diff),
                           "false_recommendation_delta":frac(false),
                           "score_calls":frac(mean(Fraction(r["score_calls"]) for r in runs))})
    eligible=[r for r in candidates if Fraction(r["mean_delta"])>0 and
              r["wins"]>=max(1,len(specs)//2) and Fraction(r["false_recommendation_delta"])<=0]
    best=max(eligible,key=lambda r:(Fraction(r["mean_delta"]),-Fraction(r["score_calls"]),
                   r["config"]["gate"],r["config"]["margin"])) if eligible else None
    return {"selected":best["config"] if best else None,"candidates":candidates,
            "scope":"full-prior calibration trajectories only; no evaluation-root tuning"}

def campaign(out,*,counts=(24,8,8,8),caps=CAPS,limit=96):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid split counts")
    require(caps and len(set(caps))==len(caps) and all(type(x) is int and x>0 for x in caps),"invalid caps")
    require(type(limit) is int and 8<=limit<=512,"invalid row cap")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    specs={}
    seen=set()
    manifest=[]
    for (split,seed,family),n in zip(SPLITS,counts):
        ss=[new_spec(seed,i,family,("standard","frontier")[i%2]) for i in range(n)]
        ids=[digest(s) for s in ss]
        require(len(set(ids))==len(ids) and not seen.intersection(ids),"split collision")
        seen.update(ids)
        specs[split]=ss
        manifest.append({"split":split,"seed":seed,"family":family,"roots":ids})
    timings={}
    start=time.process_time_ns()
    first=[labeled_rows(s,limit=limit) for s in specs["train"]]
    timings["first_labels_ns"]=time.process_time_ns()-start
    start=time.process_time_ns()
    provisional=Residual.fit([r for rows,_ in first for r in rows])
    timings["provisional_fit_ns"]=time.process_time_ns()-start
    start=time.process_time_ns()
    final=[labeled_rows(s,provisional=provisional,limit=limit) for s in specs["train"]]
    timings["augmented_labels_ns"]=time.process_time_ns()-start
    rows=[r for rr,_ in final for r in rr]
    start=time.process_time_ns()
    learned=Residual.fit(rows)
    shuffled=Residual.fit(rows,shuffled=True,seed=139)
    gate=learn_gate(rows)
    timings["final_fits_ns"]=time.process_time_ns()-start
    def aggregate(groups):
        return {k:sum(s[k] for _,s in groups) for k in groups[0][1]}
    stats={"initial":aggregate(first),"augmented":aggregate(final)}
    fit={"schema_version":"distill.residual_student.v1","feature_version":FEATURE_VERSION,
         "weights":learned.weights,"action_counts":learned.counts,"gate":gate,
         "shuffled_weights":shuffled.weights,"shuffled_counts":shuffled.counts,
         "training_admission":False}
    fit["fit_id"]=digest(fit)
    write_json(out/"fit.json",fit)
    start=time.process_time_ns()
    gates={str(cap):calibration(specs["calibration"],cap,learned,gate) for cap in caps}
    timings["calibration_cpu_ns"]=time.process_time_ns()-start
    results=[]
    start=time.process_time_ns()
    for split in ("ordinary","shifted"):
        for spec in specs[split]:
            oracle=Oracle(Model(spec))
            bound=oracle.expected()
            oracle.close()
            for cap in caps:
                config=gates[str(cap)]["selected"]
                runs={
                    "reuse_first":evaluate(spec,"reuse_first",cap),
                    "budget_rule":evaluate(spec,"budget_rule",cap),
                    "always":evaluate(spec,"always",cap,learned),
                    "fixed_gate":evaluate(spec,"gated",cap,learned,gate,FIXED),
                    "fixed_shuffled":evaluate(spec,"shuffled",cap,shuffled,gate,FIXED),
                    "fixed_forced_base":evaluate(spec,"forced_base",cap,learned,gate,FIXED),
                    "calibrated":evaluate(spec,"gated",cap,learned,gate,config),
                }
                for p in runs.values():
                    require(Fraction(p["utility"])<=bound,"policy exceeds exact oracle")
                # Force-base control must match raw baseline; scoring might
                # consume the cap, but the admission reserve is intended to
                # preserve enough budget. Treat a mismatch as a reported cost.
                results.append({"split":split,"root_id":digest(spec),"cap":cap,
                                "oracle_value":frac(bound),"policies":runs})
    timings["evaluation_cpu_ns"]=time.process_time_ns()-start
    summary={}
    for split in ("ordinary","shifted"):
        for cap in caps:
            group=[r for r in results if r["split"]==split and r["cap"]==cap]
            metrics={}
            for arm in ARMS:
                ps=[r["policies"][arm] for r in group]
                diffs=[Fraction(r["policies"][arm]["priced_utility"])-Fraction(r["policies"]["reuse_first"]["priced_utility"]) for r in group]
                metrics[arm]={k:frac(mean(Fraction(p[k]) for p in ps)) for k in
                    ("utility","priced_utility","work","certified","transports","false_recommendations","score_calls","deviations","fallbacks","gate_skips")}
                metrics[arm].update(priced_delta=frac(mean(diffs)),wins=sum(v>0 for v in diffs),
                    ties=sum(v==0 for v in diffs),losses=sum(v<0 for v in diffs),
                    world_transitions=sum(p["world_transitions"] for p in ps),
                    policy_cpu_ns=sum(p["policy_cpu_ns"] for p in ps),
                    cpu_envelope_overruns=sum(p["cpu_envelope_overruns"] for p in ps),
                    max_work=max(p["max_work"] for p in ps))
            summary[split+"/"+str(cap)]={"roots":len(group),"policies":metrics}
    report={"schema_version":"distill.residual_campaign.v1","fit_id":fit["fit_id"],
            "feature_dimension":DIM,"parameters":DIM*len(learned.weights),"gate_buckets":len(gate),
            "source_kind":"synthetic_oracle_only","training_admission":False,"splits":manifest,
            "caps":list(caps),"price_per_unit":"1/"+str(PRICE),"training_stats":stats,
            "timings":timings,"calibration":gates,"summary":summary,"rows":results,
            "scope":"public development; exact full-prior replay, matched logical-work caps, measured process CPU separately; no promotion"}
    report["report_id"]=digest(report)
    write_json(out/"report.json",report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--train",type=int,default=24)
    p.add_argument("--calibration",type=int,default=8)
    p.add_argument("--ordinary",type=int,default=8)
    p.add_argument("--shifted",type=int,default=8)
    a=p.parse_args()
    r=campaign(a.out,counts=(a.train,a.calibration,a.ordinary,a.shifted))
    print(json.dumps({k:v for k,v in r.items() if k not in ("rows","splits")},sort_keys=True))

if __name__=="__main__":
    main()
