"""Frozen DISTILL-09 bounded opportunity evaluation.

Calibration chooses a search config on complete ordinary programs. All arms
replay the 64 worlds with exact prior weights. Independent receipt auditing
is referee work; every actor search and rejected search is charged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import time
from .core import digest, require, write_json
from .research_program import Model, new_spec, validate_spec
from .program_student import Meter
from .program_oracle import Oracle, actual_transition, frac
from .program_opportunity import Policy, Auditor, CONFIGS, reserve
from .instances import Rng
from .student_campaign import PRICE, CPU_ENVELOPE_NS

SPLITS=(("calibration",2026112851,"shared_control"),("ordinary",2026112852,"shared_control"),
        ("shifted",2026112853,"shifted_control"),("scarce",2026112854,"shifted_control"))
CAPS=(8192,32768)
ARMS=("budget_rule","late","forced_late","bounded","forced_bounded","broad","forced_broad","calibrated")

def make_spec(seed,i,family,scarce=False):
    spec=new_spec(seed,i,family,("standard","frontier")[i%2])
    if scarce:
        r=Rng(["distill.opportunity.scarcity.v1",seed,i])
        spec["budget"]=r.between(4,7)
        spec["prior_weights"]["law"]=[r.between(3,6),r.between(1,3)]
        spec["costs"]["control"]=r.between(1,3)
        validate_spec(spec)
    return spec

def mean(values):
    values=list(values)
    require(bool(values),"empty mean")
    return sum(values,Fraction(0))/len(values)

def evaluate(spec,config,cap,*,forced=False,audit=True):
    m=Model(spec)
    auditor=Auditor(m) if audit else None
    totals=defaultdict(int)
    total_mass=m.mass(m.full_mask)
    transitions=0
    policy_cpu=0
    cpu_overruns=0
    cpu_max=0
    max_work=0
    receipts={}
    complete=0
    accepted=0
    replay_checks=0
    accepted_margins=[]
    for i,w in enumerate(m.worlds):
        s=m.initial()
        meter=Meter(cap)
        actor=Policy(config,forced=forced)
        counts=defaultdict(int)
        elapsed=0
        while s.stage<4:
            require(counts["actions"]<32,"nonterminating policy")
            t0=time.process_time_ns()
            a,info,receipt=actor.choose(m,s,meter)
            elapsed+=time.process_time_ns()-t0
            require(a in m.legal(s),"illegal actor output")
            if receipt is not None:
                complete+=1
                rid=receipt["receipt_id"]
                if rid not in receipts:
                    receipts[rid]=receipt
                    if audit:
                        auditor.verify(s,receipt)
                        if receipt["chosen"] is not None:
                            auditor.replay_action(s,receipt["baseline"])
                            auditor.replay_action(s,receipt["chosen"])
                            replay_checks+=2
                if receipt["chosen"] is not None:
                    accepted+=1
                    accepted_margins.append(Fraction(receipt["net_lower_numerator"],receipt["mass"]*PRICE))
                    require((a==receipt["baseline"] if forced else a==receipt["chosen"]),"certificate action mismatch")
            bit,t,reward=actual_transition(m,s,a,i)
            require([(b,x) for b,x in m.branches(s,a) if x.mask>>i&1]==[(bit,t)],"world replay mismatch")
            counts["utility"]+=reward
            counts["credits"]+=m.cost(a)
            counts["certified"]+=int(a.startswith("certify:"))
            counts["transports"]+=int(a=="transport")
            counts["false_recommendations"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            counts["deferred"]+=int(a=="defer")
            counts["actions"]+=1
            for k,v in info.items():
                counts[k]+=v
            s=t
            transitions+=1
        require(meter.used<=cap and counts["credits"]<=spec["budget"],"budget violation")
        require(counts["attempts"]<=1 and counts["deviations"]<=1,"one-shot policy violation")
        if counts["attempts"]:
            require(counts["fallbacks"]==0,"reserved incumbent continuation exhausted")
        counts["work"]=meter.used
        counts["priced_numerator"]=counts["utility"]*PRICE-meter.used
        for k,v in counts.items():
            totals[k]+=w.mass*v
        max_work=max(max_work,meter.used)
        policy_cpu+=elapsed
        cpu_max=max(cpu_max,elapsed)
        cpu_overruns+=int(elapsed>CPU_ENVELOPE_NS)
    keys=("utility","credits","certified","transports","false_recommendations","deferred","actions",
          "gates","attempts","accepted","deviations","search_work","trials","cutoffs","expanded",
          "cache_hits","fallbacks","exhausted","work")
    result={k:frac(Fraction(totals[k],total_mass)) for k in keys}
    result.update(priced_utility=frac(Fraction(totals["priced_numerator"],total_mass*PRICE)),
                  world_transitions=transitions,policy_cpu_ns=policy_cpu,
                  max_world_policy_cpu_ns=cpu_max,cpu_envelope_overruns=cpu_overruns,max_work=max_work,
                  receipt_occurrences=complete,accepted_occurrences=accepted,
                  unique_receipts=len(receipts),audited_intervals=auditor.checks if audit else 0,
                  independent_action_replays=replay_checks,
                  min_accepted_net_lower=frac(min(accepted_margins)) if accepted_margins else None,
                  receipt_set_sha256=digest([receipts[r] for r in sorted(receipts)]))
    if audit:
        auditor.close()
    return result, [receipts[r] for r in sorted(receipts)]

def calibration(specs,cap):
    base=[evaluate(s,None,cap,audit=False)[0] for s in specs]
    candidates=[]
    for name,config in CONFIGS.items():
        rr=[evaluate(s,config,cap,audit=False)[0] for s in specs]
        diffs=[Fraction(r["priced_utility"])-Fraction(b["priced_utility"]) for r,b in zip(rr,base)]
        false=mean(Fraction(r["false_recommendations"])-Fraction(b["false_recommendations"]) for r,b in zip(rr,base))
        candidates.append({"name":name,"mean_delta":frac(mean(diffs)),"wins":sum(d>0 for d in diffs),
            "ties":sum(d==0 for d in diffs),"false_delta":frac(false),
            "mean_attempts":frac(mean(Fraction(r["attempts"]) for r in rr)),
            "mean_accepted":frac(mean(Fraction(r["accepted"]) for r in rr))})
    valid=[r for r in candidates if Fraction(r["mean_delta"])>0 and r["wins"]>=max(1,len(specs)//2)
           and Fraction(r["false_delta"])<=0]
    best=max(valid,key=lambda r:(Fraction(r["mean_delta"]),-Fraction(r["mean_attempts"]),r["name"])) if valid else None
    return {"selected":best["name"] if best else None,"candidates":candidates}

def campaign(out,*,counts=(8,12,12,12),caps=CAPS):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid split sizes")
    require(caps and len(set(caps))==len(caps) and all(type(c) is int and c>=4096 for c in caps),"invalid caps")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    specs={}
    manifest=[]
    seen=set()
    for (name,seed,family),n in zip(SPLITS,counts):
        ss=[make_spec(seed,i,family,name=="scarce") for i in range(n)]
        ids=[digest(s) for s in ss]
        require(len(set(ids))==len(ids) and not seen.intersection(ids),"split collision")
        seen.update(ids)
        specs[name]=ss
        manifest.append({"split":name,"seed":seed,"family":family,"scarce":name=="scarce","roots":ids})
    start=time.process_time_ns()
    choices={str(cap):calibration(specs["calibration"],cap) for cap in caps}
    cal_cpu=time.process_time_ns()-start
    rows=[]
    rdir=out/"receipts"
    rdir.mkdir()
    start=time.process_time_ns()
    for split in ("ordinary","shifted","scarce"):
        for spec in specs[split]:
            rid=digest(spec)
            oracle=Oracle(Model(spec))
            opt=oracle.expected()
            oracle.close()
            for cap in caps:
                arms={"budget_rule":(None,False)}
                for name,config in CONFIGS.items():
                    arms[name]=(config,False)
                    arms["forced_"+name]=(config,True)
                selected=choices[str(cap)]["selected"]
                arms["calibrated"]=(CONFIGS[selected] if selected else None,False)
                policies={}
                for name,(config,forced) in arms.items():
                    row,receipts=evaluate(spec,config,cap,forced=forced)
                    require(Fraction(row["utility"])<=opt,"actor exceeds raw optimum")
                    if receipts:
                        write_json(rdir/(rid+"-"+str(cap)+"-"+name+".json"),
                                   {"spec":spec,"cap":cap,"arm":name,"receipts":receipts})
                    policies[name]=row
                b=policies["budget_rule"]
                # Each forced arm commits to the incumbent, so sufficient
                # reserve should preserve all raw decisions and rewards.
                for name in CONFIGS:
                    p=policies["forced_"+name]
                    require(all(p[k]==b[k] for k in ("utility","credits","certified","transports","false_recommendations","deferred","actions")),
                            "forced-incumbent raw parity failed")
                rows.append({"split":split,"root_id":rid,"cap":cap,"oracle_value":frac(opt),"policies":policies})
    eval_cpu=time.process_time_ns()-start
    summary={}
    keys=("utility","priced_utility","work","search_work","attempts","accepted","deviations",
          "gates","trials","cutoffs","expanded","cache_hits","fallbacks","exhausted","certified","false_recommendations")
    for split in ("ordinary","shifted","scarce"):
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            metrics={}
            for name in ARMS:
                rr=[r["policies"][name] for r in group]
                diffs=[Fraction(p["priced_utility"])-Fraction(r["policies"]["budget_rule"]["priced_utility"])
                       for p,r in zip(rr,group)]
                metrics[name]={k:frac(mean(Fraction(p[k]) for p in rr)) for k in keys}
                metrics[name].update(priced_delta=frac(mean(diffs)),wins=sum(x>0 for x in diffs),
                    ties=sum(x==0 for x in diffs),losses=sum(x<0 for x in diffs),
                    max_work=max(p["max_work"] for p in rr))
                for k in ("world_transitions","policy_cpu_ns","cpu_envelope_overruns","receipt_occurrences","accepted_occurrences",
                          "unique_receipts","audited_intervals","independent_action_replays"):
                    metrics[name][k]=sum(p[k] for p in rr)
                mins=[Fraction(p["min_accepted_net_lower"]) for p in rr if p["min_accepted_net_lower"] is not None]
                metrics[name]["min_accepted_net_lower"]=frac(min(mins)) if mins else None
            summary[split+"/"+str(cap)]={"roots":len(group),"policies":metrics}
    report={"schema_version":"distill.opportunity_campaign.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"caps":list(caps),"price_per_unit":"1/"+str(PRICE),
            "configs":CONFIGS,"split_manifest":manifest,"calibration":choices,
            "calibration_cpu_ns":cal_cpu,"evaluation_cpu_ns":eval_cpu,
            "summary":summary,"rows":rows,
            "scope":"one-shot local certificates relative to budget-rule continuation; whole-trajectory cost includes rejects, gate and search; public development, not native or strict CPU-time match"}
    report["report_id"]=digest(report)
    write_json(out/"report.json",report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--calibration",type=int,default=8)
    p.add_argument("--ordinary",type=int,default=12)
    p.add_argument("--shifted",type=int,default=12)
    p.add_argument("--scarce",type=int,default=12)
    args=p.parse_args()
    report=campaign(args.out,counts=(args.calibration,args.ordinary,args.shifted,args.scarce))
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","split_manifest")},sort_keys=True))

if __name__=="__main__":
    main()
