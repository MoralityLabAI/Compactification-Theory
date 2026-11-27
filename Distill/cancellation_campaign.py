"""Frozen DISTILL-10 comparison of signed cancellation and separate frontiers."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import time
from .core import digest, require, write_json
from .research_program import Model
from .program_student import Meter
from .program_oracle import Oracle, actual_transition, frac
from .program_cancellation import Policy, Auditor, CONFIGS
from .opportunity_campaign import make_spec, mean, evaluate as legacy_evaluate
from .program_opportunity import CONFIGS as LEGACY_CONFIGS
from .student_campaign import PRICE, CPU_ENVELOPE_NS

SPLITS=(("calibration",2026112861,"shared_control"),("ordinary",2026112862,"shared_control"),
        ("shifted",2026112863,"shifted_control"),("scarce",2026112864,"shifted_control"))
CAPS=(8192,32768)
ARMS=("budget_rule","legacy_late","legacy_bounded","paired_late","split_late",
      "forced_late","paired_tail","split_tail","forced_tail",
      "paired_broad","split_broad","forced_broad","calibrated")
MEANS=("utility","priced_utility","work","search_work","attempts","accepted","deviations",
       "gates","trials","expanded","cancelled_pairs","frontier","exhausted","fallbacks",
       "certified","false_recommendations","credits","transports","deferred","actions")
COUNTS=("world_transitions","policy_cpu_ns","cpu_envelope_overruns","receipt_occurrences",
        "accepted_occurrences","unique_receipts","audited_intervals","independent_action_replays")

def evaluate(spec,config,cap,*,cancel=True,forced=False,audit=True):
    m=Model(spec)
    ref=Auditor(m) if audit else None
    total=defaultdict(int)
    M=m.mass(m.full_mask)
    transitions=cpu=overruns=max_work=max_cpu=0
    receipts={}
    occurrences=accepted=0
    margins=[]
    for i,w in enumerate(m.worlds):
        s=m.initial()
        actor=Policy(config,cancel=cancel,forced=forced)
        meter=Meter(cap)
        c=defaultdict(int)
        elapsed=0
        while s.stage<4:
            require(c["actions"]<32,"nonterminating policy")
            t0=time.process_time_ns()
            a,info,receipt=actor.choose(m,s,meter)
            elapsed+=time.process_time_ns()-t0
            require(a in m.legal(s),"illegal actor")
            if receipt is not None:
                occurrences+=1
                rid=receipt["receipt_id"]
                if rid not in receipts:
                    receipts[rid]=receipt
                    if ref:
                        ref.verify(s,receipt)
                if receipt["chosen"] is not None:
                    accepted+=1
                    require(a==(receipt["baseline"] if forced else receipt["chosen"]),"receipt/action mismatch")
                    margins.append(Fraction(receipt["net_lower_numerator"],receipt["mass"]*PRICE))
            bit,t,reward=actual_transition(m,s,a,i)
            require([(b,x) for b,x in m.branches(s,a) if x.mask>>i&1]==[(bit,t)],"world replay mismatch")
            c["utility"]+=reward
            c["credits"]+=m.cost(a)
            c["certified"]+=int(a.startswith("certify:"))
            c["transports"]+=int(a=="transport")
            c["deferred"]+=int(a=="defer")
            c["false_recommendations"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            c["actions"]+=1
            for k,v in info.items():
                c[k]+=v
            s=t
            transitions+=1
        require(meter.used<=cap and c["credits"]<=spec["budget"],"budget violated")
        require(c["attempts"]<=1 and c["deviations"]<=1,"one-shot contract")
        if c["attempts"]:
            require(c["fallbacks"]==0,"reserved continuation exhausted")
        c["work"]=meter.used
        c["priced_numerator"]=PRICE*c["utility"]-meter.used
        for k,v in c.items():
            total[k]+=w.mass*v
        max_work=max(max_work,meter.used)
        cpu+=elapsed
        max_cpu=max(max_cpu,elapsed)
        overruns+=int(elapsed>CPU_ENVELOPE_NS)
    r={k:frac(Fraction(total[k],M)) for k in MEANS if k!="priced_utility"}
    r["priced_utility"]=frac(Fraction(total["priced_numerator"],PRICE*M))
    r.update(world_transitions=transitions,policy_cpu_ns=cpu,cpu_envelope_overruns=overruns,
             max_work=max_work,max_world_policy_cpu_ns=max_cpu,receipt_occurrences=occurrences,
             accepted_occurrences=accepted,unique_receipts=len(receipts),
             audited_intervals=ref.checked if ref else 0,independent_action_replays=ref.replays if ref else 0,
             min_accepted_net_lower=frac(min(margins)) if margins else None,
             receipt_set_sha256=digest([receipts[r] for r in sorted(receipts)]))
    if ref:
        ref.close()
    return r,[receipts[r] for r in sorted(receipts)]

def calibration(specs,cap):
    base=[evaluate(s,None,cap,audit=False)[0] for s in specs]
    candidates=[]
    for name,config in CONFIGS.items():
        rr=[evaluate(s,config,cap,audit=False)[0] for s in specs]
        diff=[Fraction(r["priced_utility"])-Fraction(b["priced_utility"]) for r,b in zip(rr,base)]
        risk=[Fraction(r["false_recommendations"])-Fraction(b["false_recommendations"]) for r,b in zip(rr,base)]
        require(all(d<=0 for d in risk),"calibration risk increased")
        candidates.append({"name":name,"mean_delta":frac(mean(diff)),"wins":sum(d>0 for d in diff),
                           "mean_risk_delta":frac(mean(risk)),"mean_attempts":frac(mean(Fraction(r["attempts"]) for r in rr))})
    valid=[r for r in candidates if Fraction(r["mean_delta"])>0 and r["wins"]>=max(1,len(specs)//2)]
    best=max(valid,key=lambda r:(Fraction(r["mean_delta"]),-Fraction(r["mean_attempts"]),r["name"])) if valid else None
    return {"selected":best["name"] if best else None,"candidates":candidates}

def campaign(out,*,counts=(8,12,12,12),caps=CAPS):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid counts")
    require(caps and len(set(caps))==len(caps) and all(type(c) is int and c>=4096 for c in caps),"invalid caps")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"receipts").mkdir()
    (out/"rows").mkdir()
    specs={}
    manifest=[]
    seen=set()
    for (split,seed,family),n in zip(SPLITS,counts):
        ss=[make_spec(seed,i,family,split=="scarce") for i in range(n)]
        ids=[digest(s) for s in ss]
        require(len(set(ids))==len(ids) and not seen.intersection(ids),"split collision")
        seen.update(ids)
        specs[split]=ss
        manifest.append({"split":split,"seed":seed,"family":family,"roots":ids})
    start=time.process_time_ns()
    choices={str(cap):calibration(specs["calibration"],cap) for cap in caps}
    cal_cpu=time.process_time_ns()-start
    rows=[]
    row_index=[]
    start=time.process_time_ns()
    for split in ("ordinary","shifted","scarce"):
        for spec in specs[split]:
            rid=digest(spec)
            oracle=Oracle(Model(spec))
            optimum=oracle.expected()
            oracle.close()
            for cap in caps:
                selected=choices[str(cap)]["selected"]
                arms={"budget_rule":(None,True,False),"calibrated":(CONFIGS[selected] if selected else None,True,False)}
                for name,cfg in CONFIGS.items():
                    arms["paired_"+name]=(cfg,True,False)
                    arms["split_"+name]=(cfg,False,False)
                    arms["forced_"+name]=(cfg,True,True)
                policies={}
                def store(name,r,receipts):
                    require(Fraction(r["utility"])<=optimum,"actor exceeds exact optimum")
                    require(r["max_work"]<=cap,"work exceeds cap")
                    if receipts:
                        write_json(out/"receipts"/(rid+"-"+str(cap)+"-"+name+".json"),
                                   {"spec":spec,"cap":cap,"arm":name,"receipts":receipts})
                    policies[name]=r
                for name,(cfg,cancel,forced) in arms.items():
                    store(name,*evaluate(spec,cfg,cap,cancel=cancel,forced=forced))
                # Prior search implementations, intentionally without local
                # risk certification; separate comparison from paired/split.
                for name in ("late","bounded"):
                    r,rs=legacy_evaluate(spec,LEGACY_CONFIGS[name],cap)
                    r.update(cancelled_pairs="0/1",frontier=r["cutoffs"])
                    store("legacy_"+name,r,rs)
                b=policies["budget_rule"]
                for name in CONFIGS:
                    f=policies["forced_"+name]
                    require(all(f[k]==b[k] for k in ("utility","credits","certified","transports","false_recommendations","deferred","actions")),
                            "forced baseline raw parity")
                    for arm in ("paired_"+name,"split_"+name):
                        require(Fraction(policies[arm]["false_recommendations"])<=Fraction(b["false_recommendations"]),"incremental risk violation")
                require(Fraction(policies["calibrated"]["false_recommendations"])<=Fraction(b["false_recommendations"]),"calibrated risk violation")
                row={"root_id":rid,"split":split,"cap":cap,"spec":spec,"oracle_value":frac(optimum),"policies":policies}
                name=rid+"-"+str(cap)+".json"
                write_json(out/"rows"/name,row)
                row_index.append({"path":"rows/"+name,"sha256":digest(row)})
                rows.append(row)
    eval_cpu=time.process_time_ns()-start
    summary={}
    for split in ("ordinary","shifted","scarce"):
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            result={}
            for name in ARMS:
                pp=[r["policies"][name] for r in group]
                diffs=[Fraction(p["priced_utility"])-Fraction(r["policies"]["budget_rule"]["priced_utility"]) for p,r in zip(pp,group)]
                risk=[Fraction(p["false_recommendations"])-Fraction(r["policies"]["budget_rule"]["false_recommendations"]) for p,r in zip(pp,group)]
                result[name]={k:frac(mean(Fraction(p[k]) for p in pp)) for k in MEANS}
                result[name].update(priced_delta=frac(mean(diffs)),risk_delta=frac(mean(risk)),
                                    wins=sum(x>0 for x in diffs),ties=sum(x==0 for x in diffs),losses=sum(x<0 for x in diffs),
                                    risk_increased_roots=sum(x>0 for x in risk),max_work=max(p["max_work"] for p in pp))
                for k in COUNTS:
                    result[name][k]=sum(p[k] for p in pp)
                margins=[Fraction(p["min_accepted_net_lower"]) for p in pp if p["min_accepted_net_lower"] is not None]
                result[name]["min_accepted_net_lower"]=frac(min(margins)) if margins else None
            paired={}
            for name in CONFIGS:
                a=result["paired_"+name]
                b=result["split_"+name]
                paired[name]={"work_saved":frac(Fraction(b["search_work"])-Fraction(a["search_work"])),
                    "priced_delta":frac(Fraction(a["priced_utility"])-Fraction(b["priced_utility"])),
                    "expansions_saved":frac(Fraction(b["expanded"])-Fraction(a["expanded"])),
                    "cancellations":a["cancelled_pairs"]}
            summary[split+"/"+str(cap)]={"roots":len(group),"policies":result,"paired_minus_split":paired}
    report={"schema_version":"distill.cancellation_campaign.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"caps":list(caps),"price_per_unit":"1/"+str(PRICE),
            "configs":CONFIGS,"split_manifest":manifest,"calibration":choices,
            "calibration_cpu_ns":cal_cpu,"evaluation_cpu_ns":eval_cpu,
            "summary":summary,"row_index":row_index,
            "scope":"signed exact-state cancellation; one intervention; nonpositive conditional incremental expected false recommendations; public finite model; logical work caps"}
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
    r=campaign(args.out,counts=(args.calibration,args.ordinary,args.shifted,args.scarce))
    print(json.dumps({k:v for k,v in r.items() if k not in ("row_index","split_manifest")},sort_keys=True))

if __name__=="__main__":
    main()
