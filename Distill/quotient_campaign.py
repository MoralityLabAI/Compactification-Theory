"""DISTILL-11 frozen quotient benchmark and complete reconvergence diagnostic."""
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
from .program_quotient import diagnostic,MODES
from .quotient_policy import Policy,Auditor,CONFIGS
from .program_projection import METRICS
from .cancellation_campaign import evaluate as original_evaluate
from .opportunity_campaign import make_spec,mean
from .student_campaign import PRICE,CPU_ENVELOPE_NS

SPLITS=(("calibration",2026112871,"shared_control"),("ordinary",2026112872,"shared_control"),
        ("shifted",2026112873,"shifted_control"),("scarce",2026112874,"shifted_control"))
CAPS=(8192,32768)
MEANS=("utility","priced_utility","work","search_work","attempts","accepted","deviations",
       "gates","trials","expanded","cancelled_pairs","quotient_merges","frontier","exhausted","fallbacks",
       "certified","false_recommendations","credits","transports","deferred","actions")
COUNTS=("world_transitions","policy_cpu_ns","cpu_envelope_overruns","receipt_occurrences",
        "accepted_occurrences","unique_receipts","audited_intervals","independent_action_replays")
CANDIDATES=tuple((mode,name) for mode in MODES for name in ("tail","broad"))
ARMS=("budget_rule","original_tail","original_broad","late",
      *(mode+"_"+name for mode,name in CANDIDATES),
      "forced_flags_tail","forced_marginal_tail","calibrated")

def evaluate(spec,config,cap,*,mode="marginal",forced=False,audit=True,optimization=None):
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
        actor=Policy(config,mode=mode,forced=forced,optimization=optimization)
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
    if optimization is not None:
        r.update({k:frac(Fraction(total[k],M)) for k in METRICS})
    if ref:
        ref.close()
    return r,[receipts[r] for r in sorted(receipts)]


def calibration(specs,cap):
    base=[evaluate(s,None,cap,audit=False)[0] for s in specs]
    configs=[]
    for mode,name in (*CANDIDATES,("literal","late")):
        rr=[evaluate(s,CONFIGS[name],cap,mode=mode,audit=False)[0] for s in specs]
        diff=[Fraction(r["priced_utility"])-Fraction(b["priced_utility"]) for r,b in zip(rr,base)]
        risks=[Fraction(r["false_recommendations"])-Fraction(b["false_recommendations"]) for r,b in zip(rr,base)]
        require(all(x<=0 for x in risks),"calibration risk increased")
        configs.append({"mode":mode,"name":name,"mean_delta":frac(mean(diff)),
                        "wins":sum(x>0 for x in diff),"mean_risk_delta":frac(mean(risks))})
    viable=[r for r in configs if Fraction(r["mean_delta"])>0 and r["wins"]>=max(1,len(specs)//2)]
    selected=max(viable,key=lambda r:(Fraction(r["mean_delta"]),r["mode"],r["name"])) if viable else None
    return {"selected":selected,"candidates":configs}

def diag_summary(rows):
    out={"roots":len(rows),"candidate_pairs":sum(r["candidate_pairs"] for r in rows),
         "incumbent_states":sum(r["incumbent_states"] for r in rows)}
    out["overlap"]={mode:{k:sum(r["overlap"][mode][k] for r in rows)
                          for k in ("pairs_with_overlap","shared_classes")} for mode in MODES}
    out["quotient_only_pairs"]={mode:sum(r["quotient_only_pairs"][mode] for r in rows)
                                for mode in ("flags","marginal")}
    out["congruence"]={mode:{k:sum(r["congruence"][mode][k] for r in rows)
                                  for k in ("states","classes","nontrivial_classes")} for mode in MODES}
    return out

def campaign(out,*,counts=(8,12,12,12),caps=CAPS):
    require(len(counts)==4 and all(type(n) is int and 1<=n<=64 for n in counts),"invalid counts")
    require(caps and len(set(caps))==len(caps) and all(type(c) is int and c>=4096 for c in caps),"invalid caps")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    for name in ("rows","receipts","diagnostics"):
        (out/name).mkdir()
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
    diag_rows=defaultdict(list)
    index=[]
    diag_index=[]
    start=time.process_time_ns()
    for split in ("ordinary","shifted","scarce"):
        for spec in specs[split]:
            rid=digest(spec)
            m=Model(spec)
            diag=diagnostic(m)
            diag_rows[split].append(diag)
            drow={"root_id":rid,"split":split,"spec":spec,"diagnostic":diag}
            dpath="diagnostics/"+rid+".json"
            write_json(out/dpath,drow)
            diag_index.append({"path":dpath,"sha256":digest(drow)})
            oracle=Oracle(m)
            opt=oracle.expected()
            oracle.close()
            for cap in caps:
                selected=choices[str(cap)]["selected"]
                arms={"budget_rule":(None,"literal",False),"late":(CONFIGS["late"],"literal",False),
                      "calibrated":(CONFIGS[selected["name"]] if selected else None,
                                    selected["mode"] if selected else "literal",False)}
                for mode,name in CANDIDATES:
                    arms[mode+"_"+name]=(CONFIGS[name],mode,False)
                for mode in ("flags","marginal"):
                    arms["forced_"+mode+"_tail"]=(CONFIGS["tail"],mode,True)
                policies={}
                def store(name,r,receipts):
                    require(Fraction(r["utility"])<=opt,"actor exceeds raw oracle")
                    require(r["max_work"]<=cap,"work exceeded")
                    if receipts:
                        write_json(out/"receipts"/(rid+"-"+str(cap)+"-"+name+".json"),
                                   {"spec":spec,"cap":cap,"arm":name,"receipts":receipts})
                    policies[name]=r
                for name,(cfg,mode,forced) in arms.items():
                    store(name,*evaluate(spec,cfg,cap,mode=mode,forced=forced))
                for name in ("tail","broad"):
                    r,rs=original_evaluate(spec,CONFIGS[name],cap)
                    r["quotient_merges"]="0/1"
                    store("original_"+name,r,rs)
                b=policies["budget_rule"]
                for name,r in policies.items():
                    require(Fraction(r["false_recommendations"])<=Fraction(b["false_recommendations"]),
                            "root incremental risk increased")
                    if name.startswith("forced_"):
                        require(all(r[k]==b[k] for k in ("utility","credits","certified","transports","deferred","false_recommendations","actions")),
                                "forced control parity")
                row={"root_id":rid,"split":split,"cap":cap,"spec":spec,"oracle_value":frac(opt),"policies":policies}
                path="rows/"+rid+"-"+str(cap)+".json"
                write_json(out/path,row)
                index.append({"path":path,"sha256":digest(row)})
                rows.append(row)
    eval_cpu=time.process_time_ns()-start
    summary={}
    for split in ("ordinary","shifted","scarce"):
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            metrics={}
            for name in ARMS:
                pp=[r["policies"][name] for r in group]
                ds=[Fraction(p["priced_utility"])-Fraction(r["policies"]["budget_rule"]["priced_utility"]) for p,r in zip(pp,group)]
                risk=[Fraction(p["false_recommendations"])-Fraction(r["policies"]["budget_rule"]["false_recommendations"]) for p,r in zip(pp,group)]
                x={k:frac(mean(Fraction(p[k]) for p in pp)) for k in MEANS}
                x.update(priced_delta=frac(mean(ds)),risk_delta=frac(mean(risk)),
                         wins=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d<0 for d in ds),
                         risk_increased_roots=sum(d>0 for d in risk),max_work=max(p["max_work"] for p in pp))
                for k in COUNTS:
                    x[k]=sum(p[k] for p in pp)
                margins=[Fraction(p["min_accepted_net_lower"]) for p in pp if p["min_accepted_net_lower"] is not None]
                x["min_accepted_net_lower"]=frac(min(margins)) if margins else None
                metrics[name]=x
            contrasts={}
            for mode in ("flags","marginal"):
                for name in ("tail","broad"):
                    a=metrics[mode+"_"+name]
                    b=metrics["literal_"+name]
                    contrasts[mode+"_"+name]={"search_work_saved":frac(Fraction(b["search_work"])-Fraction(a["search_work"])),
                                             "expanded_saved":frac(Fraction(b["expanded"])-Fraction(a["expanded"])),
                                             "priced_delta_vs_literal":frac(Fraction(a["priced_utility"])-Fraction(b["priced_utility"])),
                                             "quotient_merges":a["quotient_merges"]}
            summary[split+"/"+str(cap)]={"roots":len(group),"policies":metrics,"quotient_minus_literal":contrasts}
    report={"schema_version":"distill.quotient_campaign.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"caps":list(caps),"price_per_unit":"1/"+str(PRICE),
            "split_manifest":manifest,"configs":CONFIGS,"calibration":choices,
            "calibration_cpu_ns":cal_cpu,"evaluation_cpu_ns":eval_cpu,
            "diagnostics":{k:diag_summary(v) for k,v in diag_rows.items()},
            "summary":summary,"row_index":index,"diagnostic_index":diag_index,
            "scope":"policy-relative quotient; exact normalized reward/risk/successor congruence on complete candidate closures; charged projection; finite public development"}
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
    a=p.parse_args()
    r=campaign(a.out,counts=(a.calibration,a.ordinary,a.shifted,a.scarce))
    print(json.dumps({k:v for k,v in r.items() if k not in ("row_index","diagnostic_index","split_manifest")},sort_keys=True))

if __name__=="__main__":
    main()
