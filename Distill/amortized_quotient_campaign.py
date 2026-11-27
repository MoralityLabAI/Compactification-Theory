"""DISTILL-12: frozen amortization campaign; independent exact risk auditor."""
import argparse
from fractions import Fraction
from pathlib import Path
import time

from .core import digest, require, write_json
from .research_program import Model
from .program_student import Meter
from .program_quotient import signature, closure
from .program_projection import Projector, METRICS
from .quotient_campaign import evaluate, CONFIGS
from .opportunity_campaign import make_spec
from .student_campaign import PRICE

SPLITS=(("calibration",2026112891,"shared_control"),
        ("ordinary",2026112892,"shared_control"),
        ("shifted",2026112893,"shifted_control"),
        ("scarce",2026112894,"shifted_control"))
CAPS=(8192,32768)
KERNELS=(("literal","literal",None),("scan","marginal",None),
         ("bitset","marginal","bitset"),("cache","marginal","cache"),
         ("bitset_cache","marginal","bitset_cache"),
         ("prune","marginal","bitset_cache_prune"))
ARMS={name+"_"+gate:(CONFIGS[gate],mode,opt,False)
      for name,mode,opt in KERNELS for gate in ("tail","broad")}
ARMS["late"]=(CONFIGS["late"],"literal",None,False)
BASE=(None,"literal",None,False)
FORCED={name+"_forced_"+gate:(CONFIGS[gate],mode,opt,True)
        for name,mode,opt in KERNELS[-2:] for gate in ("tail","broad")}
RAW=("utility","false_recommendations","credits","transports","certified",
     "deferred","actions")

def mean(xs):
    xs=list(xs)
    return sum(xs,Fraction(0))/len(xs)

def f(x):
    return str(Fraction(x))

def run_arm(spec,cap,arm,audit=True):
    config,mode,opt,forced=arm
    return evaluate(spec,config,cap,mode=mode,optimization=opt,
                    forced=forced,audit=audit)

def aggregate(rows,name):
    p=[r["arms"][name] for r in rows]
    base=[r["arms"]["budget_rule"] for r in rows]
    delta=[Fraction(x["priced_utility"])-Fraction(y["priced_utility"])
           for x,y in zip(p,base)]
    risks=[Fraction(x["false_recommendations"])-Fraction(y["false_recommendations"])
           for x,y in zip(p,base)]
    require(all(r<=0 for r in risks),"risk increased")
    fields=("utility","priced_utility","work","search_work","expanded",
            "quotient_merges","accepted","deviations",*METRICS)
    return {**{k:f(mean(Fraction(x.get(k,0)) for x in p)) for k in fields},
            "priced_delta":f(mean(delta)),"mean_risk_delta":f(mean(risks)),
            "wins":sum(x>0 for x in delta),"ties":sum(x==0 for x in delta),
            "losses":sum(x<0 for x in delta),"roots":len(rows),
            "max_work":max(x["max_work"] for x in p),
            **{k:sum(x[k] for x in p) for k in
               ("policy_cpu_ns","cpu_envelope_overruns","audited_intervals",
                "independent_action_replays","accepted_occurrences",
                "world_transitions")}}

def calibration(specs,cap):
    rows=[]
    for spec in specs:
        arms={"budget_rule":run_arm(spec,cap,BASE,audit=False)[0]}
        for name,arm in ARMS.items():
            arms[name]=run_arm(spec,cap,arm,audit=False)[0]
        rows.append({"arms":arms})
    metrics={name:aggregate(rows,name) for name in ARMS}
    viable=[name for name,r in metrics.items()
            if Fraction(r["priced_delta"])>0 and r["wins"]>=max(1,len(rows)//2)]
    return {"selected":max(viable,key=lambda k:(Fraction(metrics[k]["priced_delta"]),k))
            if viable else None,"candidates":metrics}

def micro(spec):
    """One cold signature per sample versus a repeated same-model stream.
    Referee-only workload; these warm results never enter actor decisions.
    """
    m=Model(spec)
    states=sorted((s for s in closure(m,[m.initial()]) if 1<=s.stage<4),
                  key=lambda s:(s.stage,s.mask,s.budget,repr(s)))
    # Deterministic sample with up to 8 states per stage.
    sample=[s for stage in (1,2,3) for s in [t for t in states if t.stage==stage][:8]]
    require(sample,"empty microbench")
    out={}
    reference=[signature(m,s,"marginal",Meter(100000)) for s in sample]
    for name,mode,opt in KERNELS:
        if mode=="literal" or name=="prune":
            continue
        for kind in ("cold","warm"):
            meter=Meter(10000000)
            projector=Projector(m,opt) if opt else None
            stream=list(zip(sample,reference))*(1 if kind=="cold" else 4)
            t0=time.process_time_ns()
            got=[]
            for s,ref in stream:
                if kind=="cold":
                    projector=Projector(m,opt) if opt else None
                got.append(signature(m,s,mode,meter,projector))
            elapsed=time.process_time_ns()-t0
            require(got==[r for _,r in stream],"microbench signature mismatch")
            out[name+"_"+kind]={"signatures":len(stream),"work":meter.used,
                                "cpu_ns":elapsed}
    return out

def campaign(out,counts=(8,12,12,12),caps=CAPS):
    require(len(counts)==4 and all(type(n) is int and n>0 for n in counts),
            "invalid counts")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"rows").mkdir()
    (out/"receipts").mkdir()
    specs={}
    seen=set()
    manifest=[]
    for (split,seed,family),n in zip(SPLITS,counts):
        ss=[make_spec(seed,i,family,split=="scarce") for i in range(n)]
        ids=[digest(s) for s in ss]
        require(len(set(ids))==len(ids) and not seen.intersection(ids),"split overlap")
        seen.update(ids)
        specs[split]=ss
        manifest.append({"split":split,"seed":seed,"family":family,"roots":ids})
    choices={str(cap):calibration(specs["calibration"],cap) for cap in caps}
    rows=[]
    index=[]
    micro_rows=[]
    for split in ("ordinary","shifted","scarce"):
        for spec in specs[split]:
            rid=digest(spec)
            micro_rows.append({"root":rid,"split":split,"results":micro(spec)})
            for cap in caps:
                choice=choices[str(cap)]["selected"]
                configs={"budget_rule":BASE,**ARMS,**FORCED,
                         "calibrated":ARMS[choice] if choice else BASE}
                arms={}
                for name,arm in configs.items():
                    result,receipts=run_arm(spec,cap,arm)
                    require(result["max_work"]<=cap,"work cap")
                    arms[name]=result
                    if receipts:
                        write_json(out/"receipts"/(rid+"-"+str(cap)+"-"+name+".json"),
                                   {"spec":spec,"cap":cap,"arm":name,"receipts":receipts})
                    if arm[-1]:
                        require(all(result[k]==arms["budget_rule"][k] for k in RAW),
                                "forced incumbent changed raw result")
                row={"root":rid,"split":split,"cap":cap,"spec":spec,"arms":arms}
                path="rows/"+rid+"-"+str(cap)+".json"
                write_json(out/path,row)
                index.append({"path":path,"sha256":digest(row)})
                rows.append(row)
    summary={}
    for split in ("ordinary","shifted","scarce"):
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            summary[split+"/"+str(cap)]={name:aggregate(group,name)
                                        for name in group[0]["arms"]}
    micro_summary={}
    for key in micro_rows[0]["results"]:
        micro_summary[key]={k:sum(r["results"][key][k] for r in micro_rows)
                            for k in ("signatures","work","cpu_ns")}
    write_json(out/"microbench.json",micro_rows)
    report={"schema_version":"distill.amortized_quotient.v1",
            "source_kind":"synthetic_finite_model","training_admission":False,
            "price_per_unit":"1/"+str(PRICE),"caps":list(caps),
            "cache_capacity":32,"manifest":manifest,"calibration":choices,
            "summary":summary,"microbench":micro_summary,"row_index":index}
    write_json(out/"report.json",report)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    report=campaign(args.out)
    import json
    print(json.dumps({k:v for k,v in report.items() if k not in ("row_index","manifest")},
                     sort_keys=True))

if __name__=="__main__":
    main()
