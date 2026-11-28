"""DISTILL-16: trajectory-local exact opportunity memory.

All optional searches are charged, including rejected and interrupted trials.
One accepted deviation at most, followed by the fixed incumbent continuation.
Persistent caches never cross a simulated world/episode boundary.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import time

from .core import digest, require, write_json
from .research_program import Model
from .program_student import Meter, Exhausted
from .program_opportunity import incumbent, reserve
from .program_cancellation import (immediate, children, candidates, gate,
    late_compare, Auditor, GATE_WORK, RECEIPT_WORK)
from .program_oracle import actual_transition
from .lazy_certificate_memory import Memo
from .opportunity_campaign import make_spec
from .student_campaign import PRICE

CAPS=(8192,32768)
SPLITS=(("calibration",2026120101,"shared_control"),
        ("ordinary",2026120102,"shared_control"),
        ("shifted",2026120103,"shifted_control"),
        ("scarce",2026120104,"shifted_control"))
CACHE=32
KERNELS=("cold","recursive_literal","recursive_quotient","recursive_hybrid")
CONFIGS={name+"_"+gate_name:(name,gate_name,cap,False)
         for name in KERNELS for gate_name,cap in (("tail",4096),("broad",8192))}
CONFIGS["late"]=("late","late",2048,False)
FORCED={name+"_forced_"+g:(name,g,cap,True)
        for name in ("recursive_literal","recursive_hybrid")
        for g,cap in (("tail",4096),("broad",8192))}
BASE=(None,"late",2048,False)
RAW=("utility","risk","actions","credits","certified","transports","deferred")


def action_value(m,s,a,memo,meter):
    v,r,_=immediate(m,s,a,meter)
    for t in children(m,s,a,meter):
        x,y,_=memo.query(t,meter)
        v+=x
        r+=y
    return v,r


class Actor:
    def __init__(self,m,config,capacity=CACHE):
        self.m=m
        self.kernel,self.kind,self.allowance,self.forced=config
        self.memo=Memo(m,self.kernel if self.kernel in KERNELS else "cold",capacity)
        self.accepted=False
        self.tried=set()
        self.optional=0
        self.stats=defaultdict(int)

    def choose(self,s,meter):
        m=self.m
        try:
            b=incumbent(m,s,meter)
        except Exhausted:
            self.stats["fallbacks"]+=1
            return "defer",None
        if self.kernel is None or self.accepted or s.stage in self.tried:
            return b,None
        room=meter.cap-meter.used-reserve(s)
        if room<=GATE_WORK+RECEIPT_WORK+64:
            return b,None
        meter.charge(GATE_WORK)
        self.optional+=GATE_WORK
        self.stats["gates"]+=1
        if not gate(m,s,b,self.kind):
            return b,None
        self.tried.add(s.stage)
        self.stats["attempts"]+=1
        allowance=min(self.allowance,meter.cap-meter.used-reserve(s))
        local=Meter(allowance-RECEIPT_WORK)
        trials=[]
        before=self.optional
        # Acquire compatible mass only after charging its metered scan.
        local.charge(64)
        M=m.mass(s.mask)
        exhausted=False
        try:
            opts=candidates(m,s,b,local)
            if self.kernel!="late" and opts:
                base=action_value(m,s,b,self.memo,local)
            for a in opts:
                if self.kernel=="late":
                    res=late_compare(m,s,a,b,local)
                    lo,hi,rl,rh=res["bounds"]
                    require(lo==hi and rl==rh,"nonexact late bound")
                    v,r=lo,rl
                else:
                    av,ar=action_value(m,s,a,self.memo,local)
                    v,r=av-base[0],ar-base[1]
                trials.append({"action":a,"value_delta":v,"risk_delta":r})
                # End promptly after a profitable, safe exact comparison.
                if v>M*(before+local.used+RECEIPT_WORK) and r<=0:
                    break
        except Exhausted:
            exhausted=True
        work=local.used+RECEIPT_WORK
        meter.charge(work)
        self.optional+=work
        self.stats["search_work"]+=work
        self.stats["trials"]+=len(trials)
        self.stats["exhausted"]+=int(exhausted)
        valid=[t for t in trials if t["value_delta"]>M*self.optional and t["risk_delta"]<=0]
        best=max(valid,key=lambda t:(t["value_delta"],-t["risk_delta"],t["action"])) if valid else None
        rec={"stage":s.stage,"baseline":b,"mass":M,"prior_optional":before,
             "search_work":work,"cumulative_optional":self.optional,
             "trials":trials,"chosen":best["action"] if best else None,
             "remaining":meter.cap-meter.used,"reserve":reserve(s),
             "exhausted":exhausted}
        if best:
            self.accepted=True
            self.stats["accepted"]+=1
            if not self.forced:
                self.stats["deviations"]+=1
                return best["action"],rec
        return b,rec


def audit_receipt(ref,s,rec):
    m=ref.m
    M=m.mass(s.mask)
    b=incumbent(m,s,Meter(100000))
    require(rec["baseline"]==b and rec["mass"]==M,"receipt scope")
    require(rec["remaining"]>=rec["reserve"]==reserve(s),"continuation reserve")
    require(rec["cumulative_optional"]==rec["prior_optional"]+rec["search_work"],"optional cost accounting")
    bv,br=ref.action(s,b)
    viable=[]
    for t in rec["trials"]:
        a=t["action"]
        require(a in m.legal(s) and a!=b,"illegal trial")
        v,r=ref.action(s,a)
        require((t["value_delta"],t["risk_delta"])==(v-bv,r-br),"wrong exact comparison")
        if v-bv>M*rec["cumulative_optional"] and r-br<=0:
            viable.append(t)
    best=max(viable,key=lambda t:(t["value_delta"],-t["risk_delta"],t["action"])) if viable else None
    require(rec["chosen"]==(best["action"] if best else None),"incorrect certificate choice")
    return best


def evaluate(spec,cap,config,do_audit=True,actor_factory=Actor):
    m=Model(spec)
    ref=Auditor(m) if do_audit else None
    total=defaultdict(int)
    counts=defaultdict(int)
    cpu=0
    receipts=[]
    accepted_seen=set()
    peak=0
    for i,w in enumerate(m.worlds):
        actor=actor_factory(m,config)
        meter=Meter(cap)
        s=m.initial()
        c=defaultdict(int)
        while s.stage<4:
            require(c["actions"]<32,"nonterminating actor")
            start=time.process_time_ns()
            a,rec=actor.choose(s,meter)
            cpu+=time.process_time_ns()-start
            require(a in m.legal(s),"illegal actor choice")
            if rec is not None:
                counts["receipts"]+=1
                counts["audited_trials"]+=len(rec["trials"]) if ref else 0
                if ref:
                    best=audit_receipt(ref,s,rec)
                    if best:
                        key=(s,best["action"])
                        if key not in accepted_seen:
                            av,ar=ref.explicit(s,best["action"])
                            bv,br=ref.explicit(s,rec["baseline"])
                            require((av-bv,ar-br)==(best["value_delta"],best["risk_delta"]),"explicit replay mismatch")
                            accepted_seen.add(key)
                if rec["chosen"] is not None:
                    counts["accepted_occurrences"]+=1
                    require(a==(rec["baseline"] if config[-1] else rec["chosen"]),"action receipt mismatch")
                if ref:
                    receipts.append({"world":i,"state":s.__dict__,"receipt":rec})
            _,t,reward=actual_transition(m,s,a,i)
            c["utility"]+=reward
            c["risk"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            c["credits"]+=m.cost(a)
            c["actions"]+=1
            c["certified"]+=int(a.startswith("certify:"))
            c["transports"]+=int(a=="transport")
            c["deferred"]+=int(a=="defer")
            s=t
            counts["transitions"]+=1
        require(meter.used<=cap and c["credits"]<=spec["budget"],"cap violation")
        require(actor.stats["deviations"]<=1 and actor.stats["accepted"]<=1,"single deviation")
        if actor.stats["attempts"]:
            require(actor.stats["fallbacks"]==0,"reserved incumbent exhausted")
        require(actor.optional<=meter.used,"optional work exceeds meter")
        c["work"]=meter.used
        c["optional_work"]=actor.optional
        for k,v in actor.stats.items():
            c[k]+=v
        for k in ("root_hits","recursive_hits","different_state_hits","different_mask_hits",
                  "writes","evictions","expansions","key_work","admin_work"):
            c["cache_"+k]=actor.memo.stats[k]
        peak=max(peak,actor.memo.peak)
        for k,v in c.items():
            total[k]+=w.mass*v
    M=m.mass(m.full_mask)
    fields=(*RAW,"work","optional_work","gates","attempts","accepted","deviations",
            "search_work","trials","exhausted","fallbacks",
            *("cache_"+k for k in ("root_hits","recursive_hits","different_state_hits","different_mask_hits",
                                   "writes","evictions","expansions","key_work","admin_work")))
    result={k:str(Fraction(total[k],M)) for k in fields}
    result["priced_utility"]=str(Fraction(PRICE*total["utility"]-total["work"],PRICE*M))
    result.update({k:counts[k] for k in ("receipts","audited_trials","accepted_occurrences","transitions")})
    result.update(cpu_ns=cpu,cache_peak=peak,independent_replays=ref.replays if ref else 0)
    if ref:
        ref.close()
    return result,receipts


def aggregate(rows,name):
    arms=[r["arms"][name] for r in rows]
    base=[r["arms"]["budget_rule"] for r in rows]
    delta=[Fraction(x["priced_utility"])-Fraction(y["priced_utility"]) for x,y in zip(arms,base)]
    risks=[Fraction(x["risk"])-Fraction(y["risk"]) for x,y in zip(arms,base)]
    require(all(x<=0 for x in risks),"increased expected risk")
    n=len(rows)
    fields=[k for k,v in arms[0].items() if isinstance(v,str)]
    return {**{k:str(sum(Fraction(x[k]) for x in arms)/n) for k in fields},
            "priced_delta":str(sum(delta)/n),"risk_delta":str(sum(risks)/n),
            "wins":sum(x>0 for x in delta),"ties":sum(x==0 for x in delta),"losses":sum(x<0 for x in delta),
            "roots":n,"cache_peak":max(x["cache_peak"] for x in arms),
            **{k:sum(x[k] for x in arms) for k in ("cpu_ns","receipts","audited_trials","accepted_occurrences","transitions","independent_replays")}}


def calibrate(ss,cap):
    rows=[]
    for spec in ss:
        arms={"budget_rule":evaluate(spec,cap,BASE,False)[0]}
        for name,config in CONFIGS.items():
            arms[name]=evaluate(spec,cap,config,False)[0]
        rows.append({"arms":arms})
    out={k:aggregate(rows,k) for k in CONFIGS}
    viable=[k for k,r in out.items() if Fraction(r["priced_delta"])>0 and r["wins"]>=max(1,len(rows)//2)]
    return {"selected":max(viable,key=lambda k:(Fraction(out[k]["priced_delta"]),k)) if viable else None,
            "candidates":out}


def campaign(out,counts=(8,12,12,12),caps=CAPS):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"rows").mkdir()
    (out/"receipts").mkdir()
    roots={}
    seen=set()
    for (split,seed,family),n in zip(SPLITS,counts):
        ss=[make_spec(seed,i,family,split=="scarce") for i in range(n)]
        ids=[digest(s) for s in ss]
        require(len(set(ids))==len(ids) and not seen.intersection(ids),"split overlap")
        seen.update(ids)
        roots[split]=ss
    cal={str(cap):calibrate(roots["calibration"],cap) for cap in caps}
    rows=[]
    index=[]
    for split,_,_ in SPLITS[1:]:
        for spec in roots[split]:
            rid=digest(spec)
            for cap in caps:
                choice=cal[str(cap)]["selected"]
                configs={"budget_rule":BASE,**CONFIGS,**FORCED,
                         "calibrated":CONFIGS[choice] if choice else BASE}
                arms={}
                for name,config in configs.items():
                    result,receipts=evaluate(spec,cap,config)
                    arms[name]=result
                    if config[-1]:
                        require(all(result[k]==arms["budget_rule"][k] for k in RAW),"forced incumbent changed raw outcomes")
                    if receipts:
                        # One file per root/cap/arm; at most 64*3 receipts.
                        require(len(receipts)<=192,"receipt count")
                        write_json(out/"receipts"/(rid+"-"+str(cap)+"-"+name+".json"),receipts)
                row={"root":rid,"spec":spec,"split":split,"cap":cap,"arms":arms}
                name="rows/"+rid+"-"+str(cap)+".json"
                write_json(out/name,row)
                index.append({"path":name,"sha256":digest(row)})
                rows.append(row)
    summary={}
    for split,_,_ in SPLITS[1:]:
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            summary[split+"/"+str(cap)]={name:aggregate(group,name) for name in group[0]["arms"]}
    report={"schema_version":"distill.trajectory_memory.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"cache_capacity":CACHE,"caps":list(caps),
            "price_per_unit":"1/"+str(PRICE),"calibration":cal,"summary":summary,"row_index":index}
    write_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    r=campaign(args.out)
    print(json.dumps({k:v for k,v in r.items() if k!="row_index"},sort_keys=True))


if __name__=="__main__":
    main()
