"""DISTILL-17: decision-equivalent compiled incumbent and fused late certificates.

The optimized incumbent must choose exactly the same action as the frozen
budget-rule incumbent and never exceed its declared selection cost. Optional
late certificates still price the *original* future terminal selection, so
their guarantee remains conservative when the compiled continuation is used.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json

from .core import digest, require, write_json
from .research_program import Model
from .program_student import Meter, Exhausted
from .program_opportunity import incumbent, reserve
from .program_cancellation import (Auditor, candidates, gate, GATE_WORK,
    RECEIPT_WORK)
from .program_quotient import closure
from .student_campaign import PRICE
from .trajectory_memory import Actor, evaluate, aggregate, RAW, BASE
from .opportunity_campaign import make_spec

CAPS=(8192,32768)
SPLITS=(("calibration",2026120111,"shared_control"),
        ("ordinary",2026120112,"shared_control"),
        ("shifted",2026120113,"shifted_control"),
        ("scarce",2026120114,"shifted_control"))
TERMINAL_ORIGINAL=379
COMPILED_TERMINAL=184
ARMS={
    "budget_rule":BASE,
    "legacy_late":("late","late",2048,False),
    "compiled":("compiled","late",2048,False),
    "fused":("fused","late",2048,False),
    "fused_ceiling":("fused_ceiling","late",2048,False),
    "compiled_fused":("compiled_fused","late",2048,False),
    "compiled_ceiling":("compiled_ceiling","late",2048,False),
    "compiled_forced":("compiled_ceiling","late",2048,True),
}
OPTIONAL=("compiled_fused","compiled_ceiling")


def terminal(zero,one,cr,wr):
    # Same lexicographic tie rule as student_campaign._terminal.
    return max((0,"defer",0),
               (zero*cr+one*wr,"recommend:0",one),
               (one*cr+zero*wr,"recommend:1",zero))


def compiled_incumbent(m,s,meter):
    """Exact specialization of cheap(reuse_first) followed by budget_rule.

    Charge 16 for legality, 8 for branch tests and 16 for the budget rule.
    In uncertified terminal cases charge two 64-slot mass acquisitions and
    16 units for reward arithmetic/tie selection. No oracle or lookup table.
    """
    meter.charge(16)
    legal=m.legal(s)
    meter.charge(8+16)
    if s.cert>=0:
        a="certify:"+str(s.cert)
        require(a in legal,"invalid certificate action")
        return a
    if s.stage==0 and s.anchor<0 and "direct_audit" in legal:
        return "direct_audit"
    if s.stage<=1 and s.anchor>=0 and s.law<0 and "law_audit" in legal:
        if s.budget>=m.cost("law_audit")+m.cost("transport"):
            return "law_audit"
        # budget_rule substitutes cheap(direct_first), which never transports.
        if "direct_audit" in legal:
            return "direct_audit"
    elif "transport" in legal:
        return "transport"
    elif "direct_audit" in legal:
        return "direct_audit"
    meter.charge(64)
    M=m.mass(s.mask)
    meter.charge(64)
    one=m.mass(s.mask&m.truth_masks[s.stage])
    meter.charge(16)
    return terminal(M-one,one,m.spec["correct_reward"],m.spec["wrong_reward"])[1]


def base_table(m,s,b,meter):
    meter.charge(64)
    M=m.mass(s.mask)
    meter.charge(64)
    one=m.mass(s.mask&m.truth_masks[3])
    meter.charge(16)
    z=M-one
    v,a,r=terminal(z,one,m.spec["correct_reward"],m.spec["wrong_reward"])
    require(a==b,"late incumbent tie mismatch")
    return M,one,v,r


def fused_trial(m,s,a,table,meter):
    """Exact final-stage experiment value/risk from one parent and two joint masses.

    The branch outcome 0 masses are complements, so no branch state needs
    construction or separate mass scan. Charge two 64-slot acquisitions,
    mask extraction and bounded arithmetic. Future work is the frozen 379.
    """
    M,one,base_v,base_r=table
    meter.charge(8)
    part=s.mask&{"screen":m.screen_masks[3],"control":m.nuisance_mask,
                 "law_audit":m.law_mask}[a]
    meter.charge(64)
    p=m.mass(part)
    meter.charge(64)
    q=m.mass(part&m.truth_masks[3])
    meter.charge(32)
    cr,wr=m.spec["correct_reward"],m.spec["wrong_reward"]
    v1,_,r1=terminal(p-q,q,cr,wr)
    v0,_,r0=terminal((M-p)-(one-q),one-q,cr,wr)
    return ((v0+v1-base_v-m.cost(a)*M)*PRICE-TERMINAL_ORIGINAL*M,
            r0+r1-base_r)


class FusedActor(Actor):
    def choose(self,s,meter):
        m=self.m
        use_compiled=self.kernel.startswith("compiled")
        try:
            b=compiled_incumbent(m,s,meter) if use_compiled else incumbent(m,s,meter)
        except Exhausted:
            self.stats["fallbacks"]+=1
            return "defer",None
        if self.kernel=="compiled" or self.accepted or s.stage in self.tried:
            return b,None
        room=meter.cap-meter.used-reserve(s)
        if room<=GATE_WORK+RECEIPT_WORK+160:
            return b,None
        meter.charge(GATE_WORK)
        self.optional+=GATE_WORK
        self.stats["gates"]+=1
        if not gate(m,s,b,"late"):
            return b,None
        self.tried.add(s.stage)
        self.stats["attempts"]+=1
        allowance=min(self.allowance,meter.cap-meter.used-reserve(s))
        local=Meter(allowance-RECEIPT_WORK)
        before=self.optional
        # Always acquire parent table before recording a certificate.
        table=base_table(m,s,b,local)
        M,_,base_v,_=table
        trials=[]
        exhausted=False
        try:
            for a in candidates(m,s,b,local):
                if self.kernel in ("fused_ceiling","compiled_ceiling"):
                    local.charge(8)
                    # Perfect-information upper bound, ignoring risk.
                    hi=(M*(m.spec["correct_reward"]-m.cost(a))-base_v)*PRICE-TERMINAL_ORIGINAL*M
                    if hi<=M*(before+local.used+RECEIPT_WORK):
                        continue
                v,r=fused_trial(m,s,a,table,local)
                trials.append({"action":a,"value_delta":v,"risk_delta":r})
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
        viable=[t for t in trials if t["value_delta"]>M*self.optional and t["risk_delta"]<=0]
        best=max(viable,key=lambda t:(t["value_delta"],-t["risk_delta"],t["action"])) if viable else None
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


def check_compiler(spec,check_trials=True):
    m=Model(spec)
    ref=Auditor(m) if check_trials else None
    n=old_work=new_work=strict=trials=0
    for s in closure(m,[m.initial()],all_actions=True):
        if s.stage==4:
            continue
        a,b=Meter(100000),Meter(100000)
        base=incumbent(m,s,a)
        new=compiled_incumbent(m,s,b)
        require(base==new and b.used<=a.used,"compiler parity/work failure")
        n+=1
        old_work+=a.used
        new_work+=b.used
        strict+=int(b.used<a.used)
        if ref and gate(m,s,base,"late"):
            require(a.used==TERMINAL_ORIGINAL and b.used==COMPILED_TERMINAL,"terminal cost mismatch")
            table=base_table(m,s,base,Meter(100000))
            bv,br=ref.action(s,base)
            for opt in candidates(m,s,base,Meter(100000)):
                av,ar=ref.action(s,opt)
                require(fused_trial(m,s,opt,table,Meter(100000))==(av-bv,ar-br),"fused contingency mismatch")
                trials+=1
    if ref:
        ref.close()
    return {"states":n,"original_selection_work":old_work,"compiled_selection_work":new_work,
            "strictly_cheaper_states":strict,"fused_trials":trials}


def run(spec,cap,name,do_audit=True):
    cfg=ARMS[name]
    factory=Actor if name in ("budget_rule","legacy_late") else FusedActor
    return evaluate(spec,cap,cfg,do_audit,actor_factory=factory)


def comparison(rows,name,baseline):
    diffs=[Fraction(r["arms"][name]["priced_utility"])-Fraction(r["arms"][baseline]["priced_utility"]) for r in rows]
    risks=[Fraction(r["arms"][name]["risk"])-Fraction(r["arms"][baseline]["risk"]) for r in rows]
    return {"priced_delta":str(sum(diffs)/len(rows)),
            "risk_delta":str(sum(risks)/len(rows)),
            "wins":sum(x>0 for x in diffs),"ties":sum(x==0 for x in diffs),
            "losses":sum(x<0 for x in diffs),"risk_increased_roots":sum(x>0 for x in risks)}


def calibrate(ss,cap):
    rows=[]
    for spec in ss:
        arms={k:run(spec,cap,k,False)[0] for k in ("compiled",*OPTIONAL)}
        rows.append({"arms":arms})
    metrics={k:comparison(rows,k,"compiled") for k in OPTIONAL}
    viable=[k for k,x in metrics.items() if Fraction(x["priced_delta"])>0 and
            x["wins"]>=max(1,len(rows)//2) and x["risk_increased_roots"]==0]
    return {"selected":max(viable,key=lambda k:(Fraction(metrics[k]["priced_delta"]),k)) if viable else None,
            "candidates":metrics}


def campaign(out,counts=(8,12,12,12),caps=CAPS):
    require(len(counts)==4 and min(counts)>0,"bad split counts")
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
    diagnostics=[]
    for split,_,_ in SPLITS[1:]:
        for spec in roots[split]:
            rid=digest(spec)
            d=check_compiler(spec)
            diagnostics.append({"root":rid,"split":split,**d})
            for cap in caps:
                arms={}
                for name in ARMS:
                    result,receipts=run(spec,cap,name)
                    arms[name]=result
                    if name in ("compiled","compiled_forced"):
                        require(all(result[k]==arms["budget_rule"][k] for k in RAW),"raw outcome changed")
                    if receipts:
                        write_json(out/"receipts"/(rid+"-"+str(cap)+"-"+name+".json"),receipts)
                selected=cal[str(cap)]["selected"] or "compiled"
                arms["calibrated_compiled"]=arms[selected]
                require(Fraction(arms["compiled"]["work"])<=Fraction(arms["budget_rule"]["work"]),"compiled work increased")
                row={"root":rid,"spec":spec,"split":split,"cap":cap,"arms":arms}
                write_json(out/"rows"/(rid+"-"+str(cap)+".json"),row)
                rows.append(row)
    summary={}
    for split,_,_ in SPLITS[1:]:
        for cap in caps:
            group=[r for r in rows if r["split"]==split and r["cap"]==cap]
            summary[split+"/"+str(cap)]={name:{**aggregate(group,name),
                    "increment_vs_compiled":comparison(group,name,"compiled")}
                for name in group[0]["arms"]}
    totals={k:sum(d[k] for d in diagnostics) for k in ("states","original_selection_work",
                "compiled_selection_work","strictly_cheaper_states","fused_trials")}
    report={"schema_version":"distill.compiled_incumbent.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"price_per_unit":"1/"+str(PRICE),"caps":list(caps),
            "compiler_diagnostics":totals,"calibration":cal,"summary":summary}
    write_json(out/"diagnostics.json",diagnostics)
    write_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    print(json.dumps(campaign(args.out),sort_keys=True))


if __name__=="__main__":
    main()
