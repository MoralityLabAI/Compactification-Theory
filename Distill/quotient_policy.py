"""DISTILL-11 metered quotient actor; prior search and risk contract unchanged."""
from __future__ import annotations
from dataclasses import asdict
from .core import digest, require
from .program_student import Meter, Exhausted
from .program_opportunity import incumbent, reserve
from .program_cancellation import (CONFIGS,GATE_WORK,RECEIPT_WORK, mass, immediate,
    candidates,gate,late_compare,Auditor as RiskAuditor)
from .program_quotient import compare,MODES
from .student_campaign import PRICE

def certificate(m,s,b,meter,config,prior_gate_work,*,mode="marginal"):
    require(config in CONFIGS.values(),"unregistered config")
    room=meter.cap-meter.used-reserve(s)
    if room<=RECEIPT_WORK:
        return None,{"search_work":0,"exhausted":1,"trials":0,"cancelled_pairs":0,"quotient_merges":0,"expanded":0,"frontier":0}
    allowance=min(config["search_cap"],room)
    local=Meter(allowance-RECEIPT_WORK)
    trials=[]
    attempted=0
    try:
        M=mass(m,s,local)
        # A perfect-information ceiling for a final unauditable claim can
        # rule out any positive one-step investigation before building it.
        if s.stage==3 and m.terminal(b):
            v,r,_=immediate(m,s,b,local)
            possible=M*m.spec["correct_reward"]*PRICE-v
            if possible<=M*(PRICE+prior_gate_work+local.used+RECEIPT_WORK):
                choices=[]
            else:
                choices=candidates(m,s,b,local)
        else:
            choices=candidates(m,s,b,local)
        for idx,a in enumerate(choices):
            left=local.cap-local.used
            if left<256:
                break
            # Reserve each trial independently, keeping earlier certificates.
            share=min(left,max(256,allowance//2)) if idx+1<len(choices) else left
            sub=Meter(share)
            attempted+=1
            trial=None
            try:
                threshold=(prior_gate_work+local.used+RECEIPT_WORK)*M
                trial=(late_compare(m,s,a,b,sub) if s.stage==3 and m.terminal(b) else
                       compare(m,s,a,b,sub,mode,threshold=threshold))
            except Exhausted:
                pass
            local.charge(sub.used)
            if trial is not None:
                trial["trial_work"]=sub.used
                trial.setdefault("quotient_merges",0)
                trials.append(trial)
            # A complete positive certificate can end search now; no need
            # to spend remaining allowance searching for a larger gain.
            if trial is not None and trial["bounds"][0]>M*(prior_gate_work+local.used+RECEIPT_WORK) and trial["bounds"][3]<=0:
                break
    except Exhausted:
        M=None
    meter.charge(local.used+RECEIPT_WORK)
    work=local.used+RECEIPT_WORK
    stats={"search_work":work,"exhausted":int(M is None or (attempted>0 and not trials)),
           "trials":len(trials),"cancelled_pairs":sum(t["cancelled_pairs"] for t in trials),
           "quotient_merges":sum(t["quotient_merges"] for t in trials),
           "expanded":sum(t["expanded"] for t in trials),"frontier":sum(t["frontier"] for t in trials)}
    if M is None:
        return None,stats
    viable=[t for t in trials if t["bounds"][0]>M*(work+prior_gate_work) and t["bounds"][3]<=0]
    best=max(viable,key=lambda t:(t["bounds"][0],-t["bounds"][3],t["action"])) if viable else None
    body={"schema_version":"distill.quotient_opportunity.v1","spec_sha256":digest(m.spec),
          "state":asdict(s),"baseline":b,"mass":M,"price":PRICE,"config":config,"cancel":True,"quotient_mode":mode,
          "search_work":work,"prior_gate_work":prior_gate_work,"reserve":reserve(s),
          "remaining_after_search":meter.cap-meter.used,"trials":trials,
          "chosen":best["action"] if best else None,
          "net_lower_numerator":best["bounds"][0]-M*(work+prior_gate_work) if best else None,
          "risk_upper_numerator":best["bounds"][3] if best else None}
    return {**body,"receipt_id":digest(body)},stats


class Auditor(RiskAuditor):
    def verify(self,s,receipt):
        require(receipt["schema_version"]=="distill.quotient_opportunity.v1" and receipt["quotient_mode"] in MODES,"quotient receipt scope")
        return super().verify(s,receipt)

class Policy:
    def __init__(self,config=None,*,mode="marginal",forced=False):
        self.config=config
        require(mode in MODES,"unknown mode")
        self.mode=mode
        self.forced=forced
        self.attempted=False
        self.gate_spent=0
    def choose(self,m,s,meter):
        info={k:0 for k in ("gates","attempts","accepted","deviations","search_work","trials","expanded",
                              "cancelled_pairs","quotient_merges","frontier","exhausted","fallbacks")}
        try:
            b=incumbent(m,s,meter)
        except Exhausted:
            info["fallbacks"]=1
            return "defer",info,None
        if self.config is None or self.attempted:
            return b,info,None
        if meter.cap-meter.used<reserve(s)+GATE_WORK+RECEIPT_WORK+1:
            return b,info,None
        meter.charge(GATE_WORK)
        self.gate_spent+=GATE_WORK
        info["gates"]=1
        if not gate(m,s,b,self.config["gate"]):
            return b,info,None
        self.attempted=True
        info["attempts"]=1
        receipt,stats=certificate(m,s,b,meter,self.config,self.gate_spent,mode=self.mode)
        info.update(stats)
        if receipt and receipt["chosen"] is not None:
            info["accepted"]=1
            if not self.forced:
                info["deviations"]=1
                return receipt["chosen"],info,receipt
        return b,info,receipt
