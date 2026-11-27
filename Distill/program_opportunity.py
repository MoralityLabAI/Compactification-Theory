"""DISTILL-09: bounded certificates of incumbent-relative research value.

All bounds refer to the finite public-belief model and the exact budget-rule
continuation, with a declared logical-work price. No learned or hidden-world
inputs. The actor never consults the referee's exact continuation evaluator.
"""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from .core import digest, require
from .program_student import Meter, Exhausted
from .program_residual import budget_rule
from .student_campaign import cheap, PRICE
from .program_oracle import actual_transition

GATE_WORK=16
NONTERMINAL_WORK=64
TERMINAL_WORK=468
CONFIGS={
    "late":{"gate":"late","search_cap":2048,"depth":1},
    "bounded":{"gate":"tail","search_cap":4096,"depth":5},
    "broad":{"gate":"broad","search_cap":8192,"depth":9},
}

def incumbent(m,s,meter):
    b=cheap(m,s,meter,"reuse_first")
    return budget_rule(m,s,b,meter)

def reserve(s):
    # Each experiment costs >=1 credit. The budget rule costs <=64 metered
    # units when it chooses an experiment and <=468 when it advances a project.
    return NONTERMINAL_WORK*s.budget+TERMINAL_WORK*(4-s.stage)

def _charge_mass(m,mask,meter):
    meter.charge(64)
    return m.mass(mask)

def _transition(m,s,a,meter):
    meter.charge(64)
    branches=m.branches(s,a)
    meter.charge(16*len(branches))
    return branches

def _reward(m,s,a,meter):
    meter.charge(65+(64 if a.startswith("recommend:") else 0))
    return m.reward_sum(s,a)

class Bounds:
    """Intervals in integer units of mass * (PRICE * raw utility - work).

    At a cutoff, every incumbent experimental credit costs at most the
    remaining budget, each terminal decision has nonnegative conditional
    expected raw reward, and total future selection work is bounded by reserve.
    The upper bound ignores experiment and planner costs. Branch masses are
    not divided until the final certificate.
    """
    def __init__(self,m,meter):
        self.m,self.meter=m,meter
        self.cache={}
        self.expanded=0
        self.cutoffs=0
        self.cache_hits=0
    def tail(self,s,depth):
        if s.stage==4:
            return 0,0
        self.meter.charge(16)
        key=(s,depth)
        if key in self.cache:
            self.cache_hits+=1
            return self.cache[key]
        mass=_charge_mass(self.m,s.mask,self.meter)
        if depth==0:
            self.cutoffs+=1
            return (-(s.budget*PRICE+reserve(s))*mass,
                    (4-s.stage)*self.m.spec["certificate_reward"]*PRICE*mass)
        start=self.meter.used
        a=incumbent(self.m,s,self.meter)
        future_work=self.meter.used-start
        require(future_work<=(TERMINAL_WORK if self.m.terminal(a) else NONTERMINAL_WORK),
                "incumbent work bound violated")
        lo,hi=self.action(s,a,depth-1)
        self.expanded+=1
        result=(lo-future_work*mass,hi-future_work*mass)
        self.cache[key]=result
        return result
    def action(self,s,a,depth):
        # The current action selection is sunk/common. All future incumbent
        # selections are counted in the continuation objective.
        r=_reward(self.m,s,a,self.meter)*PRICE
        lo=hi=r
        for _,child in _transition(self.m,s,a,self.meter):
            x,y=self.tail(child,depth)
            lo+=x
            hi+=y
        return lo,hi

def tail_exact(m,s,base,meter):
    """Last-project special case: at most one experiment, then terminal.

    With no current certificate and no affordable direct audit, an ordinary
    information experiment cannot create a certificate or make an audit
    affordable. The incumbent then selects from recommend/defer. Compute
    the exact weighted maximum from the two truth masses per observation.
    """
    require(s.stage==3 and s.cert<0 and m.terminal(base),"not a late opportunity")
    meter.charge(16)
    legal=m.legal(s)
    require("direct_audit" not in legal,"late shortcut with affordable audit")
    mass=_charge_mass(m,s.mask,meter)
    baseline=_reward(m,s,base,meter)*PRICE
    best=None
    trials=[]
    cr,wr=m.spec["correct_reward"],m.spec["wrong_reward"]
    # cheap reuse-first: 24 fixed + 16 terminal scan + 2*129 + 65;
    # budget_rule adds 16. No certificates, direct audits, or transfers.
    terminal_work=379
    for a in ("screen","control","law_audit"):
        if a not in legal:
            continue
        try:
            meter.charge(16)
            val=-m.cost(a)*mass*PRICE
            for _,child in _transition(m,s,a,meter):
                require(child.cert<0 and "direct_audit" not in m.legal(child),"invalid late shortcut")
                cm=_charge_mass(m,child.mask,meter)
                good=_charge_mass(m,child.mask&m.truth_masks[3],meter)
                meter.charge(16)
                reward=max(0,good*cr+(cm-good)*wr,(cm-good)*cr+good*wr)
                val+=reward*PRICE-terminal_work*cm
            trials.append((a,val,val))
            if best is None or (val,a)>(best[1],best[0]):
                best=(a,val,val)
        except Exhausted:
            break
    return mass,baseline,baseline,best,trials,{"expanded":0,"cutoffs":0,"cache_hits":0}

def bounded(m,s,base,meter,depth):
    meter.charge(16)
    legal=m.legal(s)
    mass=_charge_mass(m,s.mask,meter)
    engine=Bounds(m,meter)
    baseline=engine.action(s,base,depth)
    trials=[]
    # Prefer cheap informative experiments, then terminal alternatives.
    choices=sorted((a for a in legal if a!=base),key=lambda a:(m.terminal(a),m.cost(a),a))
    best=None
    for a in choices:
        try:
            meter.charge(8)
            lo,hi=engine.action(s,a,depth)
            trials.append((a,lo,hi))
            if best is None or (lo,a)>(best[1],best[0]):
                best=(a,lo,hi)
        except Exhausted:
            break
    return mass,*baseline,best,trials,{"expanded":engine.expanded,"cutoffs":engine.cutoffs,"cache_hits":engine.cache_hits}

def gate(m,s,base,kind):
    if s.cert>=0 or s.budget==0:
        return False
    late=s.stage==3 and m.terminal(base) and s.budget<m.cost("direct_audit")
    if kind=="late":
        return late
    if s.stage==3:
        return late
    if kind=="tail":
        return s.stage>=2 and (m.terminal(base) or base=="direct_audit")
    require(kind=="broad","unknown gate")
    return s.stage>=1

def certificate(m,s,base,meter,config,prior_gate_work=0):
    """Search at most the configured logical allowance; reserve continuation.

    The common incumbent selection was paid before this call. Returned
    intervals include all future incumbent work. Cost of searching/checking
    is subtracted once. A positive accepted bound remains valid when the
    rest of the program actually uses the budget-rule continuation.
    """
    require(config in CONFIGS.values(),"unregistered search config")
    begin=meter.used
    room=meter.cap-meter.used-reserve(s)
    if room<=32:
        return None,{"search_work":0,"trials":0,"cutoffs":0,"exhausted":1}
    local=Meter(min(config["search_cap"],room)-32)
    result=None
    try:
        if s.stage==3 and m.terminal(base):
            result=tail_exact(m,s,base,local)
        else:
            result=bounded(m,s,base,local,config["depth"])
    except Exhausted:
        pass
    meter.charge(local.used+32)
    work=meter.used-begin
    stats={"search_work":work,"trials":0,"cutoffs":0,"exhausted":int(result is None)}
    if result is None:
        return None,stats
    mass,blo,bhi,best,trials,counters=result
    stats.update(trials=len(trials),cutoffs=counters["cutoffs"],expanded=counters["expanded"],cache_hits=counters["cache_hits"])
    # The receipt stores all completed interval trials, including rejected
    # alternatives. The final decision uses the entire charged search cost.
    body={"schema_version":"distill.opportunity.v1","spec_sha256":digest(m.spec),
          "state":asdict(s),"baseline":base,"baseline_bounds":[blo,bhi],
          "trials":[{"action":a,"bounds":[lo,hi]} for a,lo,hi in trials],
          "mass":mass,"price":PRICE,"search_work":work,"prior_gate_work":prior_gate_work,
          "reserve":reserve(s),"remaining_after_search":meter.cap-meter.used,
          "config":config,"chosen":None,"net_lower_numerator":None}
    if best is not None:
        a,lo,hi=best
        margin=lo-bhi-(work+prior_gate_work)*mass
        if margin>0:
            body["chosen"]=a
            body["net_lower_numerator"]=margin
    return {**body,"receipt_id":digest(body)},stats

class Auditor:
    """Exact evaluator-only continuation with independent world replay.

    The exact signed value is checked for every completed interval; it is
    unavailable to the actor. Cache is local to the evaluation root.
    """
    def __init__(self,m):
        self.m=m
        self.checks=0
        @lru_cache(None)
        def tail(s):
            if s.stage==4:
                return 0
            meter=Meter(100000)
            a=incumbent(m,s,meter)
            require(meter.used<=(TERMINAL_WORK if m.terminal(a) else NONTERMINAL_WORK),"reserve bound")
            return self.action(s,a)-meter.used*m.mass(s.mask)
        self.tail=tail
    def action(self,s,a):
        m=self.m
        return PRICE*m.reward_sum(s,a)+sum(self.tail(t) for _,t in m.branches(s,a))
    def verify(self,s,receipt):
        m=self.m
        require(receipt["receipt_id"]==digest({k:v for k,v in receipt.items() if k!="receipt_id"}),"receipt digest")
        require(receipt["spec_sha256"]==digest(m.spec) and receipt["state"]==asdict(s)
                and receipt["mass"]==m.mass(s.mask) and receipt["price"]==PRICE,"receipt binding")
        require(receipt["reserve"]==reserve(s) and receipt["remaining_after_search"]>=reserve(s),"future work not reserved")
        meter=Meter(100000)
        require(receipt["baseline"]==incumbent(m,s,meter),"wrong incumbent")
        base=self.action(s,receipt["baseline"])
        lo,hi=receipt["baseline_bounds"]
        require(lo<=base<=hi,"invalid baseline interval")
        self.checks+=1
        for trial in receipt["trials"]:
            val=self.action(s,trial["action"])
            lo,hi=trial["bounds"]
            require(lo<=val<=hi,"invalid alternative interval")
            self.checks+=1
        chosen=receipt["chosen"]
        if chosen is not None:
            row=next(r for r in receipt["trials"] if r["action"]==chosen)
            lower=row["bounds"][0]-receipt["baseline_bounds"][1]-(receipt["search_work"]+receipt["prior_gate_work"])*receipt["mass"]
            exact=self.action(s,chosen)-base-(receipt["search_work"]+receipt["prior_gate_work"])*receipt["mass"]
            require(lower==receipt["net_lower_numerator"] and 0<lower<=exact,"invalid positive certificate")
        return True
    def replay_action(self,s,a):
        """Check weighted priced value via explicit world-by-world rollout."""
        m=self.m
        total=0
        for i,w in enumerate(m.worlds):
            if not s.mask>>i&1:
                continue
            st=s
            _,st,reward=actual_transition(m,st,a,i)
            val=reward*PRICE
            while st.stage<4:
                meter=Meter(100000)
                choice=incumbent(m,st,meter)
                _,st,reward=actual_transition(m,st,choice,i)
                val+=reward*PRICE-meter.used
            total+=w.mass*val
        require(total==self.action(s,a),"independent priced rollout mismatch")
        return total
    def close(self):
        self.tail.cache_clear()

class Policy:
    def __init__(self,config=None,*,forced=False):
        require(config is None or config in CONFIGS.values(),"unknown opportunity config")
        self.config=config
        self.forced=forced
        self.attempted=False
        self.gate_spent=0
    def choose(self,m,s,meter):
        info={"gates":0,"attempts":0,"accepted":0,"deviations":0,"search_work":0,
              "trials":0,"cutoffs":0,"expanded":0,"cache_hits":0,"fallbacks":0,"exhausted":0}
        try:
            b=incumbent(m,s,meter)
        except Exhausted:
            info["fallbacks"]=1
            return "defer",info,None
        if self.config is None or self.attempted:
            return b,info,None
        # Never spend optional work if doing so could invalidate the
        # continuation reserve. This arithmetic admission is covered by gate.
        if meter.cap-meter.used<reserve(s)+GATE_WORK+33:
            return b,info,None
        meter.charge(GATE_WORK)
        self.gate_spent+=GATE_WORK
        info["gates"]=1
        if not gate(m,s,b,self.config["gate"]):
            return b,info,None
        self.attempted=True
        info["attempts"]=1
        receipt,stats=certificate(m,s,b,meter,self.config,self.gate_spent)
        info.update(stats)
        if receipt and receipt["chosen"] is not None:
            info["accepted"]=1
            if not self.forced:
                info["deviations"]=1
                return receipt["chosen"],info,receipt
        return b,info,receipt
