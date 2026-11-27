"""DISTILL-10: signed continuation frontier and dual utility/risk certificates.

Every term is an unnormalized compatible-world sum. A state includes its
complete belief mask, stage, remaining experimental budget and audit flags.
Cancellation uses exact state equality only. No actual-world id or oracle
value enters actor decisions. All optional logical work is metered.
"""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from .core import digest, require
from .program_student import Meter, Exhausted
from .program_opportunity import incumbent, reserve, NONTERMINAL_WORK, TERMINAL_WORK
from .program_oracle import actual_transition
from .student_campaign import PRICE

GATE_WORK=16
# Fixed charges for frontier administration in addition to inherited scans.
SCAN_WORK=8
TERM_WORK=16
RECEIPT_WORK=64
CONFIGS={
    "late":{"gate":"late","search_cap":2048},
    "tail":{"gate":"tail","search_cap":4096},
    "broad":{"gate":"broad","search_cap":8192},
}

def order(s):
    # Every successor has strictly lower rank. Processing maximal rank
    # permits reconvergence before expanding a shared continuation.
    return ((4-s.stage)*16+s.budget,s.mask,s.screened,s.control,s.law,s.anchor,s.cert)

def mass(m,s,meter):
    meter.charge(64)
    return m.mass(s.mask)

def immediate(m,s,a,meter):
    M=mass(m,s,meter)
    meter.charge(65+(64 if a.startswith("recommend:") else 0))
    raw=m.reward_sum(s,a)
    risk=0
    if a.startswith("recommend:"):
        # Recover exact false mass without an additional posterior scan.
        numer=m.spec["correct_reward"]*M-raw
        den=m.spec["correct_reward"]-m.spec["wrong_reward"]
        risk,rem=divmod(numer,den)
        require(rem==0 and 0<=risk<=M,"invalid false mass")
    return raw*PRICE,risk,M

def children(m,s,a,meter):
    meter.charge(64)
    rows=m.branches(s,a)
    meter.charge(16*len(rows))
    require(all(order(t)[0]<order(s)[0] for _,t in rows),"nondecreasing continuation rank")
    return [t for _,t in rows if t.stage<4]

def cutoff(m,s,meter):
    M=mass(m,s,meter)
    r=4-s.stage
    # Incumbent terminal choices have nonnegative conditional expected raw
    # payoff. Experiments consume <=b; future work is bounded by reserve.
    # Expected false recommendation count at a terminal choice is bounded
    # by cr/(cr-wr), since recommendation has nonnegative expected reward.
    cr,wr=m.spec["correct_reward"],m.spec["wrong_reward"]
    return (-(s.budget*PRICE+reserve(s))*M,r*m.spec["certificate_reward"]*PRICE*M,
            0,(r*M*cr)//(cr-wr))

def signed_interval(c,b):
    lo,hi,rl,rh=b
    return (c*lo,c*hi,c*rl,c*rh) if c>=0 else (c*hi,c*lo,c*rh,c*rl)

class Frontier:
    def __init__(self,m,meter,*,cancel=True):
        self.m=m
        self.meter=meter
        self.cancel=cancel
        self.entries={}
        self.total=[0,0,0,0]
        self.expansions=0
        self.cancelled_pairs=0
        self.max_frontier=0
        self.cutoffs=0
    def key(self,s,c):
        return (s,0 if self.cancel else (1 if c>0 else -1))
    def prepare(self,edits,exact):
        """Stage a transaction; if it exhausts, leave the frontier unchanged.

        A term stores its cutoff bound, so the running interval is always
        available even at the work cap. No uncharged post-cap traversal.
        """
        pending={}
        new=list(self.total)
        new[0]+=exact[0]; new[1]+=exact[0]
        new[2]+=exact[1]; new[3]+=exact[1]
        for s,c in edits:
            self.meter.charge(TERM_WORK)
            k=self.key(s,c)
            old,b=pending.get(k,self.entries.get(k,(0,None)))
            if b is None:
                b=cutoff(self.m,s,self.meter)
            nxt=old+c
            # Removal of an expanded term is not a reconvergence. Only count
            # cancellation for inserted opposing branches (marked separately
            # below by the caller via the state being different from removal).
            if old:
                delta=signed_interval(old,b)
                new=[x-y for x,y in zip(new,delta)]
            if nxt:
                delta=signed_interval(nxt,b)
                new=[x+y for x,y in zip(new,delta)]
            pending[k]=(nxt,b)
        self.meter.charge(TERM_WORK*len(pending)+8)
        return pending,new
    def apply(self,pending,new):
        for k,(c,b) in pending.items():
            if c:
                self.entries[k]=(c,b)
            else:
                self.entries.pop(k,None)
        self.total=new
        self.max_frontier=max(self.max_frontier,len(self.entries))
    def init(self,s,a,b):
        av,ar,_=immediate(self.m,s,a,self.meter)
        bv,br,_=immediate(self.m,s,b,self.meter)
        ac=children(self.m,s,a,self.meter)
        bc=children(self.m,s,b,self.meter)
        edits=[]
        # Aggregate equal signed children before acquiring any cutoff mass.
        # For the split control the sign remains part of the key.
        d={}
        for sign,ss in ((1,ac),(-1,bc)):
            for t in ss:
                self.meter.charge(TERM_WORK)
                k=self.key(t,sign)
                old=d.get(k,0)
                if old*sign<0:
                    self.cancelled_pairs+=min(abs(old),1)
                d[k]=old+sign
        for k,c in d.items():
            self.meter.charge(8)
            if c:
                edits.append((k[0],c))
        p,v=self.prepare(edits,(av-bv,ar-br))
        self.apply(p,v)
    def expand(self):
        self.meter.charge(SCAN_WORK*len(self.entries)+8)
        key=max(self.entries,key=lambda k:(order(k[0]),k[1]))
        s=key[0]
        c,b=self.entries[key]
        start=self.meter.used
        a=incumbent(self.m,s,self.meter)
        work=self.meter.used-start
        require(work<=(TERMINAL_WORK if self.m.terminal(a) else NONTERMINAL_WORK),"continuation reserve violated")
        v,r,M=immediate(self.m,s,a,self.meter)
        ss=children(self.m,s,a,self.meter)
        # In the split control removal must use the original sign-key.
        # Remove from a staged copy of totals, but do not modify actual
        # entries until the new terms have been successfully priced.
        delta=signed_interval(c,b)
        oldtotal=self.total
        self.total=[x-y for x,y in zip(oldtotal,delta)]
        cancels=0
        try:
            edits=[]
            for t in ss:
                self.meter.charge(TERM_WORK)
                k=self.key(t,c)
                existing=self.entries.get(k,(0,None))[0]
                if existing*c<0:
                    cancels+=min(abs(existing),abs(c))
                edits.append((t,c))
            p,new=self.prepare(edits,(c*(v-work*M),c*r))
        except BaseException:
            self.total=oldtotal
            raise
        # Children have strictly lower rank and cannot be the removed key.
        self.entries.pop(key)
        self.apply(p,new)
        self.cancelled_pairs+=cancels
        self.expansions+=1
    def finish(self,*,risk_limit=0,threshold=0):
        # Stop once the interval is decisive, or when we exhaust the cap.
        # A partial expansion leaves the preceding interval valid.
        while self.entries:
            lo,hi,rl,rh=self.total
            if hi<=threshold or rl>risk_limit or (lo>threshold and rh<=risk_limit):
                break
            try:
                self.expand()
            except Exhausted:
                break
        self.cutoffs=len(self.entries)
        return list(self.total)

def compare(m,s,a,b,meter,*,cancel=True,threshold=0):
    f=Frontier(m,meter,cancel=cancel)
    f.init(s,a,b)
    bounds=f.finish(threshold=threshold)
    return {"action":a,"bounds":bounds,"expanded":f.expansions,
            "cancelled_pairs":f.cancelled_pairs,"frontier":f.cutoffs,
            "max_frontier":f.max_frontier}

def late_compare(m,s,a,b,meter):
    """Exact one-step final-project utility/risk difference.

    No affordable direct audit and no certificate; the incumbent after
    the candidate experiment can only choose recommend or defer. Both
    frontier arms use this same shortcut, so it cannot create a paired
    cancellation advantage.
    """
    require(s.stage==3 and s.cert<0 and m.terminal(b) and
            "direct_audit" not in m.legal(s) and a in ("screen","control","law_audit"),
            "late shortcut outside scope")
    av,ar,M=immediate(m,s,a,meter)
    bv,br,_=immediate(m,s,b,meter)
    cr,wr=m.spec["correct_reward"],m.spec["wrong_reward"]
    count=0
    for t in children(m,s,a,meter):
        require(t.stage==3 and t.cert<0 and "direct_audit" not in m.legal(t),"late child scope")
        cm=mass(m,t,meter)
        meter.charge(64)
        one=m.mass(t.mask&m.truth_masks[3])
        zero=cm-one
        meter.charge(16)
        # Match incumbent reward and lexicographic tie rule exactly.
        reward,action,false=max((0,"defer",0),
            (zero*cr+one*wr,"recommend:0",one),
            (one*cr+zero*wr,"recommend:1",zero))
        av+=reward*PRICE-379*cm
        ar+=false
        count+=1
    return {"action":a,"bounds":[av-bv,av-bv,ar-br,ar-br],
            "expanded":count,"cancelled_pairs":0,"frontier":0,"max_frontier":0}

def candidates(m,s,b,meter):
    meter.charge(16)
    legal=m.legal(s)
    out=[]
    for a in legal:
        meter.charge(8)
        if a==b:
            continue
        # Two terminal decisions have the same next state. The incumbent
        # terminal decision maximizes conditional expected raw reward.
        if m.terminal(b) and m.terminal(a):
            continue
        if s.stage==3 and m.terminal(b) and a in ("screen","control","law_audit"):
            # With no affordable direct audit, a deterministic experiment
            # cannot improve the next terminal choice. Scope is final project.
            mask={"screen":m.screen_masks[3],"control":m.nuisance_mask,"law_audit":m.law_mask}[a]
            part=s.mask&mask
            if part==0 or part==s.mask:
                continue
            # Before screening the final project's independent local truth
            # cannot be informed by a law or nuisance audit alone.
            if not s.screened and a in ("control","law_audit"):
                continue
        out.append(a)
    # Action-specific allocation: skip dominated terminal comparisons and
    # prefer direct/transport evidence, then informative cheap screens.
    priority={"transport":0,"direct_audit":1,"screen":2,"control":3,"law_audit":4}
    return sorted(out,key=lambda a:(priority.get(a,5),m.cost(a),a))

def gate(m,s,b,kind):
    if s.cert>=0 or s.budget==0:
        return False
    late=s.stage==3 and m.terminal(b) and s.budget<m.cost("direct_audit")
    if s.stage==3:
        return late
    if kind=="late":
        return False
    if kind=="tail":
        return s.stage>=2
    require(kind=="broad","bad gate")
    return s.stage>=1

def certificate(m,s,b,meter,config,prior_gate_work,*,cancel=True):
    require(config in CONFIGS.values(),"unregistered config")
    room=meter.cap-meter.used-reserve(s)
    if room<=RECEIPT_WORK:
        return None,{"search_work":0,"exhausted":1,"trials":0,"cancelled_pairs":0,"expanded":0,"frontier":0}
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
                       compare(m,s,a,b,sub,cancel=cancel,threshold=threshold))
            except Exhausted:
                pass
            local.charge(sub.used)
            if trial is not None:
                trial["trial_work"]=sub.used
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
           "expanded":sum(t["expanded"] for t in trials),"frontier":sum(t["frontier"] for t in trials)}
    if M is None:
        return None,stats
    viable=[t for t in trials if t["bounds"][0]>M*(work+prior_gate_work) and t["bounds"][3]<=0]
    best=max(viable,key=lambda t:(t["bounds"][0],-t["bounds"][3],t["action"])) if viable else None
    body={"schema_version":"distill.paired_opportunity.v1","spec_sha256":digest(m.spec),
          "state":asdict(s),"baseline":b,"mass":M,"price":PRICE,"config":config,"cancel":cancel,
          "search_work":work,"prior_gate_work":prior_gate_work,"reserve":reserve(s),
          "remaining_after_search":meter.cap-meter.used,"trials":trials,
          "chosen":best["action"] if best else None,
          "net_lower_numerator":best["bounds"][0]-M*(work+prior_gate_work) if best else None,
          "risk_upper_numerator":best["bounds"][3] if best else None}
    return {**body,"receipt_id":digest(body)},stats

class Auditor:
    """Independent exact weighted utility/risk continuation and world replay."""
    def __init__(self,m):
        self.m=m
        self.checked=0
        self.replays=0
        @lru_cache(None)
        def tail(s):
            if s.stage==4:
                return 0,0
            met=Meter(100000)
            a=incumbent(m,s,met)
            v,r=self.action(s,a)
            return v-met.used*m.mass(s.mask),r
        self.tail=tail
    def action(self,s,a):
        m=self.m
        v=m.reward_sum(s,a)*PRICE
        risk=m.truth_mass(s,1-int(a[-1])) if a.startswith("recommend:") else 0
        for _,t in m.branches(s,a):
            x,y=self.tail(t)
            v+=x; risk+=y
        return v,risk
    def explicit(self,s,a):
        m=self.m
        v=r=0
        for i,w in enumerate(m.worlds):
            if not s.mask>>i&1:
                continue
            state=s
            choice=a
            work=0
            while state.stage<4:
                r+=w.mass*int(choice.startswith("recommend:") and int(choice[-1])!=w.truths[state.stage])
                _,state,reward=actual_transition(m,state,choice,i)
                v+=w.mass*(PRICE*reward-work)
                if state.stage<4:
                    met=Meter(100000)
                    choice=incumbent(m,state,met)
                    work=met.used
            # first action selection was already paid
        require((v,r)==self.action(s,a),"explicit replay differs from exact evaluator")
        self.replays+=1
        return v,r
    def verify(self,s,receipt):
        m=self.m
        require(receipt["receipt_id"]==digest({k:v for k,v in receipt.items() if k!="receipt_id"}),"receipt digest")
        M=m.mass(s.mask)
        require(receipt["spec_sha256"]==digest(m.spec) and receipt["state"]==asdict(s) and receipt["mass"]==M and receipt["price"]==PRICE,"receipt scope")
        require(receipt["reserve"]==reserve(s) and receipt["remaining_after_search"]>=reserve(s),"reserve")
        require(receipt["config"] in CONFIGS.values() and type(receipt["cancel"]) is bool,"config")
        require(type(receipt["search_work"]) is int and RECEIPT_WORK<=receipt["search_work"]<=receipt["config"]["search_cap"],"search work")
        require(type(receipt["prior_gate_work"]) is int and receipt["prior_gate_work"]>=0,"gate work")
        b=incumbent(m,s,Meter(100000))
        require(b==receipt["baseline"],"incumbent")
        bv,br=self.action(s,b)
        best=[]
        for t in receipt["trials"]:
            a=t["action"]
            require(a!=b and a in m.legal(s),"illegal trial")
            v,r=self.action(s,a)
            lo,hi,rl,rh=t["bounds"]
            require(lo<=v-bv<=hi and rl<=r-br<=rh,"invalid paired interval")
            self.checked+=1
            if lo>M*(receipt["search_work"]+receipt["prior_gate_work"]) and rh<=0:
                best.append(t)
        chosen=receipt["chosen"]
        if chosen is None:
            require(not best and receipt["net_lower_numerator"] is None and receipt["risk_upper_numerator"] is None,"missing viable certificate")
        else:
            t=max(best,key=lambda q:(q["bounds"][0],-q["bounds"][3],q["action"]))
            require(t["action"]==chosen,"chosen certificate")
            lo,hi,rl,rh=t["bounds"]
            lower=lo-M*(receipt["search_work"]+receipt["prior_gate_work"])
            require(lower==receipt["net_lower_numerator"] and lower>0 and rh==receipt["risk_upper_numerator"] and rh<=0,"invalid certified margin/risk")
            av,ar=self.explicit(s,chosen)
            bv,br=self.explicit(s,b)
            require(av-bv>=lo and ar-br<=rh,"world replay does not certify")
        return True
    def close(self):
        self.tail.cache_clear()

class Policy:
    def __init__(self,config=None,*,cancel=True,forced=False):
        self.config=config
        self.cancel=cancel
        self.forced=forced
        self.attempted=False
        self.gate_spent=0
    def choose(self,m,s,meter):
        info={k:0 for k in ("gates","attempts","accepted","deviations","search_work","trials","expanded",
                              "cancelled_pairs","frontier","exhausted","fallbacks")}
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
        receipt,stats=certificate(m,s,b,meter,self.config,self.gate_spent,cancel=self.cancel)
        info.update(stats)
        if receipt and receipt["chosen"] is not None:
            info["accepted"]=1
            if not self.forced:
                info["deviations"]=1
                return receipt["chosen"],info,receipt
        return b,info,receipt
