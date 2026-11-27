"""DISTILL-11: policy-relative research-state quotient.

Equivalence applies only to the frozen budget-rule continuation, inside one
public program specification. It preserves the chosen action, its metered
selection cost, normalized expected reward/risk, and weighted successor classes.
It is not a quotient of arbitrary experimental policies.
"""
from __future__ import annotations
from collections import defaultdict
from functools import reduce
from math import gcd
from fractions import Fraction
from .core import require
from .program_student import Meter, Exhausted
from .program_opportunity import incumbent, reserve, NONTERMINAL_WORK, TERMINAL_WORK
from .program_cancellation import (immediate, children, signed_interval, late_compare,
    candidates, gate, CONFIGS, GATE_WORK, RECEIPT_WORK, TERM_WORK, SCAN_WORK)
from .student_campaign import PRICE

MODES=("literal","flags","marginal")

def budget_cap(m,s):
    """Saturation threshold for the incumbent's remaining decisions."""
    if s.stage==4:
        return 0
    D,T=m.cost("direct_audit"),m.cost("transport")
    if s.stage==3:
        return 0 if s.cert>=0 else D
    # Each uncertified project may spend at most max(D,T) to certify.
    # A single law audit needs its cost plus T for the budget-rule guard.
    r=4-s.stage-int(s.cert>=0)
    return r*max(D,T)+(m.cost("law_audit")+T if s.stage<=1 and s.law<0 else 0)

def signature(m,s,mode,meter):
    """Return (class key, integer mass factor, normalized mass).

    For marginal keys each cell is an exact prior mass over a future-truth
    tuple (and the law bit if the policy might still audit it). Divide by
    gcd so proportional distributions share a key. Charge the actual
    declared world scan and bin reduction before acquiring the signature.
    """
    require(mode in MODES and 0<=s.stage<=4,"bad signature mode/stage")
    meter.charge(16)
    if s.stage==4:
        return (4,),1,0
    if mode=="literal":
        meter.charge(64)
        M=m.mass(s.mask)
        return (s.stage,s.budget,s.mask,s.screened,s.control,s.law,s.anchor,s.cert),1,M
    b=min(s.budget,budget_cap(m,s))
    if s.stage==3 and s.budget<m.cost("direct_audit"):
        b=0  # all unaffordable terminal budgets are equivalent here
    law,anchor=(s.law,s.anchor) if s.stage<3 else (-1,-1)
    prefix=(s.stage,b,law,anchor,s.cert)
    if mode=="flags":
        meter.charge(64)
        return prefix+(s.mask,),1,m.mass(s.mask)
    bits=4-s.stage+int(s.stage<=1)
    meter.charge(64*(bits+2)+8*(1<<bits)+16)
    bins=[0]*(1<<bits)
    for i,w in enumerate(m.worlds):
        if s.mask>>i&1:
            k=0
            for bit in w.truths[s.stage:]:
                k=2*k+bit
            if s.stage<=1:
                k=2*k+w.bits[1]
            bins[k]+=w.mass
    factor=reduce(gcd,bins)
    require(factor>0,"empty projected belief")
    reduced=tuple(x//factor for x in bins)
    return prefix+(reduced,),factor,sum(reduced)

def exact_div(a,b):
    q,r=divmod(a,b)
    require(r==0,"nonintegral normalized continuation")
    return q

def rank(key):
    return (4-key[0])*16+key[1]

def bounds(m,s,N):
    r=4-s.stage
    cr,wr=m.spec["correct_reward"],m.spec["wrong_reward"]
    return (-(s.budget*PRICE+reserve(s))*N,r*m.spec["certificate_reward"]*PRICE*N,
            0,(r*N*cr)//(cr-wr))

class Frontier:
    """Each entry is a signed integer mass factor times a canonical value.

    A representative need not have the same absolute prior mass as another
    class member. Its gcd factor is divided out on expansion; conditioning
    on a projected bit and summing bins preserve that divisibility.
    """
    def __init__(self,m,meter,mode):
        self.m,self.meter,self.mode=m,meter,mode
        self.entries={}
        self.total=[0,0,0,0]
        self.expanded=0
        self.cancelled_pairs=0
        self.quotient_merges=0
        self.max_frontier=0
    def prepare(self,edits,exact):
        pending={}
        new=[self.total[0]+exact[0],self.total[1]+exact[0],
             self.total[2]+exact[1],self.total[3]+exact[1]]
        cancels=merges=0
        for s,c in edits:
            self.meter.charge(TERM_WORK)
            k,g,N=signature(self.m,s,self.mode,self.meter)
            # c scales the unnormalized state value.
            w=exact_div(c.numerator*g,c.denominator)
            old,rep,rg,bb=pending.get(k,self.entries.get(k,(0,s,g,bounds(self.m,s,N))))
            if old:
                if old*w<0:
                    cancels+=1
                if rep!=s:
                    merges+=1
                new=[x-y for x,y in zip(new,signed_interval(old,bb))]
            nxt=old+w
            if nxt:
                new=[x+y for x,y in zip(new,signed_interval(nxt,bb))]
            pending[k]=(nxt,rep,rg,bb)
        self.meter.charge(TERM_WORK*len(pending)+8)
        return pending,new,cancels,merges
    def apply(self,p,new,cancels,merges):
        for k,(w,s,g,b) in p.items():
            if w:
                self.entries[k]=(w,s,g,b)
            else:
                self.entries.pop(k,None)
        self.total=new
        self.cancelled_pairs+=cancels
        self.quotient_merges+=merges
        self.max_frontier=max(self.max_frontier,len(self.entries))
    def init(self,s,a,b):
        av,ar,_=immediate(self.m,s,a,self.meter)
        bv,br,_=immediate(self.m,s,b,self.meter)
        ac=children(self.m,s,a,self.meter)
        bc=children(self.m,s,b,self.meter)
        self.apply(*self.prepare([(t,Fraction(1)) for t in ac]+[(t,Fraction(-1)) for t in bc],(av-bv,ar-br)))
    def expand(self):
        self.meter.charge(SCAN_WORK*len(self.entries)+8)
        key=max(self.entries,key=lambda k:(rank(k),k))
        w,s,g,bb=self.entries[key]
        start=self.meter.used
        a=incumbent(self.m,s,self.meter)
        work=self.meter.used-start
        require(work<=(TERMINAL_WORK if self.m.terminal(a) else NONTERMINAL_WORK),"reserve bound")
        v,r,M=immediate(self.m,s,a,self.meter)
        ss=children(self.m,s,a,self.meter)
        coeff=Fraction(w,g)
        self.meter.charge(16)
        v=exact_div((v-work*M)*w,g)
        r=exact_div(r*w,g)
        old=self.total
        self.total=[x-y for x,y in zip(old,signed_interval(w,bb))]
        try:
            pending,new,cancels,merges=self.prepare([(t,coeff) for t in ss],(v,r))
            require(all(rank(k)<rank(key) for k in pending),"quotient rank not decreasing")
        except BaseException:
            self.total=old
            raise
        self.entries.pop(key)
        self.apply(pending,new,cancels,merges)
        self.expanded+=1
    def finish(self,threshold=0):
        while self.entries:
            lo,hi,rl,rh=self.total
            if hi<=threshold or rl>0 or (lo>threshold and rh<=0):
                break
            try:
                self.expand()
            except Exhausted:
                break
        return list(self.total)

def compare(m,s,a,b,meter,mode,threshold=0):
    f=Frontier(m,meter,mode)
    f.init(s,a,b)
    bounds_=f.finish(threshold)
    return {"action":a,"bounds":bounds_,"expanded":f.expanded,
            "cancelled_pairs":f.cancelled_pairs,"quotient_merges":f.quotient_merges,
            "frontier":len(f.entries),"max_frontier":f.max_frontier}

def local_witness(m,s,mode):
    """Exact normalized one-step signature for independent congruence audit."""
    met=Meter(100000)
    k,g,N=signature(m,s,mode,met)
    M=g*N
    if s.stage==4:
        return k,("done",)
    met=Meter(100000)
    a=incumbent(m,s,met)
    work=met.used
    v,r,_=immediate(m,s,a,Meter(100000))
    succ=defaultdict(Fraction)
    for _,t in m.branches(s,a):
        key,_,_=signature(m,t,mode,Meter(100000))
        if t.stage<4:
            require(rank(key)<rank(k),"bad abstract rank")
        succ[key]+=Fraction(m.mass(t.mask),M)
    require(sum(succ.values())==1,"bad successor probability")
    return k,(a,work,Fraction(v,M),Fraction(r,M),tuple(sorted(succ.items())))

def audit_states(m,states):
    counts={}
    for mode in MODES:
        classes={}
        members=defaultdict(set)
        for s in states:
            k,w=local_witness(m,s,mode)
            require(classes.setdefault(k,w)==w,"quotient congruence counterexample")
            members[k].add(s)
        counts[mode]={"states":len(states),"classes":len(classes),
                      "nontrivial_classes":sum(len(v)>1 for v in members.values()),
                      "largest_class":max(map(len,members.values()),default=0)}
    return counts

def closure(m,starts,*,all_actions=False):
    todo=list(starts)
    seen=set()
    while todo:
        s=todo.pop()
        if s in seen:
            continue
        seen.add(s)
        if s.stage==4:
            continue
        actions=m.legal(s) if all_actions else (incumbent(m,s,Meter(100000)),)
        for a in actions:
            todo.extend(t for _,t in m.branches(s,a))
    return seen

def diagnostic(m):
    """Complete, unmetered referee closures; never used by the actor.

    Count opportunities before depth/work truncation. Intersections exclude
    finished states. Include only actual candidate-filtered decisions at
    incumbent-reachable states admitted by the broad gate.
    """
    base_states=closure(m,[m.initial()])
    audit=set(base_states)
    counts={mode:{"pairs_with_overlap":0,"shared_classes":0} for mode in MODES}
    extra={"flags":0,"marginal":0}
    pairs=0
    for s in sorted((x for x in base_states if x.stage<4),key=lambda s:(s.stage,s.budget,s.mask,s.cert,s.law,s.anchor,s.control,s.screened)):
        b=incumbent(m,s,Meter(100000))
        if not gate(m,s,b,"broad"):
            continue
        bc=closure(m,[t for _,t in m.branches(s,b)])
        audit.update(bc)
        bb={mode:{signature(m,t,mode,Meter(100000))[0] for t in bc if t.stage<4} for mode in MODES}
        for a in candidates(m,s,b,Meter(100000)):
            ac=closure(m,[t for _,t in m.branches(s,a)])
            audit.update(ac)
            pairs+=1
            overlap={}
            for mode in MODES:
                aa={signature(m,t,mode,Meter(100000))[0] for t in ac if t.stage<4}
                overlap[mode]=len(aa&bb[mode])
                counts[mode]["pairs_with_overlap"]+=int(bool(overlap[mode]))
                counts[mode]["shared_classes"]+=overlap[mode]
            for mode in extra:
                extra[mode]+=int(overlap[mode]>0 and overlap["literal"]==0)
    return {"candidate_pairs":pairs,"incumbent_states":len(base_states),
            "overlap":counts,"quotient_only_pairs":extra,"congruence":audit_states(m,{s for s in audit if s.stage<4})}
