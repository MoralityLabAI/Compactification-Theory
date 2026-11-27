"""DISTILL-08: incumbent-relative research decisions.

Use exact one-deviation values with reuse-first continuation, alongside the
optimal-continuation residual. All labels are evaluator-owned synthetic values.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import math
import random
from .core import require
from .research_program import Model
from .program_oracle import Oracle, Baseline, actual_transition
from .program_student import ACTIONS, FEATURES, Meter, Exhausted, public_features
from .student_campaign import cheap, PRICE

SCALE=10.0
FEATURE_VERSION="distill.residual_features.v1"
GATE_WORK=32
# Public belief summary plus incumbent action; no oracle quantities.
DIM=FEATURES+len(ACTIONS)+1

class Counterfactual:
    def __init__(self,model):
        self.model=model
        self.base=Baseline(model,"reuse_first")
        self.oracle=Oracle(model)
        @lru_cache(None)
        def incumbent(s):
            if s.stage==4:
                return 0
            a=self.base.choose(s)
            return model.reward_sum(s,a)+sum(incumbent(t) for _,t in model.branches(s,a))
        self._incumbent=incumbent
    def label(self,s):
        m=self.model
        b=self.base.choose(s)
        mass=m.mass(s.mask)
        v=self._incumbent(s)
        qstar=self.oracle.action_value(s,b)
        residual={}
        optimal={}
        for a in m.legal(s):
            q=m.reward_sum(s,a)+sum(self._incumbent(t) for _,t in m.branches(s,a))
            residual[a]=Fraction(q-v,mass)
            optimal[a]=self.oracle.action_value(s,a)-qstar
        require(residual[b]==optimal[b]==0,"incumbent residual not zero")
        require(all(self.oracle.action_value(s,a)>=Fraction(v,mass)+residual[a] for a in residual),"incumbent continuation exceeds optimum")
        return b,residual,optimal
    def close(self):
        self._incumbent.cache_clear()
        self.oracle.close()

def gate_key(s,base):
    # State-only prefilter: no posterior mass scans or dense feature work.
    return f"{s.stage}/{base}/{min(3,s.budget//3)}/{int(s.cert>=0)}"

def features(m,s,base,meter=None):
    x=public_features(m,s,meter)
    if meter is not None:
        meter.charge(len(ACTIONS)+1)
    return x+[float(a==base) for a in ACTIONS]+[1.0]

def solve(A,b):
    """Deterministic small ridge linear system, partial pivoting."""
    n=len(b)
    M=[list(row)+[v] for row,v in zip(A,b)]
    for i in range(n):
        p=max(range(i,n),key=lambda k:abs(M[k][i]))
        require(abs(M[p][i])>1e-12,"singular system")
        M[i],M[p]=M[p],M[i]
        d=M[i][i]
        M[i]=[v/d for v in M[i]]
        for k in range(n):
            if k==i:
                continue
            d=M[k][i]
            if d:
                M[k]=[a-d*v for a,v in zip(M[k],M[i])]
    return [M[i][-1] for i in range(n)]

class Residual:
    def __init__(self,weights,counts):
        require(set(weights)==set(counts)==set(ACTIONS),"weight action mismatch")
        require(all(len(w)==DIM and all(math.isfinite(v) for v in w) for w in weights.values()),"bad weights")
        self.weights=weights
        self.counts=counts
    @classmethod
    def fit(cls,rows,*,shuffled=False,seed=17):
        rng=random.Random(seed)
        acc={a:([[0.0]*DIM for _ in range(DIM)],[0.0]*DIM,0) for a in ACTIONS}
        for row in rows:
            x=row["features"]
            b=row["base"]
            y={a:float(v)/SCALE for a,v in row["residual"].items() if a!=b}
            if shuffled:
                keys=list(y)
                vals=list(y.values())
                rng.shuffle(vals)
                y=dict(zip(keys,vals))
            # Root-balanced sampling is approximated by a fixed cap per root;
            # exception rows receive a bounded weight. The same weights are used
            # for shuffled labels, and no oracle feature enters deployment.
            weight=1.0+float(max(row["residual"].values())>0)
            for a,v in y.items():
                A,z,n=acc[a]
                for i,xi in enumerate(x):
                    z[i]+=weight*xi*v
                    for j,xj in enumerate(x):
                        A[i][j]+=weight*xi*xj
                acc[a]=(A,z,n+1)
        weights={}
        counts={}
        for a,(A,b,n) in acc.items():
            for i in range(DIM):
                A[i][i]+=.75
            weights[a]=solve(A,b)
            counts[a]=n
        return cls(weights,counts)
    def choose(self,m,s,base,meter):
        legal=[a for a in ACTIONS if a in m.legal(s) and a!=base and self.counts[a]>0]
        meter.charge(16+len(ACTIONS))
        if not legal:
            return base,0.0
        x=features(m,s,base,meter)
        meter.charge(len(legal)*(DIM+1))
        scored=[(SCALE*sum(v*w for v,w in zip(x,self.weights[a])),a) for a in legal]
        gain,a=max(scored,key=lambda t:(t[0],-m.cost(t[1]),t[1]))
        return (a,gain) if gain>0 else (base,0.0)

def learn_gate(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["gate_key"]].append(r)
    result={}
    for key,group in sorted(groups.items()):
        roots={r["root_id"] for r in group}
        means=[]
        for root in sorted(roots):
            vals=[max(r["residual"].values()) for r in group if r["root_id"]==root]
            means.append(sum(vals,Fraction(0))/len(vals))
        if len(roots)>=3 and len(group)>=6:
            result[key]={"mean_opportunity":float(sum(means,Fraction(0))/len(means)),
                         "states":len(group),"roots":len(roots)}
    return result

def budget_rule(m,s,base,meter):
    """Hand-authored exception policy, with all terminal scans charged.

    Preserve a valid certificate immediately, and avoid an audit of the law
    when remaining credits cannot fund even one subsequent transport.
    """
    meter.charge(16)
    if s.cert>=0 and not base.startswith("certify:"):
        return "certify:"+str(s.cert)
    if base=="law_audit" and s.budget<m.cost("law_audit")+m.cost("transport"):
        return cheap(m,s,meter,"direct_first")
    return base

def select(m,s,meter,arm,model=None,gate=None,config=None):
    """Compute the incumbent before spending optional work.

    A failed optional reservation keeps the completed incumbent action. No
    candidate can alter the host's legality or use extra work after exhaustion.
    """
    info={"score_calls":0,"deviations":0,"fallbacks":0,"gate_skips":0}
    try:
        b=cheap(m,s,meter,"reuse_first")
    except Exhausted:
        info["fallbacks"]=1
        return "defer",info
    if arm=="reuse_first" or (config is None and arm in ("gated","shuffled","forced_base")):
        return b,info
    try:
        if arm=="budget_rule":
            a=budget_rule(m,s,b,meter)
            info["deviations"]=int(a!=b)
            return a,info
        require(arm in ("always","gated","shuffled","forced_base") and model is not None,"bad residual policy")
        if arm!="always":
            meter.charge(GATE_WORK)
            entry=gate.get(gate_key(s,b))
            if entry is None or entry["mean_opportunity"]<config["gate"]:
                info["gate_skips"]=1
                return b,info
        # Keep a fixed reserve for inexpensive future incumbent choices.
        # Reserve is an admission check, not an extra deduction.
        meter.charge(1)
        reserve=400*(4-s.stage)+24*s.budget
        if meter.cap-meter.used<reserve+1800:
            info["gate_skips"]=1
            return b,info
        info["score_calls"]=1
        a,gain=model.choose(m,s,b,meter)
        meter.charge(1)
        threshold=0.25 if arm=="always" else config["margin"]
        if arm=="forced_base" or gain<=threshold:
            a=b
        info["deviations"]=int(a!=b)
        return a,info
    except Exhausted:
        info["fallbacks"]=1
        return b,info

def reachable(model,chooser):
    """Unique sufficient public states reached by a deterministic policy."""
    seen=set()
    stack=[model.initial()]
    while stack:
        s=stack.pop()
        if s.stage==4 or s in seen:
            continue
        seen.add(s)
        a=chooser(s)
        stack.extend(t for _,t in model.branches(s,a))
    return seen

def order(s):
    return (s.stage,-s.budget,s.mask,s.screened,s.control,s.law,s.anchor,s.cert)

def labeled_rows(spec,*,provisional=None,limit=96):
    m=Model(spec)
    cf=Counterfactual(m)
    base=reachable(m,cf.base.choose)
    oracle=reachable(m,cf.oracle.choose)
    deviated=set()
    if provisional is not None:
        # Explore the provisional policy without a trajectory work cap for
        # label collection; the final benchmark charges the actual meter.
        deviated=reachable(m,lambda s:select(m,s,Meter(32768),"always",provisional)[0])
    all_states=base|oracle|deviated
    rows=[]
    rid=__import__("Distill.core",fromlist=["digest"]).digest(spec)
    for s in sorted(all_states,key=order):
        b,q,star=cf.label(s)
        rows.append({"root_id":rid,"state":s,"base":b,"features":features(m,s,b),
                     "residual":q,"optimal_residual":star,"gate_key":gate_key(s,b),
                     "incumbent_reached":s in base,"provisional_reached":s in deviated,
                     "new_after_deviation":s in deviated and s not in base|oracle})
    # Keep representative off-policy mistakes and positive opportunities while
    # bounding each root's influence. No evaluation-root labels are fitted.
    priority=lambda r:(-int(r["new_after_deviation"]),-int(max(r["residual"].values())>0),
                       -int(r["incumbent_reached"]),order(r["state"]))
    chosen=sorted(rows,key=priority)[:limit]
    stats={"reachable":len(rows),"incumbent_states":len(base),"oracle_states":len(oracle),
           "provisional_states":len(deviated),"novel_states":sum(r["new_after_deviation"] for r in rows),
           "retained":len(chosen),"retained_novel":sum(r["new_after_deviation"] for r in chosen),
           "positive_one_deviation":sum(max(r["residual"].values())>0 for r in chosen),
           "positive_optimal_continuation":sum(max(r["optimal_residual"].values())>0 for r in chosen),
           "action_labels":sum(len(r["residual"]) for r in chosen)}
    cf.close()
    return chosen,stats
