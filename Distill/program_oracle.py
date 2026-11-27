"""Exact whole-program planning and cheap research baselines.

Oracle search is evaluator work. All policies pay identical environment action
costs; planner work is counted separately, so no matched-runtime claim follows.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from .core import require
from .research_program import Model

def frac(x):
    x=Fraction(x)
    return f"{x.numerator}/{x.denominator}"

class Oracle:
    def __init__(self,model):
        self.model=model
        self.state_evaluations=0
        self.action_evaluations=0
        self.branch_evaluations=0
        @lru_cache(None)
        def value(s):
            self.state_evaluations+=1
            if s.stage==4:
                return 0,None
            choices=[]
            for a in model.legal(s):
                self.action_evaluations+=1
                children=model.branches(s,a)
                self.branch_evaluations+=len(children)
                v=model.reward_sum(s,a)+sum(value(t)[0] for _,t in children)
                choices.append((v,a))
            return max(choices,key=lambda item:(item[0],-model.cost(item[1]),item[1]))
        self._value=value
    def choose(self,s):
        return self._value(s)[1]
    def expected(self,s=None):
        s=self.model.initial() if s is None else s
        return Fraction(self._value(s)[0],self.model.mass(s.mask))
    def action_value(self,s,a):
        value=self.model.reward_sum(s,a)+sum(self._value(t)[0] for _,t in self.model.branches(s,a))
        return Fraction(value,self.model.mass(s.mask))
    def work(self):
        return {"states":self.state_evaluations,"actions":self.action_evaluations,"branches":self.branch_evaluations}
    def close(self):
        self._value.cache_clear()

class Baseline:
    def __init__(self,model,name):
        require(name in ("stop","myopic","direct_first","reuse_first"),"unknown baseline")
        self.model=model
        self.name=name
        self.action_evaluations=0
    def best_terminal(self,s):
        model=self.model
        values=[]
        for a in model.legal(s):
            if model.terminal(a):
                self.action_evaluations+=1
                values.append((model.reward_sum(s,a),a))
        return max(values,key=lambda row:(row[0],row[1]))
    def choose(self,s):
        model=self.model
        legal=model.legal(s)
        best=self.best_terminal(s)
        if self.name=="stop":
            return best[1]
        if self.name=="direct_first":
            return "direct_audit" if "direct_audit" in legal else best[1]
        if self.name=="reuse_first":
            if s.stage==0 and s.anchor<0 and "direct_audit" in legal:
                return "direct_audit"
            if s.stage<=1 and s.anchor>=0 and s.law<0 and "law_audit" in legal:
                return "law_audit"
            if "transport" in legal:
                return "transport"
            if "direct_audit" in legal:
                return "direct_audit"
            return best[1]
        # One-step local replanning. It ignores the value of future projects.
        candidates=[best]
        for a in legal:
            if not model.terminal(a):
                self.action_evaluations+=1
                v=model.reward_sum(s,a)+sum(self.best_terminal(t)[0] for _,t in model.branches(s,a))
                candidates.append((v,a))
        return max(candidates,key=lambda row:(row[0],-model.cost(row[1]),row[1]))[1]

def actual_transition(model,s,action,world_index):
    """Explicit world-level replay of observations and state, without DP values."""
    from dataclasses import replace
    require(action in model.legal(s),"illegal replay action")
    w=model.worlds[world_index]
    stage=s.stage
    require(s.mask>>world_index&1,"world outside compatible belief")
    if model.terminal(action):
        reward=model.realized_reward(s,action,w)
        return None,type(s)(stage+1,s.mask,s.budget,control=s.control,law=s.law,anchor=s.anchor),reward
    cost=model.cost(action)
    if action=="screen":
        bit=w.truths[stage]^w.bits[5]^(w.bits[1] if stage==3 and model.spec["family"]=="shifted_control" else 0)
        kwargs={"screened":True}
        pred=lambda u:u.truths[stage]^u.bits[5]^(u.bits[1] if stage==3 and model.spec["family"]=="shifted_control" else 0)
    elif action=="control":
        bit=w.bits[5]
        kwargs={"control":bit}
        pred=lambda u:u.bits[5]
    elif action=="law_audit":
        bit=w.bits[1]
        kwargs={"law":bit}
        pred=lambda u:u.bits[1]
    elif action=="direct_audit":
        bit=w.truths[stage]
        kwargs={"cert":bit}
        if stage==0:
            kwargs["anchor"]=bit
        pred=lambda u:u.truths[stage]
    elif action=="transport":
        bit=s.anchor^model.spec["offsets"][stage-1]
        require(w.truths[stage]==bit and all(u.truths[stage]==bit for i,u in enumerate(model.worlds) if s.mask>>i&1),
                "transport witness fails")
        return bit,replace(s,budget=s.budget-cost,cert=bit),-cost
    else:
        raise ValueError("unknown experiment")
    mask=sum(1<<i for i,u in enumerate(model.worlds) if s.mask>>i&1 and pred(u)==bit)
    return bit,replace(s,budget=s.budget-cost,mask=mask,**kwargs),-cost

def replay(model,choose):
    """Enumerate actual worlds; independently replay actions and integer rewards."""
    sums={k:0 for k in ("utility","credits","certified","transports","false_recommendations","deferred","actions")}
    transitions=0
    total=model.mass(model.full_mask)
    for i,w in enumerate(model.worlds):
        s=model.initial()
        metrics={k:0 for k in sums}
        while s.stage<4:
            require(metrics["actions"]<32,"nonterminating policy")
            a=choose(s)
            bit,t,reward=actual_transition(model,s,a,i)
            matched=[(b,c) for b,c in model.branches(s,a) if c.mask>>i&1]
            require(matched==[(bit,t)],"explicit replay/kernel mismatch")
            metrics["utility"]+=reward
            metrics["credits"]+=model.cost(a)
            metrics["certified"]+=int(a.startswith("certify:"))
            metrics["transports"]+=int(a=="transport")
            metrics["false_recommendations"]+=int(a.startswith("recommend:") and int(a[-1])!=w.truths[s.stage])
            metrics["deferred"]+=int(a=="defer")
            metrics["actions"]+=1
            s=t
            transitions+=1
        require(metrics["credits"]<=model.spec["budget"],"budget violation")
        for k in sums:
            sums[k]+=w.mass*metrics[k]
    return {**{k:frac(Fraction(v,total)) for k,v in sums.items()},"world_transitions":transitions}

def evaluate_root(spec):
    model=Model(spec)
    oracle=Oracle(model)
    optimal=oracle.expected()
    work=oracle.work()
    results={"oracle":replay(model,oracle.choose)}
    require(Fraction(results["oracle"]["utility"])==optimal,"oracle value/replay mismatch")
    results["oracle"]["planner_work"]=work
    for name in ("stop","myopic","direct_first","reuse_first"):
        policy=Baseline(model,name)
        row=replay(model,policy.choose)
        require(Fraction(row["utility"])<=optimal,"baseline exceeds oracle")
        row["planner_work"]={"action_evaluations":policy.action_evaluations,
                             "scope":"counts during all-world replay, not matched runtime"}
        results[name]=row
    no_model=Model(spec,allow_transport=False)
    no=Oracle(no_model)
    v=no.expected()
    results["oracle_no_transport"]=replay(no_model,no.choose)
    require(Fraction(results["oracle_no_transport"]["utility"])==v and v<=optimal,"transport ablation mismatch")
    results["oracle_no_transport"]["planner_work"]=no.work()
    for row in results.values():
        row["regret"]=frac(optimal-Fraction(row["utility"]))
    oracle.close()
    no.close()
    return {"oracle_value":frac(optimal),"transport_value":frac(optimal-v),"policies":results,
            "scope":"exact synthetic expected utility over 64 worlds; environment costs matched, planner work separate"}
