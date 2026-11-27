"""Small, tied-recurrent research ranker with public-state inputs.

Synthetic oracle labels only. The input is a deterministic compression of the
public observation (including its prior and evidence ledger). No hidden world,
root ID, exact continuation value, or provider output enters the features.
"""
from __future__ import annotations
import math
import random
from .core import digest, require
from .research_program import Model, PROJECTS

ACTIONS=("recommend:0","recommend:1","defer","certify:0","certify:1",
         "screen","control","law_audit","direct_audit","transport")
COST_NAMES=("screen","control","law_audit","direct_audit","transport")
FEATURE_VERSION="distill.program_features.v1"
HIDDEN=10
REGRET_SCALE=10.0

class Exhausted(Exception):
    pass

class Meter:
    """Declared deterministic logical-work units, not a count of CPU cycles."""
    def __init__(self,cap):
        require(type(cap) is int and cap>=0,"bad work cap")
        self.cap=cap
        self.used=0
    def charge(self,n):
        require(type(n) is int and n>=0,"invalid charge")
        if self.used+n>self.cap:
            raise Exhausted()
        self.used+=n

def state_from_observation(obs):
    # The host may use the equivalent sufficient public state directly.
    require(obs["state_id"]==digest({k:v for k,v in obs.items() if k!="state_id"}),"observation digest mismatch")
    model=Model(obs["spec"])
    require(obs["program_id"]==digest(obs["spec"]),"root binding mismatch")
    s=model.initial()
    for event in obs["ledger"]:
        require(s.stage<4 and event["project"]==PROJECTS[s.stage],"ledger stage mismatch")
        a=event["action"]
        require(event["cost"]==model.cost(a),"ledger cost mismatch")
        successors=[t for bit,t in model.branches(s,a) if bit==event["observation"]]
        require(len(successors)==1,"ledger observation not possible")
        s=successors[0]
        require(s.budget==event["remaining_credits"],"ledger budget mismatch")
    require(s.stage==obs["stage"] and s.budget==obs["remaining_credits"],"state mismatch")
    require(s.mask.bit_count()==obs["compatible_worlds"],"compatible-world mismatch")
    require([model.truth_mass(s,b) for b in (0,1)]==obs["current_truth_masses"],"mass mismatch")
    require([{"action":a,"cost":model.cost(a)} for a in model.legal(s)]==obs["legal_actions"],"legal action mismatch")
    return model,s

def _mass(model,mask,meter):
    # Charge all 64 world slots even if the trusted host's cache is warm.
    if meter is not None:
        meter.charge(64)
    return model.mass(mask)

def public_features(model,s,meter=None):
    """Feature acquisition is charged before doing each unit of work."""
    if meter is not None:
        meter.charge(64)  # fixed state/context and legality extraction
    legal=set(model.legal(s))
    d=model.spec
    m=_mass(model,s.mask,meter)
    require(s.stage<4 and m>0,"bad feature state")
    def prob(mask):
        return _mass(model,s.mask&mask,meter)/m
    x=[float(s.stage==i) for i in range(4)]
    x += [s.budget/14,d["budget"]/14,(4-s.stage)/4,
          float(d["family"]=="shifted_control"),float(s.screened)]
    for flag in (s.control,s.law,s.anchor,s.cert):
        x += [float(flag==v) for v in (-1,0,1)]
    x += list(map(float,d["offsets"]))
    x += [prob(model.law_mask),prob(model.nuisance_mask)]
    x += [prob(mask) for mask in model.truth_masks]
    x += [prob(model.screen_masks[s.stage]),s.mask.bit_count()/64]
    x += [d["costs"][a]/6 for a in COST_NAMES]
    x += [d["correct_reward"]/20,d["wrong_reward"]/40,d["certificate_reward"]/30]
    x += [float(a in legal) for a in ACTIONS]
    require(len(x)==FEATURES and all(math.isfinite(v) for v in x),"invalid feature shape")
    return x

FEATURES=4+5+12+2+2+4+2+5+3+len(ACTIONS)

class Ranker:
    """h[t+1]=tanh(Wx*x+b+Wh*h[t]); y=Wo*h[T]+bo.

    Stored tensor shapes are identical for T=1 and T=3. Wh has no effect at
    one visit since h[0]=0. This is tied within-decision computation, not a
    demonstration of learned memory across environment steps.
    """
    def __init__(self,seed,visits,width=HIDDEN):
        require(type(seed) is int and visits in (1,3) and width==HIDDEN,"invalid ranker")
        self.seed,self.visits,self.width=seed,visits,width
        self.offset_h=width*(FEATURES+1)
        self.offset_o=self.offset_h+width*width
        self.nparams=self.offset_o+len(ACTIONS)*(width+1)
        rng=random.Random(seed)
        self.p=[rng.uniform(-.15,.15) for _ in range(self.nparams)]
    def work(self):
        h=self.width
        # Scalar multiply-accumulate units, bias and activation charges.
        return h*(FEATURES+1)+self.visits*(h*h+2*h)+len(ACTIONS)*(h+1)
    def forward(self,x):
        h=self.width
        p=self.p
        u=[p[i*(FEATURES+1)+FEATURES]+sum(p[i*(FEATURES+1)+j]*x[j] for j in range(FEATURES)) for i in range(h)]
        hist=[[0.0]*h]
        for _ in range(self.visits):
            prev=hist[-1]
            hist.append([math.tanh(u[i]+sum(p[self.offset_h+i*h+j]*prev[j] for j in range(h))) for i in range(h)])
        last=hist[-1]
        y=[p[self.offset_o+k*(h+1)+h]+sum(p[self.offset_o+k*(h+1)+j]*last[j] for j in range(h)) for k in range(len(ACTIONS))]
        return y,hist
    def predict(self,x,meter=None):
        if meter is not None:
            meter.charge(self.work()+len(ACTIONS))
        return self.forward(x)[0]
    def loss_gradient(self,x,targets):
        require(bool(targets),"empty legal targets")
        h=self.width
        y,hist=self.forward(x)
        g=[0.0]*len(self.p)
        delta=[0.0]*h
        loss=0.0
        for k,t in targets.items():
            err=y[k]-t
            loss+=err*err/len(targets)
            dy=2*err/len(targets)
            base=self.offset_o+k*(h+1)
            g[base+h]+=dy
            for j in range(h):
                g[base+j]+=dy*hist[-1][j]
                delta[j]+=dy*self.p[base+j]
        du=[0.0]*h
        for t in range(self.visits,0,-1):
            dh=[delta[i]*(1-hist[t][i]**2) for i in range(h)]
            prev=hist[t-1]
            delta=[0.0]*h
            for i in range(h):
                du[i]+=dh[i]
                base=self.offset_h+i*h
                for j in range(h):
                    g[base+j]+=dh[i]*prev[j]
                    delta[j]+=dh[i]*self.p[base+j]
        for i in range(h):
            base=i*(FEATURES+1)
            for j in range(FEATURES):
                g[base+j]+=du[i]*x[j]
            g[base+FEATURES]+=du[i]
        return loss,g
    def fit(self,rows,*,epochs=18,shuffle_labels=False):
        require(type(epochs) is int and 1<=epochs<=128 and type(shuffle_labels) is bool,"invalid fit config")
        require(bool(rows),"empty training rows")
        # Shuffle once, within each state's legal actions. Keep target support
        # and numerical distribution, but destroy action/utility association.
        rng=random.Random(self.seed+9101)
        data=[]
        for x,targets in rows:
            t=dict(targets)
            if shuffle_labels:
                keys=list(t)
                values=list(t.values())
                rng.shuffle(values)
                t=dict(zip(keys,values))
            data.append((x,t))
        first=[0.0]*len(self.p)
        second=[0.0]*len(self.p)
        updates=0
        losses=[]
        for epoch in range(epochs):
            ids=list(range(len(data)))
            rng.shuffle(ids)
            total=0.0
            for idx in ids:
                loss,g=self.loss_gradient(*data[idx])
                total+=loss
                updates+=1
                b1=1-.9**updates
                b2=1-.999**updates
                norm=math.sqrt(sum(v*v for v in g))
                scale=min(1.0,5.0/max(norm,1e-12))
                for j,v in enumerate(g):
                    v*=scale
                    first[j]=.9*first[j]+.1*v
                    second[j]=.999*second[j]+.001*v*v
                    self.p[j]-=.003*(first[j]/b1)/(math.sqrt(second[j]/b2)+1e-8)
            losses.append(total/len(data))
        require(all(math.isfinite(v) for v in self.p),"nonfinite weights")
        return {"updates":updates,"epochs":epochs,"training_loss_first":losses[0],
                "training_loss_last":losses[-1],"parameters":len(self.p),
                "active_parameters":len(self.p)-(self.width*self.width if self.visits==1 else 0),
                "weights_sha256":digest(self.p),"shuffle_labels":shuffle_labels}
    def choose(self,model,s,meter):
        x=public_features(model,s,meter)
        y=self.predict(x,meter)
        legal=set(model.legal(s))
        return max((a for a in ACTIONS if a in legal),key=lambda a:(y[ACTIONS.index(a)],-model.cost(a),a)),y
