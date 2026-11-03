"""Algernon H-M1: CPU-only synthetic framework revision with multi-step probes.

A rule learner begins with six primitive Boolean hypotheses. A paid EXPAND action
enumerates six additional two-variable compositions from a fixed Boolean DSL.
Earlier evidence is re-scored under the expanded grammar. Policy controls must
pay the same expansion and probe costs. There is no model-weight editing, Qwen,
or unrestricted invention of hypotheses.

The exact planner receives the full task-generating prior and is an ORACLE
teacher; the distilled small network sees observations, costs, simple-family
posterior and public task parameters. No true rule identifier is provided.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

INPUTS=np.asarray([[0,0],[0,1],[1,0],[1,1]], dtype=np.int8)
SIMPLE=("false","true","x0","x1","not_x0","not_x1")
COMPOSED=("and","or","xor","xnor","nand","nor")
ACTIONS=("probe_00","probe_01","probe_10","probe_11","expand","stop")
EXPAND=4; STOP=5; NQ=4; BUDGET=3.2; PENALTY=0.028
TRAIN_START=610000000; VAL_START=620000000; TEST_START=630000000; CONFIRM_START=640000000


def rules():
    """Enumerate a small, explicitly bounded compositional grammar."""
    a,b=INPUTS[:,0].astype(bool), INPUTS[:,1].astype(bool)
    return np.array([
        np.zeros(4,dtype=bool),np.ones(4,dtype=bool),a,b,~a,~b,
        a&b,a|b,a^b,~(a^b),~(a&b),~(a|b)
    ],dtype=np.int8)

TABLE=rules()
assert TABLE.shape==(12,4) and len({tuple(r) for r in TABLE})==12

@dataclass
class World:
    seed: int
    q: float
    prior: np.ndarray
    noise: np.ndarray
    costs: np.ndarray
    expand_cost: float
    weights: np.ndarray
    truth: int
    observed: np.ndarray


def make_world(seed:int, q_override:float|None=None) -> World:
    rng=np.random.default_rng(seed)
    q=float(q_override if q_override is not None else rng.uniform(.12,.88))
    # Random within-family prior is visible in the task specification.
    simple=rng.dirichlet(np.full(6,1.7))
    complex_=rng.dirichlet(np.full(6,1.7))
    prior=np.concatenate(((1-q)*simple,q*complex_))
    noise=rng.uniform(.025,.17,4)
    costs=rng.choice([.65,.80,.95,1.10,1.30],4).astype(float)
    expand_cost=float(rng.choice([.85,1.05,1.25]))
    weights=rng.dirichlet(np.full(4,.85))
    truth=int(rng.choice(len(TABLE),p=prior))
    observed=(TABLE[truth] != (rng.random(4)<noise)).astype(np.int8)
    return World(seed,q,prior,noise,costs,expand_cost,weights,truth,observed)


def state_cost(w:World,mask:int,expanded:bool)->float:
    return float(sum(w.costs[i] for i in range(NQ) if mask & (1<<i))+(w.expand_cost if expanded else 0.))


def posterior(w:World,mask:int,outcomes:int,expanded:bool)->np.ndarray:
    # Full Bayes posteriors serve as the generative teacher's belief state;
    # simple-only posteriors are used for actual unexpanded predictions.
    n=12 if expanded else 6
    p=w.prior[:n].copy()
    for i in range(NQ):
        if mask&(1<<i):
            probs=np.where(TABLE[:n,i]==((outcomes>>i)&1),1-w.noise[i],w.noise[i])
            p*=probs
    total=p.sum()
    assert total>0 and np.isfinite(total)
    return p/total


def predicted_labels(w:World,mask:int,outcomes:int,expanded:bool)->np.ndarray:
    p=posterior(w,mask,outcomes,expanded)
    return (p@TABLE[:len(p),:] > .5).astype(np.int8)


def expected_accuracy(w:World,mask:int,outcomes:int,expanded:bool)->float:
    # Judge prediction against ALL possible true rules using the full generative
    # posterior. The unexpanded rule learner has to make simple-family predictions.
    ptrue=posterior(w,mask,outcomes,True)
    predicted=predicted_labels(w,mask,outcomes,expanded)
    per_rule=((TABLE == predicted[None,:])*w.weights[None,:]).sum(1)
    return float(np.dot(ptrue,per_rule))


def legal(w:World,mask:int,expanded:bool)->list[int]:
    cost=state_cost(w,mask,expanded)
    x=[i for i in range(NQ) if not(mask&(1<<i)) and cost+w.costs[i]<=BUDGET+1e-9]
    if not expanded and cost+w.expand_cost<=BUDGET+1e-9:x.append(EXPAND)
    return x+[STOP]


class Planner:
    """Perfect-simulator, finite-horizon planning; expensive oracle only."""
    def __init__(self,w:World):self.w=w;self.calls=0

    @lru_cache(maxsize=None)
    def value(self,mask:int=0,outcomes:int=0,expanded:bool=False)->tuple[float,int]:
        self.calls+=1
        spent=state_cost(self.w,mask,expanded)
        best=expected_accuracy(self.w,mask,outcomes,expanded)-PENALTY*spent
        action=STOP
        ptrue=posterior(self.w,mask,outcomes,True)
        for i in legal(self.w,mask,expanded):
            if i==STOP:continue
            if i==EXPAND:
                v,_=self.value(mask,outcomes,True)
            else:
                pos=float(np.dot(ptrue,np.where(TABLE[:,i]==1,1-self.w.noise[i],self.w.noise[i])))
                plus,_=self.value(mask|1<<i,outcomes|1<<i,expanded)
                minus,_=self.value(mask|1<<i,outcomes,expanded)
                v=pos*plus+(1-pos)*minus
            if v>best+1e-10: best,action=v,i
        return best,action


def feature(w:World,mask:int,outcomes:int,expanded:bool)->np.ndarray:
    ps=posterior(w,mask,outcomes,expanded)
    simple=posterior(w,mask,outcomes,False)
    # A model is allowed to use the fixed grammar once expanded. Unexpanded
    # features use simple posterior only. Both see source provenance and data.
    obs=[]
    for i in range(NQ):
        observed=bool(mask&(1<<i))
        obs.extend([float(observed),float((outcomes>>i)&1) if observed else .5,
                    float(w.costs[i]/BUDGET),float(w.noise[i]),float(w.weights[i])])
    slots=np.zeros(12,np.float32)
    slots[:len(ps)]=ps
    # Prior over the six currently unavailable compositions is hidden until expansion.
    prior=np.asarray(w.prior if expanded else np.r_[w.prior[:6], np.zeros(6)],np.float32)
    base=np.concatenate((simple,slots,prior,np.array(obs),[
        float(expanded),w.q,float((BUDGET-state_cost(w,mask,expanded))/BUDGET),
        float(mask.bit_count()/4),float(w.expand_cost/BUDGET),
        float(sum(w.weights[i]*max(float(ps@TABLE[:len(ps),i]),1-float(ps@TABLE[:len(ps),i])) for i in range(NQ)))
    ])).astype(np.float32)
    assert len(base)==56
    return base


def action_mask(w:World,mask:int,expanded:bool)->np.ndarray:
    a=np.zeros(6,np.float32)
    a[legal(w,mask,expanded)]=1
    return a


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(56,72),nn.Tanh(),nn.Linear(72,56),nn.Tanh(),nn.Linear(56,6))
    def forward(self,x,valid):
        return self.net(x).masked_fill(valid<.5,-1e8)


def training_samples(start:int,n:int,seed:int,paths:int=4):
    rng=np.random.default_rng(seed)
    x,v,y=[],[],[]
    diag={"profiles":n,"paths":paths,"teacher_calls":0,"label_count":{a:0 for a in ACTIONS}}
    for s in range(start,start+n):
        w=make_world(s)
        planner=Planner(w)
        for rollout in range(paths):
            mask=outcomes=0;expanded=False
            for turn in range(6):
                lab=planner.value(mask,outcomes,expanded)[1]
                x.append(feature(w,mask,outcomes,expanded))
                v.append(action_mask(w,mask,expanded))
                y.append(lab)
                diag["label_count"][ACTIONS[lab]]+=1
                acts=legal(w,mask,expanded)
                if rollout==0 or rng.random()<.66:
                    chosen=lab
                else:
                    chosen=int(rng.choice(acts))
                if chosen==STOP:break
                if chosen==EXPAND:expanded=True
                else:
                    mask|=1<<chosen
                    outcomes|=int(w.observed[chosen])<<chosen
        diag["teacher_calls"]+=planner.calls
    return np.stack(x),np.stack(v),np.asarray(y,np.int64),diag


def train(train_data,val_data,seed:int,epochs:int=24):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=Policy()
    opt=torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.0001)
    x,v,y=map(torch.from_numpy,train_data[:3]);vx,vv,vy=map(torch.from_numpy,val_data[:3])
    rng=np.random.default_rng(seed+21)
    best=None; best_ce=float('inf');best_epoch=0;history=[]
    for ep in range(epochs):
        model.train()
        for indices in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
            logits=model(x[indices],v[indices]);loss=F.cross_entropy(logits,y[indices])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            train_accuracy=float((model(x,v).argmax(1)==y).float().mean())
            ce=float(F.cross_entropy(model(vx,vv),vy))
            val_accuracy=float((model(vx,vv).argmax(1)==vy).float().mean())
        history.append({"epoch":ep+1,"train_accuracy":train_accuracy,"validation_accuracy":val_accuracy,"validation_CE":ce})
        if ce<best_ce:
            best_ce=ce;best_epoch=ep+1
            best={k:val.detach().clone() for k,val in model.state_dict().items()}
    model.load_state_dict(best);model.eval()
    return model,{"params":sum(p.numel() for p in model.parameters()),"selected_epoch":best_epoch,"history":history}


def action(w:World,arm:str,mask:int,outcomes:int,expanded:bool,policy:Policy|None=None,
           rng:np.random.Generator|None=None,teacher:Planner|None=None)->int:
    available=legal(w,mask,expanded)
    if arm=="oracle_dp":return teacher.value(mask,outcomes,expanded)[1]
    if arm.startswith("policy"):
        z=feature(w,mask,outcomes,expanded)
        if arm=="policy_shuffled_feedback":
            # Keep observed-mask fixed; invert the evidence bits seen by controller
            # while actual inference uses the real observations.
            wrong=outcomes^mask
            z=feature(w,mask,wrong,expanded)
        with torch.no_grad():
            score=policy(torch.from_numpy(z)[None],torch.from_numpy(action_mask(w,mask,expanded))[None])
        return int(score.argmax(1).item())
    if arm=="random":return int(rng.choice(available))
    if arm=="simple_only":
        possible=[i for i in available if i<4]
        if not possible:return STOP
        # Variance under simple models: no representation of missing explanations.
        p=posterior(w,mask,outcomes,False)
        gains=[]
        for i in possible:
            b=float(np.dot(p,TABLE[:6,i]));gains.append((b*(1-b)*w.weights[i]/w.costs[i],i))
        return max(gains)[1] if max(gains)[0]>1e-4 else STOP
    if arm=="always_expand":
        if not expanded and EXPAND in available:return EXPAND
        # After expansion, use myopic conditional improvement in terminal quality.
        # This uses the true generative prior and is therefore a privileged comparator.
        return myopic(w,mask,outcomes,expanded,exclude_expand=True)
    if arm=="myopic_expand":return myopic(w,mask,outcomes,expanded)
    if arm=="surprise_revision":
        if not expanded and EXPAND in available and mask.bit_count()>=2:
            # Threshold fixed before any testing, based on observed evidence under
            # the primitive class. Does not consult hidden truth.
            ll=np.ones(6)
            for i in range(NQ):
                if (mask>>i)&1:
                    ll*=np.where(TABLE[:6,i]==((outcomes>>i)&1),1-w.noise[i],w.noise[i])
            neg_log=-np.log(max(1e-9,float(np.dot(posterior(w,0,0,False),ll))))
            if neg_log>2.0:return EXPAND
        # Query where current simple posterior is most uncertain, then stop.
        opts=[i for i in available if i<4]
        if not opts:return STOP
        p=posterior(w,mask,outcomes,expanded)
        rates=p@TABLE[:len(p),:]
        scores=[(float(rates[i]*(1-rates[i])*w.weights[i]/w.costs[i]),i) for i in opts]
        return max(scores)[1] if max(scores)[0]>.0001 else STOP
    raise ValueError(arm)


def myopic(w:World,mask:int,outcomes:int,expanded:bool,exclude_expand:bool=False)->int:
    # Privileged one-step oracle comparator, same probe and expansion costs.
    score0=expected_accuracy(w,mask,outcomes,expanded)-PENALTY*state_cost(w,mask,expanded)
    chosen=STOP; best=score0
    post=posterior(w,mask,outcomes,True)
    for a in legal(w,mask,expanded):
        if a==STOP or (exclude_expand and a==EXPAND):continue
        if a==EXPAND:
            v=expected_accuracy(w,mask,outcomes,True)-PENALTY*state_cost(w,mask,True)
        else:
            p1=float(np.dot(post,np.where(TABLE[:,a]==1,1-w.noise[a],w.noise[a])))
            plus=expected_accuracy(w,mask|1<<a,outcomes|1<<a,expanded)
            minus=expected_accuracy(w,mask|1<<a,outcomes,expanded)
            v=p1*plus+(1-p1)*minus-PENALTY*(state_cost(w,mask,expanded)+w.costs[a])
        if v>best+1e-10:best=v;chosen=a
    return chosen


def evaluate_episode(w:World,arm:str,model:Policy|None=None)->dict:
    mask=0;outcomes=0;expanded=False;acts=[]
    rng=np.random.default_rng(w.seed+34141)
    teacher=Planner(w) if arm=="oracle_dp" else None
    for _ in range(6):
        a=action(w,arm,mask,outcomes,expanded,model,rng,teacher)
        if a not in legal(w,mask,expanded):raise RuntimeError(f"illegal action {a} {arm}")
        if a==STOP:break
        acts.append(ACTIONS[a])
        if a==EXPAND:expanded=True
        else:
            mask|=1<<a
            outcomes|=int(w.observed[a])<<a
    pred=predicted_labels(w,mask,outcomes,expanded)
    correct=float(((pred==TABLE[w.truth])*w.weights).sum())
    spent=state_cost(w,mask,expanded)
    return {"seed":w.seed,"arm":arm,"truth":w.truth,"truth_family":"composed" if w.truth>=6 else "simple",
        "weighted_accuracy":correct,"utility":float(correct-PENALTY*spent),"cost":spent,
        "expanded":int(expanded),"probes":mask.bit_count(),"actions":acts,"predictions":pred.tolist(),
        "evidence_mask":mask,"observations":outcomes}


def compare(records:list[dict],a:str,b:str,metric:str,boot:int=2000,seed:int=314159)->dict:
    A={r['seed']:r[metric] for r in records if r['arm']==a}; B={r['seed']:r[metric] for r in records if r['arm']==b}
    keys=sorted(A.keys()&B.keys());d=np.array([A[k]-B[k] for k in keys])
    g=np.random.default_rng(seed);draw=g.integers(0,len(d),size=(boot,len(d)))
    return {"difference":float(d.mean()),"paired_95pct_CI":list(map(float,np.quantile(d[draw].mean(1),[.025,.975]))),"n":len(d)}


def output_json(path:Path,x):path.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--train",type=int,default=1400)
    parser.add_argument("--val",type=int,default=250)
    parser.add_argument("--test",type=int,default=1200)
    parser.add_argument("--confirm",type=int,default=1000)
    parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--seed",type=int,default=20261103)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    xd=training_samples(TRAIN_START,args.train,args.seed,paths=4)
    vd=training_samples(VAL_START,args.val,args.seed+1,paths=2)
    model,fit=train(xd,vd,args.seed,epochs=args.epochs)
    torch.save(model.state_dict(),out/"policy.pt")
    arms=["simple_only","random","surprise_revision","always_expand","myopic_expand","policy_shuffled_feedback","policy","oracle_dp"]
    summary={}
    for group,first,n in [("heldout",TEST_START,args.test),("confirmation",CONFIRM_START,args.confirm)]:
        rows=[]
        for s in range(first,first+n):
            w=make_world(s)
            for arm in arms:
                rows.append(evaluate_episode(w,arm,model))
        with (out/f"{group}_records.jsonl").open("w") as f:
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
        summary[group]={
            "seed_range":[first,first+n-1],"n_worlds":n,
            "means":{arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ("weighted_accuracy","utility","cost","expanded","probes")} for arm in arms},
            "family_accuracy":{arm:{fam:float(np.mean([r['weighted_accuracy'] for r in rows if r['arm']==arm and r['truth_family']==fam])) for fam in ("simple","composed")} for arm in arms},
            "comparisons":{arm:{k:compare(rows,"policy",arm,k) for k in ("utility","weighted_accuracy","cost")} for arm in arms if arm!="policy"},
            "family_counts":{fam:sum(1 for r in rows if r['arm']=="policy" and r['truth_family']==fam) for fam in ("simple","composed")},
        }
    sec=time.perf_counter()-start
    output_json(out/"metrics.json",{"status":"synthetic_CPU_exploratory_H-M1","hypothesis":"learned controller can use paid DSL expansion after multiple probes and improve held-out terminal accuracy per cost","summary":summary,"train":{"samples":len(xd[2]),"diagnostics":xd[3]},"validation":{"samples":len(vd[2]),"diagnostics":vd[3]},"fit":fit,"elapsed_seconds":sec,"semantics":{"prediction":"noiseless truth table weighted by world-specific query importance","utility":f"weighted_accuracy - {PENALTY}*total_probe_and_expand_cost","teacher":"exact-Bayesian finite horizon planning over known hypothesis class and observation noise","expansion":"paid enumeration of 6 predeclared Boolean compositions, with full replay of earlier observations"}})
    manifest={"project":"Algernon","hypothesis":"H-M1 / framework revision","status":"CPU synthetic exploratory","seed_ranges":{"train":[TRAIN_START,TRAIN_START+args.train-1],"validation":[VAL_START,VAL_START+args.val-1],"test":[TEST_START,TEST_START+args.test-1],"confirmation":[CONFIRM_START,CONFIRM_START+args.confirm-1]},"seed":args.seed,"budget":BUDGET,"cost_penalty":PENALTY,"grammar":{"primitive":SIMPLE,"compositions":COMPOSED},"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"device":"CPU","parameter_count":sum(p.numel() for p in model.parameters()),"seconds":sec}
    output_json(out/"manifest.json",manifest)
    print(json.dumps({"train_states":len(xd[2]),"n_parameters":manifest['parameter_count'],"seconds":round(sec,2),"summary":{group:{a:{k:round(v,4) for k,v in stats.items() if k in ('weighted_accuracy','utility','cost','expanded')} for a,stats in dat['means'].items()} for group,dat in summary.items()}},indent=2))

if __name__=="__main__":main()
