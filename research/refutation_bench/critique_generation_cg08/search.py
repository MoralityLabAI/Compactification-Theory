"""CG-08: metered context/rule selection, learned action sets and native certificate replay."""
import argparse
import json
import random
import time
from collections import Counter,deque
from pathlib import Path
import torch
from torch import nn
import corpus as C

SEEDS=(131,137)
POLICIES=("lex","random","global_first","goal_slice")
FEATURES=16

class Session:
    def __init__(self,case):
        self.case=case
        self.known=[C.initial(case,i) for i in range(4)]
        self.traces=[[] for _ in range(4)]
        self.firings=0;self.scans=0;self.rank_evaluations=0
    def candidates(self):
        out=[];closed=[]
        for ctx,targets in enumerate(C.goals(self.case)):
            known=self.known[ctx]
            if all(t in known for t in targets):
                closed.append(False);continue
            self.scans+=len(self.case["rules"])
            legal=[(ctx,j) for j,r in enumerate(self.case["rules"])
                   if r["head"] not in known and set(r["prem"])<=known]
            out.extend(legal);closed.append(not legal)
        return out,closed
    def step(self,action):
        ctx,j=action;r=self.case["rules"][j]
        assert r["head"] not in self.known[ctx] and set(r["prem"])<=self.known[ctx]
        self.known[ctx].add(r["head"]);self.traces[ctx].append(j);self.firings+=1

def distance(case,targets):
    """Reverse syntactic reachability, without solving any sequent."""
    d={a:0 for a in targets};q=deque(targets)
    incoming={}
    for r in case["rules"]:incoming.setdefault(r["head"],[]).extend(r["prem"])
    while q:
        a=q.popleft()
        for p in incoming.get(a,[]):
            if p not in d:d[p]=d[a]+1;q.append(p)
    return d
def features(session,actions):
    c=session.case;n=c["n"];den=max(1,len(c["rules"]))
    targets=[set(ts)-k for ts,k in zip(C.goals(c),session.known)]
    ds=[distance(c,ts) for ts in targets]
    fan=Counter(p for r in c["rules"] for p in r["prem"])
    count=Counter(ctx for ctx,_ in actions)
    settled=sum(a in session.known[ctx] for ctx,a in C.source_queries(c))
    out=[]
    for ctx,j in actions:
        r=c["rules"][j];h=r["head"];d=ds[ctx].get(h,n)
        f=[float(ctx==i) for i in range(4)]
        f.extend([len(session.known[ctx])/n,len(targets[ctx])/3,
                  float(h in targets[ctx]),d/n,float(h in ds[ctx]),
                  len(r["prem"])/3,fan[h]/den,count[ctx]/den,
                  float(h in c["roles"]),float(c["roles"][-1] in session.known[0]),
                  settled/5,len(session.traces[ctx])/n])
        assert len(f)==FEATURES;out.append(f)
    return out,ds

def teacher_plan(session):
    """Full-closure teacher; this function is never used by evaluation actors."""
    case=session.case;queries=C.source_queries(case)
    closure=[C.extend(case,k) for k in session.known]
    H=case["roles"][-1]
    def plan(ctx,targets):
        return {(ctx,j) for j in C.needed(case,session.known[ctx],closure[ctx][1],targets)}
    if all(a in closure[ctx][0] for ctx,a in queries):
        result=set()
        for ctx in range(4):
            result|=plan(ctx,[a for c,a in queries if c==ctx])
        return result
    if H not in closure[0][0]:
        return {(0,j) for j in closure[0][1]}
    base=plan(0,[H]);options=[]
    for ctx,a in queries:
        if a not in closure[ctx][0]:
            options.append(base|{(ctx,j) for j in closure[ctx][1]})
    assert options
    return min(options,key=lambda x:(len(x),sorted(x)))

def training_rows(rows):
    data=[];trajectories=0
    for row in rows:
        if row["split"]!="train":continue
        case=C.public(row);ses=Session(case);rng=random.Random(int(row["id"],16))
        for _ in range(256):
            actions,closed=ses.candidates()
            cert=C.certificate(case,ses.known,ses.traces,closed)
            if cert is not None:break
            assert actions
            wanted=teacher_plan(ses)
            fs,_=features(ses,actions);good=[a in wanted for a in actions]
            assert any(good),(row["id"],actions,wanted)
            data.append((fs,good))
            candidates=[i for i,x in enumerate(good) if x]
            index=rng.randrange(len(actions)) if rng.random()<.15 else rng.choice(candidates)
            ses.step(actions[index])
        else:raise AssertionError("train trajectory exhausted")
        assert C.verify(case,cert) and cert["kind"]==row["expected"]
        trajectories+=1
    assert trajectories==256 and data
    return data,trajectories

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)
def fit(data,seed,out):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Ranker();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    history=[]
    for _ in range(12):
        ids=list(range(len(data)));rng.shuffle(ids);losses=[]
        for off in range(0,len(ids),64):
            rows=[data[i] for i in ids[off:off+64]]
            width=max(len(x[0]) for x in rows)
            x=torch.zeros(len(rows),width,FEATURES)
            valid=torch.zeros(len(rows),width,dtype=torch.bool);good=torch.zeros_like(valid)
            for i,(f,y) in enumerate(rows):
                x[i,:len(f)]=torch.tensor(f);valid[i,:len(f)]=True;good[i,:len(f)]=torch.tensor(y)
            z=model(x)
            loss=(torch.logsumexp(z.masked_fill(~valid,-1e9),-1)-torch.logsumexp(z.masked_fill(~good,-1e9),-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    path=out/f"ranker-{seed}.pt";torch.save(model.state_dict(),path)
    info={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
          "action_groups":len(data),"candidate_rows":sum(len(x[0]) for x in data),
          "loss":history,"sha256":C.sha(path.read_bytes())}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

@torch.no_grad()
def search(case,budget,policy,seed,model=None):
    ses=Session(case);rng=random.Random(seed)
    for _ in range(budget+1):
        actions,closed=ses.candidates()
        cert=C.certificate(case,ses.known,ses.traces,closed)
        if cert is not None or ses.firings==budget or not actions:break
        if model is not None:
            fs,_=features(ses,actions);scores=model(torch.tensor(fs)).tolist()
            ses.rank_evaluations+=len(actions)
            idx=max(range(len(actions)),key=lambda i:scores[i])
        elif policy=="random":idx=rng.randrange(len(actions))
        elif policy=="lex":
            idx=min(range(len(actions)),key=lambda i:(actions[i][1],actions[i][0]))
        elif policy=="global_first":idx=0
        elif policy=="goal_slice":
            _,ds=features(ses,actions);ses.rank_evaluations+=len(actions)
            idx=min(range(len(actions)),key=lambda i:(ds[actions[i][0]].get(case["rules"][actions[i][1]]["head"],case["n"]),actions[i]))
        else:raise ValueError(policy)
        ses.step(actions[idx])
    assert ses.firings<=budget
    return cert,{"firings":ses.firings,"rule_scans":ses.scans,"rank_evaluations":ses.rank_evaluations}

def evaluate(rows,models,out):
    scores=[];details=[];accepted={}
    for split in ("cal","test_depth","test_bridge"):
        subset=[r for r in rows if r["split"]==split]
        for budget in C.BUDGETS:
            for policy,model in [(p,None) for p in POLICIES]+list(models.items()):
                counts=Counter();start=time.process_time()
                for row in subset:
                    case=C.public(row)
                    cert,stats=search(case,budget,policy,int(row["id"],16),model)
                    ok=cert is not None and C.verify(case,cert)
                    counts["n"]+=1;counts["complete"]+=ok
                    counts["expected_"+row["expected"]]+=1
                    counts["abstain"]+=cert is None
                    counts["invalid"]+=cert is not None and not ok
                    assert cert is None or (ok and cert["kind"]==row["expected"]),(row["id"],cert,row["expected"])
                    if ok:
                        counts["correct_"+cert["kind"]]+=1
                        accepted.setdefault(row["id"],{})[C.sha(json.dumps(cert,sort_keys=True))]=cert
                    if cert is not None:
                        counts["false_accusation"]+=cert["kind"]=="REFUTED" and row["expected"]!="REFUTED"
                    for k,v in stats.items():counts[k]+=v
                    details.append({"id":row["id"],"split":split,"expected":row["expected"],
                                    "budget":budget,"policy":policy,"certificate":cert,"verified":ok,**stats})
                res={"split":split,"budget":budget,"policy":policy,**counts,"cpu_seconds":time.process_time()-start}
                for k in C.KINDS:res.setdefault("correct_"+k,0)
                scores.append(res);print("SCORE",json.dumps(res),flush=True)
    totals=Counter()
    for x in scores:
        if x["split"].startswith("test") and x["budget"] in (16,32):totals[x["policy"]]+=x["complete"]
    best=max(totals[p] for p in POLICIES)
    gate={"totals":dict(totals),"best_symbolic":best,
          "passed":all(totals[f"learned-{s}"]>=best+6 for s in SEEDS) and not any(x["invalid"] for x in scores)}
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)
    return {k:[v[z] for z in sorted(v)] for k,v in accepted.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=C.cases();native=C.Native(args.source,args.out,args.lean)
    refs={};inputs=[];gold=[]
    for row in rows:
        case=C.public(row);ref=C.reference(case)
        assert C.verify(case,ref) and ref["kind"]==row["expected"]
        refs[row["id"]]=[ref]
        inputs.append({"id":row["id"],"source":native.raw,"source_sha256":native.source_hash,
                       "selected":C.SELECTED,"advertised":C.prop(case,0,case["roles"][-1]),
                       "interface":case})
        gold.append({"id":row["id"],"family":row["family"],"split":row["split"],"certificate":ref})
        assert not C.verify(case,{"kind":"REFUTED","model":list(range(case["n"]))})
        cert,_=search(case,256,"global_first",0)
        assert C.verify(case,cert) and cert["kind"]==row["expected"]
    for name,records in (("inputs",inputs),("gold",gold)):
        (args.out/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records))
    assert native.replay(rows,refs,"reference")==512
    data,n=training_rows(rows);print("TRAIN",n,len(data),flush=True)
    models={};fits=[]
    for seed in SEEDS:
        model,info=fit(data,seed,args.out);models[f"learned-{seed}"]=model;fits.append(info)
    (args.out/"training.json").write_text(json.dumps(fits,indent=2))
    accepted=evaluate(rows,models,args.out)
    count=native.replay([r for r in rows if r["split"]!="train"],accepted,"actor")
    assert count==sum(map(len,accepted.values()))
    print("NATIVE_GATE",count,"distinct actor certificates",flush=True)
if __name__=="__main__":main()
