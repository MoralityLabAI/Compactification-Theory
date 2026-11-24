"""CG-10: joint metered proof provenance and allowed derivability."""
import argparse
import json
import random
import time
from collections import Counter,deque
from pathlib import Path
import torch
from torch import nn
import corpus as C

SEEDS=(151,157)
POLICIES=("provenance_first","allowed_first","alternating","goal_slice")
FEATURES=18

def distances(c):
    ds={c["goal"]:0};q=deque(ds)
    incoming={}
    for e in c["env"]:
        if e["allowed"]:incoming.setdefault(e["head"],[]).extend(e["prem"])
    while q:
        h=q.popleft()
        for a in incoming.get(h,[]):
            if a not in ds:ds[a]=ds[h]+1;q.append(a)
    return ds

class Session:
    def __init__(self,c):
        self.c=c;self.read=set();self.paths={c["root"]:[c["root"]]}
        self.known=C.initial(c);self.trace=[]
        self.prov=None;self.allowed=None;self.reads=0;self.firings=0;self.scans=0;self.ranks=0
        self.ds=distances(c);self.events=[]
    @property
    def cost(self):return self.reads+self.firings
    def candidates(self):
        actions=[]
        if self.prov is None:
            unread=sorted(set(self.paths)-self.read)
            if not unread:self.prov={"status":"CLEAN","closure":sorted(self.read)}
            else:actions.extend(("READ",j) for j in unread)
        if self.allowed is None:
            if self.c["goal"] in self.known:
                self.allowed={"status":"DERIVABLE","trace":C.proof_slice(self.c,C.initial(self.c),self.trace,self.c["goal"])}
            else:
                self.scans+=len(self.c["env"])
                legal=C.legal(self.c,self.known)
                if not legal:self.allowed={"status":"NOT_DERIVABLE","model":sorted(self.known)}
                else:actions.extend(("FIRE",j) for j in legal)
        return actions
    def step(self,action):
        typ,j=action
        if typ=="READ":
            assert self.prov is None and j in self.paths and j not in self.read
            self.read.add(j);self.reads+=1
            node=self.c["nodes"][j]
            self.events.append({"action":typ,"node":j,"deps":node["deps"],"forbidden":C.forbidden(self.c,j)})
            if C.forbidden(self.c,j):self.prov={"status":"TAINTED","path":self.paths[j]}
            else:
                for k in node["deps"]:
                    self.paths.setdefault(k,self.paths[j]+[k])
        else:
            assert typ=="FIRE" and self.allowed is None
            e=self.c["env"][j]
            assert e["allowed"] and e["head"] not in self.known and set(e["prem"])<=self.known
            self.known.add(e["head"]);self.trace.append(j);self.firings+=1
            self.events.append({"action":typ,"rule":j})
    def cert(self):return {"provenance":self.prov,"allowed":self.allowed}
    def complete(self):return self.prov is not None and self.allowed is not None
    def stats(self):
        return {"work":self.cost,"reads":self.reads,"firings":self.firings,
                "scans":self.scans,"rank_evaluations":self.ranks}

def features(s,actions):
    c=s.c;n=max(1,c["n"]);m=max(1,len(c["nodes"]));e=max(1,len(c["env"]))
    read_count=sum(t=="READ" for t,_ in actions);fire_count=len(actions)-read_count
    result=[]
    for typ,j in actions:
        # READ candidates contain discovered index and actual exposed path length;
        # no node content or policy tag is queried before paying for the read.
        is_read=typ=="READ"
        depth=len(s.paths[j])/m if is_read else 0.
        d=n if is_read else s.ds.get(c["env"][j]["head"],n)
        prem=0 if is_read else len(c["env"][j]["prem"])
        immediate=0 if is_read else float(c["env"][j]["head"]==c["goal"])
        row=[float(is_read),float(not is_read),float(s.prov is not None),float(s.allowed is not None),
             len(s.read)/m,len(s.paths)/m,read_count/m,fire_count/e,len(s.known)/n,
             s.firings/e,depth,j/(m if is_read else e),d/n,float(d<n),prem/3,
             immediate,len(s.trace)/e,s.cost/(m+e)]
        assert len(row)==FEATURES;result.append(row)
    return result

def teacher_plan(s,actions):
    c=s.c;wanted=set()
    if s.prov is None:
        # Multi-source BFS from the exposed unread frontier. Every hidden leaf
        # inspection is teacher-only; the actor sees only discovered pointers.
        frontier=[j for typ,j in actions if typ=="READ"]
        q=deque((j,j,0) for j in frontier);seen=set();first=None
        while q:
            j,start,depth=q.popleft()
            if j in seen:continue
            seen.add(j)
            if C.forbidden(c,j):first=start;break
            for k in c["nodes"][j]["deps"]:
                if k not in s.read:q.append((k,start,depth+1))
        if first is None:wanted.update(("READ",j) for j in frontier)
        else:wanted.add(("READ",first))
    if s.allowed is None:
        known,trace=C.closure(c,s.known)
        if c["goal"] in known:trace=C.proof_slice(c,s.known,trace,c["goal"])
        wanted.update(("FIRE",j) for j in trace)
    good=[a in wanted for a in actions]
    assert any(good),(actions,wanted)
    return good

def training_rows(rows):
    data=[];n=0
    for row in rows:
        if row["split"]!="train":continue
        s=Session(C.public(row));rng=random.Random(int(row["id"],16))
        for _ in range(129):
            actions=s.candidates()
            if s.complete():break
            assert actions and s.cost<128
            fs=features(s,actions);good=teacher_plan(s,actions)
            data.append((fs,good))
            candidates=[i for i,x in enumerate(good) if x]
            i=rng.randrange(len(actions)) if rng.random()<.15 else rng.choice(candidates)
            s.step(actions[i])
        assert s.complete() and C.verify(s.c,s.cert()) and C.classify(s.cert())==row["expected"]
        n+=1
    assert n==256 and data
    return data

class Ranker(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,1))
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
            x=torch.zeros(len(rows),width,FEATURES);valid=torch.zeros(len(rows),width,dtype=torch.bool);good=torch.zeros_like(valid)
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
    print("FIT",json.dumps(info),flush=True);return model.eval(),info

@torch.no_grad()
def search(c,budget,policy,model=None):
    s=Session(c)
    while True:
        actions=s.candidates()
        if s.complete() or s.cost==budget:break
        assert actions
        def lex(i):return actions[i][1]
        if model is not None:
            scores=model(torch.tensor(features(s,actions))).tolist();s.ranks+=len(actions)
            idx=max(range(len(actions)),key=lambda i:scores[i])
        elif policy in ("provenance_first","allowed_first","alternating"):
            preferred=("READ" if policy=="provenance_first" else "FIRE" if policy=="allowed_first"
                       else ("READ" if s.reads<=s.firings else "FIRE"))
            idx=min(range(len(actions)),key=lambda i:(actions[i][0]!=preferred,lex(i)))
        elif policy=="goal_slice":
            s.ranks+=len(actions)
            def score(i):
                typ,j=actions[i]
                if typ=="READ":return (0,-len(s.paths[j]),j)
                return (1,s.ds.get(c["env"][j]["head"],c["n"]),j)
            idx=min(range(len(actions)),key=score)
        else:raise ValueError(policy)
        s.step(actions[idx])
    assert s.cost<=budget
    return s.cert(),s.stats(),s.events

def evaluate(rows,models,out):
    scores=[];details=[];accepted={}
    for split in ("cal","test_depth","test_clutter"):
        subset=[r for r in rows if r["split"]==split]
        for budget in C.BUDGETS:
            for name,model in [(p,None) for p in POLICIES]+list(models.items()):
                counts=Counter();start=time.process_time()
                for r in subset:
                    c=C.public(r);cert,stats,events=search(c,budget,name,model)
                    p=cert["provenance"];a=cert["allowed"]
                    vp=p is not None and C.verify_prov(c,p)
                    va=a is not None and C.verify_allowed(c,a)
                    assert p is None or vp;assert a is None or va
                    ok=vp and va
                    counts["n"]+=1;counts["complete"]+=ok
                    counts["provenance"]+=vp;counts["allowed"]+=va
                    counts["expected_"+r["expected"]]+=1
                    counts["abstain"]+=not ok
                    counts["invalid"]+=int(p is not None and not vp)+int(a is not None and not va)
                    if ok:
                        assert C.classify(cert)==r["expected"]
                        counts["correct_"+r["expected"]]+=1
                        accepted.setdefault(r["id"],{})[C.sha(json.dumps(cert,sort_keys=True))]=cert
                    for k,v in stats.items():counts[k]+=v
                    details.append({"id":r["id"],"split":split,"expected":r["expected"],"budget":budget,
                                    "policy":name,"certificate":cert,"verified":ok,**stats,"events":events})
                result={"split":split,"budget":budget,"policy":name,**counts,"cpu_seconds":time.process_time()-start}
                for kind in C.KINDS:result.setdefault("correct_"+kind,0)
                scores.append(result);print("SCORE",json.dumps(result),flush=True)
    totals=Counter()
    for s in scores:
        if s["split"].startswith("test") and s["budget"] in (16,32):totals[s["policy"]]+=s["complete"]
    best=max(totals[p] for p in POLICIES)
    gate={"totals":dict(totals),"best_symbolic":best,"passed":all(totals[f"learned-{seed}"]>=best+6 for seed in SEEDS),
          "invalid":sum(s["invalid"] for s in scores)}
    assert gate["invalid"]==0
    (out/"scores.json").write_text(json.dumps(scores,indent=2));(out/"gate.json").write_text(json.dumps(gate,indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    print("GATE",json.dumps(gate),flush=True)
    return {k:[v[z] for z in sorted(v)] for k,v in accepted.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=C.cases();native=C.Native(a.out,a.lean);native.source(rows)
    refs={};inputs=[];gold=[]
    for r in rows:
        c=C.public(r);ref=C.reference(c)
        assert C.verify(c,ref) and C.classify(ref)==r["expected"]
        refs[r["id"]]=[ref]
        inputs.append({"id":r["id"],"source":C.raw_source(r),"source_sha256":C.sha(C.raw_source(r)),
                       "selected":"CG10_"+r["id"]+".selected","advertised":C.atom(c["goal"]),"interface":c})
        gold.append({"id":r["id"],"family":r["family"],"split":r["split"],"certificate":ref})
        for pol in POLICIES:
            cert,_,_=search(c,128,pol)
            assert C.verify(c,cert) and C.classify(cert)==r["expected"]
        assert not C.verify_allowed(c,{"status":"NOT_DERIVABLE","model":list(range(c["n"]))})
        assert not C.verify_prov(c,{"status":"CLEAN","closure":[]})
    for name,items in (("inputs",inputs),("gold",gold)):
        (a.out/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    assert native.replay(rows,refs,"reference")==512
    data=training_rows(rows);print("TRAIN",256,len(data),flush=True)
    models={};fits=[]
    for seed in SEEDS:
        model,info=fit(data,seed,a.out);models[f"learned-{seed}"]=model;fits.append(info)
    (a.out/"training.json").write_text(json.dumps(fits,indent=2))
    accepted=evaluate(rows,models,a.out)
    count=native.replay([r for r in rows if r["split"]!="train"],accepted,"actor")
    assert count==sum(map(len,accepted.values()))
    print("NATIVE_GATE",count,"distinct actor certificates",flush=True)

if __name__=="__main__":main()
