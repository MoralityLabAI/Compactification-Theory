"""CG-11 metered elaborated dependency acquisition, with lexical-only controls."""
import argparse
import json
import random
import time
from collections import Counter,deque
from pathlib import Path

import torch
from torch import nn
import corpus as C

SEEDS=(161,167)
POLICIES=("bfs","dfs","lex","lexical_risk")
FEATURES=14

def reverse_dist(graph,targets):
    reverse={}
    for a,bs in graph.items():
        for b in bs:reverse.setdefault(b,[]).append(a)
    ds={a:0 for a in targets};q=deque(sorted(targets))
    while q:
        n=q.popleft()
        for p in reverse.get(n,[]):
            if p not in ds:ds[p]=ds[n]+1;q.append(p)
    return ds

def public(case):
    return {k:case[k] for k in ("root","forbidden","source","surface")}

class Oracle:
    """Only one discovered node at a time is returned to the actor."""
    def __init__(self,graph,root):
        self._graph=graph
        self.discovered={root}
        self.read=set()
    def inspect(self,node):
        assert node in self.discovered and node not in self.read
        x=self._graph[node]
        self.read.add(node)
        self.discovered.update(C.edges(x))
        return x

class Session:
    def __init__(self,case,query):
        self.case=case;self.query=query
        self.paths={case["root"]:[case["root"]]};self.edge_types={}
        self.read={};self.frontier=[case["root"]];self.cert=None
        self.ranks=0;self.scans=0;self.events=[]
        self.bad=set(case["forbidden"]);self.surface=case["surface"]
        self.dist=reverse_dist({k:x["refs"] for k,x in self.surface.items()},self.bad)
    def actions(self):
        return list(self.frontier)
    def step(self,node):
        assert self.cert is None and node in self.frontier and node not in self.read
        self.frontier.remove(node)
        x=self.query(node);self.read[node]=x
        self.events.append({"node":node,"kind":x["kind"],"type":x["type"],"value":x["value"],"axioms":x["axioms"]})
        if C.is_bad(node,x,self.bad):
            self.cert={"status":"TAINTED","path":self.paths[node]}
            return
        deps=C.edges(x);self.scans+=len(deps)
        for child in deps:
            if child not in self.paths:
                self.paths[child]=self.paths[node]+[child]
                self.edge_types[child]=(child in x["type"],child in x["value"])
                self.frontier.append(child)
        if not self.frontier:
            self.cert={"status":"CLEAN","closure":sorted(self.read)}
    def stats(self):
        return {"reads":len(self.read),"edge_scans":self.scans,"rank_evaluations":self.ranks}

def features(s,actions):
    m=max(1,len(s.surface));out=[]
    for n in actions:
        x=s.surface.get(n,{})
        t,v=s.edge_types.get(n,(False,False))
        d=s.dist.get(n,m)
        refs=x.get("refs",[])
        fs=[float(n in s.bad),float(n in s.dist),d/m,
            len(s.paths[n])/m,len(refs)/m,
            sum(r in s.bad for r in refs)/max(1,len(refs)),
            float(x.get("kind")=="axiom"),float(x.get("kind")=="def"),
            float(x.get("tactic",False)),float(x.get("alternative",False)),
            float(x.get("shadow",False)),float(x.get("show",False)),
            float(t),float(v)]
        assert len(fs)==FEATURES;out.append(fs)
    return out

def teacher_dist(graph,bad):
    targets={n for n,x in graph.items() if C.is_bad(n,x,bad)}
    return reverse_dist({n:C.edges(x) for n,x in graph.items()},targets)

def teacher_labels(s,actions,dist):
    if not dist:return [True]*len(actions)
    finite=[dist[n] for n in actions if n in dist]
    assert finite
    best=min(finite)
    return [dist.get(n,-1)==best for n in actions]

def training(rows,graphs):
    data=[];n=0
    for case in rows:
        if case["split"]!="train":continue
        g=graphs[case["id"]];o=Oracle(g,case["root"]);s=Session(public(case),o.inspect)
        dist=teacher_dist(g,case["forbidden"]);rng=random.Random(int(case["id"],16))
        while s.cert is None:
            actions=s.actions();assert actions
            fs=features(s,actions);good=teacher_labels(s,actions,dist)
            labels=[i for i,v in enumerate(good) if v]
            assert labels
            data.append((fs,good))
            i=rng.randrange(len(actions)) if rng.random()<.15 else rng.choice(labels)
            s.step(actions[i])
        assert C.verify(g,case,s.cert)
        assert (s.cert["status"]=="TAINTED")==case["expected"]
        n+=1
    assert n==256 and data
    return data

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def fit(data,seed,out):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Ranker();opt=torch.optim.AdamW(model.parameters(),lr=.003);hist=[]
    for _ in range(12):
        ids=list(range(len(data)));rng.shuffle(ids);losses=[]
        for off in range(0,len(ids),64):
            batch=[data[j] for j in ids[off:off+64]]
            width=max(len(x[0]) for x in batch)
            x=torch.zeros(len(batch),width,FEATURES)
            valid=torch.zeros(len(batch),width,dtype=torch.bool);good=torch.zeros_like(valid)
            for i,(fs,ys) in enumerate(batch):
                x[i,:len(fs)]=torch.tensor(fs);valid[i,:len(fs)]=True;good[i,:len(fs)]=torch.tensor(ys)
            z=model(x)
            loss=(torch.logsumexp(z.masked_fill(~valid,-1e9),-1)-torch.logsumexp(z.masked_fill(~good,-1e9),-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        hist.append(sum(losses)/len(losses))
    path=out/f"ranker-{seed}.pt";torch.save(model.state_dict(),path)
    info={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
          "action_groups":len(data),"candidate_rows":sum(len(x[0]) for x in data),
          "loss":hist,"sha256":C.sha(path.read_bytes())}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

@torch.no_grad()
def search(case,graph,budget,policy,model=None):
    o=Oracle(graph,case["root"]);s=Session(public(case),o.inspect)
    while s.cert is None and len(s.read)<budget:
        actions=s.actions();assert actions
        if model is not None:
            logits=model(torch.tensor(features(s,actions))).tolist();s.ranks+=len(actions)
            n=actions[max(range(len(actions)),key=lambda i:logits[i])]
        elif policy=="bfs":n=actions[0]
        elif policy=="dfs":n=actions[-1]
        elif policy=="lex":n=min(actions)
        elif policy=="lexical_risk":
            s.ranks+=len(actions)
            n=min(actions,key=lambda n:(s.dist.get(n,10**9),-len(s.paths[n]),n))
        else:raise ValueError(policy)
        s.step(n)
    assert len(s.read)<=budget
    return s.cert,s.stats(),s.events

def lexical(case):
    graph={n:x["refs"] for n,x in case["surface"].items()}
    return case["root"] in reverse_dist(graph,set(case["forbidden"]))

def evaluate(rows,graphs,models,out):
    scores=[];details=[];lexical_scores=[]
    for split in ("cal","test_depth","test_syntax"):
        subset=[c for c in rows if c["split"]==split]
        lex=Counter()
        for c in subset:
            guess=lexical(c)
            lex["n"]+=1;lex["clean"]+=not c["expected"];lex["tainted"]+=c["expected"]
            lex["false_accusations"]+=guess and not c["expected"]
            lex["missed_taint"]+=not guess and c["expected"]
        receipt={"split":split,**lex};lexical_scores.append(receipt)
        print("LEXICAL",json.dumps(receipt),flush=True)
        for budget in C.BUDGETS:
            for policy,model in [(p,None) for p in POLICIES]+list(models.items()):
                counts=Counter();start=time.process_time()
                for c in subset:
                    graph=graphs[c["id"]]
                    cert,stats,events=search(c,graph,budget,policy,model)
                    ok=cert is not None and C.verify(graph,c,cert)
                    counts["n"]+=1;counts["clean_n"]+=not c["expected"];counts["tainted_n"]+=c["expected"]
                    counts["complete"]+=ok;counts["abstain"]+=cert is None
                    counts["invalid"]+=cert is not None and not ok
                    assert cert is None or (ok and (cert["status"]=="TAINTED")==c["expected"])
                    counts["clean"]+=ok and not c["expected"];counts["tainted"]+=ok and c["expected"]
                    for k,v in stats.items():counts[k]+=v
                    details.append({"id":c["id"],"split":split,"policy":policy,"budget":budget,
                                    "certificate":cert,"verified":ok,**stats,"events":events})
                result={"split":split,"budget":budget,"policy":policy,**counts,"cpu_seconds":time.process_time()-start}
                scores.append(result);print("SCORE",json.dumps(result),flush=True)
    totals=Counter()
    for s in scores:
        if s["split"].startswith("test") and s["budget"] in (16,32):
            totals[s["policy"]]+=s["complete"]
    best=max(totals[p] for p in POLICIES)
    gate={"totals":dict(totals),"best_symbolic":best,
          "passed":all(totals[f"learned-{s}"]>=best+6 for s in SEEDS),
          "invalid":sum(s["invalid"] for s in scores)}
    assert gate["invalid"]==0
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    (out/"lexical-scores.json").write_text(json.dumps(lexical_scores,indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);a.out=a.out.resolve()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fs=list(C.families());assert len(fs)==128 and len({C.sha(f["source"]) for f in fs})==128
    rows,graphs=C.extract(fs,a.lean,a.out,Path(__file__).with_name("Extract.lean"))
    inputs=[];gold=[]
    for c in rows:
        g=graphs[c["id"]]
        inputs.append({"id":c["id"],**public(c)})
        gold.append({"id":c["id"],"family":c["family"],"split":c["split"],"expected":c["expected"]})
        for policy in POLICIES:
            cert,_,_=search(c,g,max(128,len(g)),policy)
            assert C.verify(g,c,cert) and (cert["status"]=="TAINTED")==c["expected"]
        assert not C.verify(g,c,{"status":"CLEAN","closure":[]})
    for name,items in (("inputs",inputs),("gold",gold)):
        (a.out/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    data=training(rows,graphs);print("TRAIN",256,len(data),flush=True)
    models={};fits=[]
    for seed in SEEDS:
        m,info=fit(data,seed,a.out);models[f"learned-{seed}"]=m;fits.append(info)
    (a.out/"training.json").write_text(json.dumps(fits,indent=2))
    evaluate(rows,graphs,models,a.out)

if __name__=="__main__":main()
