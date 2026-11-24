"""Budgeted certificate search and small learned local action/repair scorers."""
import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import corpus as C

POLICIES=("lex","occurrence","jw")
SEEDS=(81,83)
BRANCH_DIM=12
REPAIR_DIM=12

class Budget:
    def __init__(self,n):self.limit=n;self.used=0;self.scans=0
    def take(self):
        if self.used>=self.limit:return False
        self.used+=1;return True

def simplify(cs,assn,budget=None):
    """Solver-side unit propagation, separate from certificate checker."""
    while True:
        reduced=[]
        for clause in cs:
            if budget is not None:budget.scans+=1
            rem=[];sat=False
            for l in clause:
                v=abs(l)
                if v in assn:
                    if assn[v]==(l>0):sat=True;break
                else:rem.append(l)
            if not sat:
                if not rem:return None
                reduced.append(rem)
        units=[c[0] for c in reduced if len(c)==1]
        if not units:return reduced
        for l in units:
            v=abs(l);val=l>0
            if v in assn and assn[v]!=val:return None
            assn[v]=val

def candidates(reduced,n,assn):
    return [l for v in sorted({abs(l) for c in reduced for l in c}) if v not in assn for l in (v,-v)]

def features(reduced,n,assn,l):
    total=max(1,len(reduced))
    pos=[len(c) for c in reduced if l in c]
    neg=[len(c) for c in reduced if -l in c]
    def jw(ls):return sum(2.**(-k) for k in ls)
    return [len(pos)/total,len(neg)/total,jw(pos)/total,jw(neg)/total,
            sum(k==2 for k in pos)/total,sum(k==2 for k in neg)/total,
            sum(k==3 for k in pos)/total,sum(k==3 for k in neg)/total,
            len(assn)/n,n/16,len(reduced)/64,min(map(len,reduced))/3]

def repair_features(cs,c,witness,n):
    total=max(1,len(cs));flat=Counter(l for c0 in cs for l in c0)
    a=[flat[l] for l in c];b=[flat[-l] for l in c]
    used={abs(l) for c0 in cs for l in c0}
    overlap=[len(set(c)&set(c0)) for c0 in cs]
    return [len(c)/3,sum(a)/total,sum(b)/total,max(a,default=0)/total,max(b,default=0)/total,
            sum(abs(l) in used for l in c)/3,sum(l>0 for l in c)/3,
            sum(witness[abs(l)-1]==(l>0) for l in c)/3,
            max(overlap,default=0)/3,sum(k>=2 for k in overlap)/total,
            len(cs)/64,n/16]

class Scorer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def choose(reduced,n,assn,policy,model,observer):
    actions=candidates(reduced,n,assn)
    fs=[features(reduced,n,assn,l) for l in actions]
    if observer is not None:observer(reduced,dict(assn),actions,fs)
    if model is not None:
        with torch.no_grad():scores=model(torch.tensor(fs)).tolist()
        return actions[max(range(len(actions)),key=lambda i:scores[i])]
    if policy=="lex":return actions[0]
    if policy=="occurrence":return actions[max(range(len(actions)),key=lambda i:fs[i][0]+fs[i][1]+.001*fs[i][0])]
    assert policy=="jw"
    return actions[max(range(len(actions)),key=lambda i:fs[i][2])]

def solve(cs,n,budget,policy,model=None,observer=None):
    def search(residual,assn):
        if not budget.take():return "OPEN",None
        reduced=simplify(residual,assn,budget)
        if reduced is None:return "UNSAT",{"conflict":True}
        if not reduced:return "SAT",[assn.get(i,False) for i in range(1,n+1)]
        l=choose(reduced,n,assn,policy,model,observer)
        v=abs(l);first=l>0
        statuses={};trees={}
        for val in (first,not first):
            status,proof=search(reduced,assn|{v:val})
            statuses[val]=status;trees[val]=proof
            if status=="SAT":return status,proof
            if status=="OPEN":return status,None
        assert all(s=="UNSAT" for s in statuses.values())
        return "UNSAT",{"var":v,"false":trees[False],"true":trees[True]}
    return search(cs,{})

def repair_order(f,a,cs,witness,policy,model):
    js=[j for j in a["omitted"] if not C.satisfied(f["clauses"][j],witness)]
    if not js:return []
    fs=[repair_features(cs,f["clauses"][j],witness,f["n"]) for j in js]
    if model is not None:
        with torch.no_grad():scores=model(torch.tensor(fs)).tolist()
    elif policy=="lex":scores=[0.]*len(js)
    elif policy=="occurrence":scores=[x[1]+x[2] for x in fs]
    else:scores=[x[8]+x[9] for x in fs]
    return [js[i] for i in sorted(range(len(js)),key=lambda i:(-scores[i],js[i]))]

def critique(f,a,cap,policy,branch=None,repair=None,observer=None,repair_observer=None):
    cs=[f["clauses"][j] for j in a["kept"]];n=f["n"];goal=f["goal"]
    budget=Budget(cap)
    status,result=solve(cs+[[-goal]],n,budget,policy,branch,observer)
    if status=="OPEN":cert={"verdict":"ABSTAIN"}
    elif status=="UNSAT":cert={"verdict":"SUPPORTED","proof":result}
    else:
        cert={"verdict":"REFUTED","witness":result}
        if repair_observer is not None:repair_observer(cs,result)
        for j in repair_order(f,a,cs,result,policy,repair):
            s,p=solve(cs+[f["clauses"][j],[-goal]],n,budget,policy,branch,observer)
            if s=="UNSAT":
                cert|={"repair_clause":j,"repair_proof":p};break
            if s=="OPEN":break
    return cert,budget

def teacher_rows(fs):
    groups=[];repairs=[]
    for f in fs:
        if f["split"]!="train":continue
        m=C.Masks(f["n"])
        for a in f["arms"]:
            cs=[f["clauses"][j] for j in a["kept"]]
            target=m.cnf(cs)&m.lit(-f["goal"])
            def observer(reduced,assn,actions,feats):
                live=m.cnf(reduced)
                for v,val in assn.items():live&=m.lit(v if val else -v)
                if live:
                    values=[(live&m.lit(l)).bit_count() for l in actions]
                    best=max(values);labels=[v==best for v in values]
                else:
                    values=[]
                    for l in actions:
                        test=simplify(reduced,assn|{abs(l):l>0})
                        values.append(-1 if test is None else len(test))
                    best=min(values);labels=[v==best for v in values]
                assert any(labels)
                groups.append((feats,labels))
            def repair_observer(retained,w):
                for j in a["omitted"]:
                    c=f["clauses"][j]
                    repairs.append((repair_features(retained,c,w,f["n"]),float(not(target&m.clause(c)))))
            for pol in ("lex","jw"):
                critique(f,a,128,pol,observer=observer,repair_observer=repair_observer)
    assert groups and repairs
    return groups,repairs

def pack(groups):
    N=len(groups);M=max(len(g[0]) for g in groups)
    x=torch.zeros(N,M,BRANCH_DIM);valid=torch.zeros(N,M,dtype=torch.bool);good=torch.zeros_like(valid)
    for i,(fs,ys) in enumerate(groups):
        x[i,:len(fs)]=torch.tensor(fs);valid[i,:len(fs)]=True;good[i,:len(fs)]=torch.tensor(ys)
    return x,valid,good

def train(groups,repairs,seed,out):
    torch.manual_seed(seed);rng=random.Random(seed)
    branch=Scorer(BRANCH_DIM);repair=Scorer(REPAIR_DIM)
    opt=torch.optim.AdamW(list(branch.parameters())+list(repair.parameters()),lr=.003)
    x,valid,good=pack(groups);rx=torch.tensor([z[0] for z in repairs]);ry=torch.tensor([z[1] for z in repairs])
    hist=[]
    for epoch in range(12):
        ids=list(range(len(groups)));rng.shuffle(ids);ri=list(range(len(repairs)));rng.shuffle(ri)
        losses=[]
        for start in range(0,max(len(ids),len(ri)),64):
            bi=ids[start:start+64];rr=ri[start:start+64]
            opt.zero_grad();loss=0.
            if bi:
                z=branch(x[bi]);loss=loss+(torch.logsumexp(z.masked_fill(~valid[bi],-1e9),-1)-torch.logsumexp(z.masked_fill(~good[bi],-1e9),-1)).mean()
            if rr:loss=loss+F.binary_cross_entropy_with_logits(repair(rx[rr]),ry[rr])
            loss.backward();opt.step();losses.append(float(loss.detach()))
        hist.append(sum(losses)/len(losses))
    path=out/f"seed-{seed}.pt"
    torch.save({"branch":branch.state_dict(),"repair":repair.state_dict()},path)
    info={"seed":seed,"parameters":sum(p.numel() for m in (branch,repair) for p in m.parameters()),
          "branch_groups":len(groups),"repair_rows":len(repairs),"loss":hist,
          "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("FIT",json.dumps(info),flush=True)
    return branch.eval(),repair.eval(),info

def native(fs,out,lean):
    for split in C.SPLITS:
        sub=[f for f in fs if f["split"]==split];total=0
        for offset in range(0,len(sub),8):
            texts=[];expected=[]
            for f in sub[offset:offset+8]:
                src,ex=C.native_source(f);texts.append(src);expected+=ex
            p=out/f"{split}-{offset}.lean"
            p.write_text("import Lean\n\n"+C.STEP+"\n"+"\n".join(texts))
            proc=subprocess.run([lean,str(p)],capture_output=True,text=True,timeout=240)
            log=proc.stdout+proc.stderr;p.with_suffix(".log").write_text(log)
            assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",log),log[-7000:]
            found={}
            for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
                assert m[1] not in found
                found[m[1]]={z.strip() for z in (m[3] or "").split(",") if z.strip()}
            assert set(found)==set(expected),(set(expected)-set(found),set(found)-set(expected))
            assert all(not (a-C.ALLOWED) for a in found.values())
            total+=len(expected)
        print("NATIVE",split,len(sub),total,flush=True)

def evaluate(fs,models,out):
    rows=[]
    for split in ("cal","test_depth","test_clutter"):
        for cap in C.BUDGETS:
            for policy,model_pair in [(p,(None,None)) for p in POLICIES]+list(models.items()):
                counts=Counter();details=[];t=time.process_time()
                for f in fs:
                    if f["split"]!=split:continue
                    for a in f["arms"]:
                        cert,b=critique(f,a,cap,policy if policy in POLICIES else "jw",*model_pair)
                        verified=C.verify(f,a,cert)
                        counts["n"]+=1;counts["bad_n"]+=a["bad"]
                        counts["verdict"]+=verified["verdict"];counts["complete"]+=verified["complete"]
                        counts["repair"]+=verified["repair"];counts["abstain"]+=cert["verdict"]=="ABSTAIN"
                        counts["rejected"]+=cert["verdict"]!="ABSTAIN" and not verified["verdict"]
                        counts["rejected_repair"]+="repair_clause" in cert and not verified["repair"]
                        counts["false_accusation"]+=cert["verdict"]=="REFUTED" and not a["bad"]
                        counts["false_support"]+=cert["verdict"]=="SUPPORTED" and a["bad"]
                        counts["nodes"]+=b.used;counts["scans"]+=b.scans
                        assert b.used<=cap
                        assert not (verified["verdict"] and ((cert["verdict"]=="REFUTED")!=a["bad"]))
                        details.append({"id":a["id"],"certificate":cert,"verified":verified,"nodes":b.used,"scans":b.scans})
                row={"split":split,"budget":cap,"policy":policy,**counts,"cpu_seconds":time.process_time()-t}
                rows.append(row)
                (out/f"{split}-{cap}-{policy}.jsonl").write_text("".join(json.dumps(z)+"\n" for z in details))
                if cap in (16,32,128):print("SCORE",json.dumps(row),flush=True)
    totals=Counter()
    for x in rows:
        if x["split"].startswith("test") and x["budget"] in (16,32):totals[x["policy"]]+=x["complete"]
    baseline=max(totals[p] for p in POLICIES)
    gate={"totals":dict(totals),"best_symbolic":baseline,
          "passed":all(totals[f"learned-{s}"]>=baseline+8 for s in SEEDS) and not any(x["rejected"] or x["rejected_repair"] for x in rows)}
    (out/"scores.json").write_text(json.dumps(rows,indent=2));(out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--lean",default="lean");ap.add_argument("--skip-native",action="store_true")
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fs=list(C.families())
    assert len(fs)==192 and len({f["id"] for f in fs})==192
    assert len({a["id"] for f in fs for a in f["arms"]})==768
    (args.out/"families.json").write_text(json.dumps(fs))
    if not args.skip_native:native(fs,args.out,args.lean)
    # Verification self-checks on full-budget exact search, and corrupted witness/repair negatives.
    for f in fs:
        for a in f["arms"]:
            cert,_=critique(f,a,4096,"jw")
            v=C.verify(f,a,cert)
            assert v["complete"],(f["id"],a["id"],cert)
            if a["bad"]:
                wrong=cert["witness"].copy();wrong[f["goal"]-1]=True
                assert not C.verify(f,a,{"verdict":"REFUTED","witness":wrong})["verdict"]
                assert not C.verify(f,a,cert|{"repair_clause":a["kept"][0]})["repair"]
            else:
                assert not C.verify(f,a,{"verdict":"SUPPORTED","proof":{"var":0,"false":{},"true":{}}})["verdict"]
    groups,repairs=teacher_rows(fs)
    print("TRAIN",len(groups),len(repairs),flush=True)
    models={};receipts=[]
    for seed in SEEDS:
        branch,repair,info=train(groups,repairs,seed,args.out)
        models[f"learned-{seed}"]=(branch,repair);receipts.append(info)
    (args.out/"training.json").write_text(json.dumps(receipts,indent=2))
    evaluate(fs,models,args.out)

if __name__=="__main__":main()
