"""Finite semantic premise repair with a hidden countermodel oracle."""
import argparse
import functools
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

SPLITS={"train":(80,104401,(2,3),(4,8)),
        "cal":(16,104402,(2,3),(4,8)),
        "test_depth":(32,104403,(4,5),(4,8)),
        "test_clutter":(32,104404,(2,3),(12,16))}
BUDGETS=(2,4,8,16,32)
POLICIES=("lex","random","public","rarity")
SEEDS=(91,97)
POOL=64
FEATURES=12
ALLOWED={"propext","Classical.choice","Quot.sound"}

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def union(masks,ids):
    result=0
    for i in ids:result|=masks[i]
    return result
def bitrows(mask):
    while mask:
        b=mask&-mask
        yield b
        mask^=b
def choosebit(bits,rng):
    return rng.choice(list(bitrows(bits)))

def generate(split,i,rng,groups,clutter):
    k=rng.randint(*groups)
    n=1+4*k
    rows=list(range(1,n));rng.shuffle(rows)
    anchors=[1<<rows[4*j] for j in range(k)]
    opts=[sum(1<<v for v in rows[4*j+1:4*j+4]) for j in range(k)]
    anchor_mask=sum(anchors);optional=sum(opts)
    masks=[];ref=[];alternatives=[]
    for j in range(k):
        core=anchors[j]|opts[j]
        ref.append(len(masks));masks.append(core)
        extra=choosebit(optional^opts[j],rng)
        alternatives.append(len(masks));masks.append(core|extra)
        # Partial hypotheses sharing the same anchor.
        masks.append(anchors[j]|choosebit(opts[j],rng))
    target=len(masks)+rng.randint(*clutter)
    while len(masks)<target:
        c=0
        if rng.random()<.75:c|=rng.choice(anchors)
        for bit in bitrows(optional):
            if rng.random()<.32:c|=bit
        if c and c not in masks:masks.append(c)
    order=list(range(len(masks)));rng.shuffle(order)
    inverse={old:j for j,old in enumerate(order)}
    masks=[masks[j] for j in order]
    ref=sorted(inverse[j] for j in ref)
    alt=sorted(inverse[j] for j in alternatives)
    assert ref!=alt
    assert all((c&anchor_mask).bit_count()<=1 and not c&1 for c in masks)
    assert union(masks,ref)==union(masks,range(len(masks)))
    ws=[]
    for _ in range(2):
        w=anchor_mask
        for bit in bitrows(optional):
            if rng.random()<.55:w|=bit
        ws.append(w)
    if ws[0]==ws[1]:ws[1]^=choosebit(optional,rng)
    family=f"CG{SPLITS[split][1]}_{i}"
    arms=[]
    for j,w in enumerate([0,*ws]):
        tie=list(range(1,n));rng.shuffle(tie)
        arms.append({"id":sha(family+str(j))[:20],"w":w,"order":tie,
                     "ref":ref if w else [],"min":k if w else 0})
    rng.shuffle(arms)
    return {"id":family,"split":split,"n":n,"masks":masks,"anchors":anchor_mask,
            "ref":ref,"alt":alt,"arms":arms}

def families():
    for split,(n,seed,g,c) in SPLITS.items():
        rng=random.Random(seed)
        for i in range(n):yield generate(split,i,rng,g,c)

def optimal_pool(masks,observed,cap=POOL):
    """Exact cardinality DP and deterministic capped enumeration of minimum covers."""
    if not observed:return 0,[()],1
    choices={bit:tuple(i for i,c in enumerate(masks) if c&bit) for bit in bitrows(observed)}
    assert all(choices.values())
    def pivot(rem):return min(bitrows(rem),key=lambda bit:(len(choices[bit]),bit))
    @functools.lru_cache(None)
    def cost(rem):
        if not rem:return 0
        return 1+min(cost(rem&~masks[i]) for i in choices[pivot(rem)])
    lower=cost(observed)
    results=[];seen=set();visited=set();nodes=0
    def walk(rem,selected):
        nonlocal nodes
        nodes+=1
        if len(results)>=cap:return
        state=(rem,selected)
        if state in visited:return
        visited.add(state)
        if not rem:
            key=tuple(sorted(selected))
            if key not in seen:
                seen.add(key);results.append(key)
            return
        for i in choices[pivot(rem)]:
            nr=rem&~masks[i]
            if cost(nr)+1==cost(rem):
                assert i not in selected
                walk(nr,selected|{i})
                if len(results)>=cap:return
    walk(observed,frozenset())
    assert results and all(len(x)==lower and observed&~union(masks,x)==0 for x in results)
    return lower,results,cost.cache_info().currsize+nodes

def feasible(masks,rows,k):
    """Independent bounded existence check, separate from optimal_pool."""
    @functools.lru_cache(None)
    def check(rem,budget):
        if rem==0:return True
        if budget<=0:return False
        # Different pivot and candidate order from the proposal enumerator.
        bit=rem&-rem
        options=sorted({m&rem for m in masks if m&bit},key=lambda m:(-m.bit_count(),m))
        return any(check(rem^v,budget-1) for v in options)
    return check(rows,k)

class Oracle:
    def __init__(self,masks,arm):
        self._masks=tuple(masks);self._w=arm["w"];self._order=tuple(arm["order"])
    def query(self,proposal):
        assert tuple(sorted(set(proposal)))==proposal
        assert all(0<=i<len(self._masks) for i in proposal)
        residual=self._w&~union(self._masks,proposal)
        if not residual:return None
        return next(j for j in self._order if residual&(1<<j))

def verify(f,a,cert):
    masks=f["masks"];w=a["w"];n=f["n"]
    obs=cert["observed"];lower=cert["lower"];repair=cert["repair"]
    assert type(obs) is list and len(obs)==len(set(obs))
    assert all(type(x) is int and 0<x<n and w&(1<<x) for x in obs)
    rows=sum(1<<j for j in obs)
    assert type(lower) is int and lower>=0
    assert feasible(masks,rows,lower)
    assert not feasible(masks,rows,lower-1) if lower else rows==0
    if repair is None:
        assert obs
        return {"supported":False,"refuted":True,"minimum":False,"lower":lower}
    assert repair==sorted(set(repair)) and len(repair)==lower
    assert all(type(j) is int and 0<=j<len(masks) for j in repair)
    assert w&~union(masks,repair)==0
    assert (not w)==(not repair)
    return {"supported":not w,"refuted":bool(w),"minimum":True,"lower":lower}

def validate(fs):
    assert len(fs)==160 and len({f["id"] for f in fs})==160
    assert len({a["id"] for f in fs for a in f["arms"]})==480
    signatures=set()
    for f in fs:
        masks=f["masks"];anchors=f["anchors"]
        sig=sha(json.dumps([f["n"],masks]))
        assert sig not in signatures;signatures.add(sig)
        assert all(not m&1 and (m&anchors).bit_count()<=1 for m in masks)
        assert len(set(masks))==len(masks)
        for a in f["arms"]:
            w=a["w"]
            assert not w&1 and not w&~union(masks,range(len(masks)))
            if w:
                assert (w&anchors)==anchors
                assert len(f["ref"])==len(f["alt"])==a["min"]==anchors.bit_count()
                assert not w&~union(masks,f["ref"]) and not w&~union(masks,f["alt"])
                assert not feasible(masks,w,a["min"]-1)
            else:assert a["min"]==0
    return signatures

def state_features(masks,n,observed,pool):
    # Public mask information only. Row 0 is the retained safe model.
    frequencies=[sum(bool(c&(1<<j)) for c in masks) for j in range(1,n)]
    weights={1<<j:1/max(1,frequencies[j-1]) for j in range(1,n)}
    denom=max(1,n-1);num=max(1,len(masks))
    obs_count=observed.bit_count()
    result=[]
    for ids in pool:
        u=union(masks,ids);cards=[masks[i].bit_count() for i in ids]
        covered=list(bitrows(u));new=u&~observed
        redundancy=sum(cards)-u.bit_count()
        rare=sum(weights[b] for b in covered)
        rare_new=sum(weights[b] for b in bitrows(new))
        result.append([len(ids)/8,u.bit_count()/denom,new.bit_count()/denom,
                       obs_count/denom,sum(cards)/(denom*8),redundancy/(denom*8),
                       min(cards,default=0)/denom,max(cards,default=0)/denom,
                       rare/denom,rare_new/denom,
                       sum(sum(bool(masks[i]&b) for i in ids)>1 for b in covered)/denom,
                       len(pool)/POOL])
    return result

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)

@torch.no_grad()
def run(masks,n,query,budget,policy,seed,model=None,observer=None):
    rng=random.Random(seed)
    obs=[];observed=0;proposal=()
    pool_count=0;states=0;steps=[]
    for call in range(1,budget+1):
        lower,pool,work=optimal_pool(masks,observed)
        pool_count+=len(pool);states+=work
        fs=state_features(masks,n,observed,pool)
        if observer is not None:observer(pool,fs)
        if model is not None:
            scores=model(torch.tensor(fs)).tolist()
            idx=max(range(len(pool)),key=lambda j:scores[j])
        elif policy=="lex":idx=0
        elif policy=="random":idx=rng.randrange(len(pool))
        elif policy=="public":idx=max(range(len(pool)),key=lambda j:fs[j][1])
        elif policy=="rarity":idx=max(range(len(pool)),key=lambda j:fs[j][8])
        else:raise ValueError(policy)
        proposal=pool[idx]
        row=query(proposal)
        steps.append({"proposal":list(proposal),"witness":row,"lower":lower,"pool_size":len(pool)})
        if row is None:
            return {"observed":obs,"repair":list(proposal),"lower":lower},{"calls":call,"pools":pool_count,"states":states,"steps":steps}
        assert 0<row<n and not observed&(1<<row) and not union(masks,proposal)&(1<<row)
        obs.append(row);observed|=1<<row
    lower,pool,work=optimal_pool(masks,observed)
    return {"observed":obs,"repair":None,"lower":lower},{"calls":budget,"pools":pool_count+len(pool),"states":states+work,"steps":steps}

def teacher(fs):
    data=[]
    for f in fs:
        if f["split"]!="train":continue
        masks=f["masks"]
        for a in f["arms"]:
            def observe(pool,features):
                scores=[(union(masks,ids)&a["w"]).bit_count() for ids in pool]
                best=max(scores);data.append((features,[x==best for x in scores]))
            for p in ("lex","public"):
                cert,stats=run(masks,f["n"],Oracle(masks,a).query,32,p,int(a["id"],16),observer=observe)
                v=verify(f,a,cert)
                if v["minimum"]:assert len(cert["repair"])==a["min"]
    return data

def train(data,seed,out):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Ranker();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    history=[]
    for epoch in range(12):
        order=list(range(len(data)));rng.shuffle(order);losses=[]
        for start in range(0,len(order),64):
            rows=[data[i] for i in order[start:start+64]]
            k=max(len(x[0]) for x in rows)
            x=torch.zeros(len(rows),k,FEATURES)
            valid=torch.zeros(len(rows),k,dtype=torch.bool);good=torch.zeros_like(valid)
            for i,(f,g) in enumerate(rows):
                x[i,:len(f)]=torch.tensor(f);valid[i,:len(f)]=True;good[i,:len(f)]=torch.tensor(g)
            z=model(x)
            loss=(torch.logsumexp(z.masked_fill(~valid,-1e9),-1)-torch.logsumexp(z.masked_fill(~good,-1e9),-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    path=out/f"ranker-{seed}.pt";torch.save(model.state_dict(),path)
    info={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),"groups":len(data),
          "loss":history,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

def native(fs,out,lean,accepted=None):
    def emit_union(ids,masks):
        if not ids:return "(0 : Nat)"
        return "("+" ||| ".join(f"c{j}" for j in ids)+")"
    for split in SPLITS:
        sub=[f for f in fs if f["split"]==split]
        total=0
        for off in range(0,len(sub),8):
            lines=["import Lean","set_option maxRecDepth 4096"]
            expected=[]
            for f in sub[off:off+8]:
                ns=f["id"];masks=f["masks"];anchors=f["anchors"]
                lines.append("namespace "+ns)
                for j,m in enumerate(masks):lines.append(f"def c{j} : Nat := {m}")
                def thm(name,prop):
                    lines.append(f"theorem {name} : {prop} := by decide")
                    lines.append("#print axioms "+name);expected.append(ns+"."+name)
                full=emit_union(range(len(masks)),masks)
                thm("safe",f"Nat.testBit {full} 0 = false")
                for j,c in enumerate(masks):
                    a=c&anchors
                    assert a.bit_count()<=1
                    thm(f"anchorLimit{j}",f"(c{j} &&& {anchors}) = {a}")
                for k,a in enumerate(f["arms"]):
                    w=a["w"];lines.append(f"def w{k} : Nat := {w}")
                    thm(f"full{k}",f"({full} &&& w{k}) = w{k}")
                    ref=emit_union(a["ref"],masks)
                    thm(f"ref{k}",f"({ref} &&& w{k}) = w{k}")
                    if w:
                        alt=emit_union(f["alt"],masks)
                        thm(f"alt{k}",f"({alt} &&& w{k}) = w{k}")
                        thm(f"anchors{k}",f"(w{k} &&& {anchors}) = {anchors}")
                    if accepted is not None:
                        for j,ids in enumerate(sorted(accepted.get(a["id"],set()))):
                            u=emit_union(ids,masks)
                            thm(f"actor{k}_{j}",f"({u} &&& w{k}) = w{k}")
                lines.append("end "+ns)
            tag="actor" if accepted is not None else "reference"
            path=out/f"{tag}-{split}-{off}.lean";path.write_text("\n".join(lines)+"\n")
            proc=subprocess.run([lean,str(path)],capture_output=True,text=True,timeout=240)
            log=proc.stdout+proc.stderr;path.with_suffix(".log").write_text(log)
            assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",log),log[-7000:]
            found={}
            for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
                assert m[1] not in found
                found[m[1]]={x.strip() for x in (m[3] or "").split(",") if x.strip()}
            assert set(found)==set(expected),(set(expected)-set(found),set(found)-set(expected))
            assert all(not (x-ALLOWED) for x in found.values())
            total+=len(expected)
        print("NATIVE",tag,split,len(sub),total,flush=True)

def evaluate(fs,models,out):
    scores=[];accepted={}
    for split in ("cal","test_depth","test_clutter"):
        for budget in BUDGETS:
            for policy,model in [(p,None) for p in POLICIES]+list(models.items()):
                start=time.process_time();counts=Counter();details=[]
                for f in fs:
                    if f["split"]!=split:continue
                    for a in f["arms"]:
                        cert,stats=run(f["masks"],f["n"],Oracle(f["masks"],a).query,budget,policy,int(a["id"],16),model)
                        ver=verify(f,a,cert)
                        counts["n"]+=1;counts["bad_n"]+=bool(a["w"])
                        counts["supported"]+=ver["supported"];counts["refuted"]+=ver["refuted"]
                        counts["minimum"]+=ver["minimum"];counts["incomplete"]+=not ver["minimum"]
                        counts["false_accusation"]+=ver["refuted"] and not a["w"]
                        counts["lower_sum"]+=ver["lower"];counts["calls"]+=stats["calls"]
                        counts["pools"]+=stats["pools"];counts["solver_states"]+=stats["states"]
                        assert stats["calls"]<=budget and ver["lower"]<=a["min"]
                        if ver["minimum"]:
                            assert len(cert["repair"])==a["min"]
                            accepted.setdefault(a["id"],set()).add(tuple(cert["repair"]))
                        details.append({"id":a["id"],"certificate":cert,"verified":ver,**stats})
                s={"split":split,"budget":budget,"policy":policy,**counts,"cpu_seconds":time.process_time()-start}
                scores.append(s)
                (out/f"{split}-{budget}-{policy}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
                if budget in (4,8,16,32):print("SCORE",json.dumps(s),flush=True)
    totals=Counter()
    for s in scores:
        if s["split"].startswith("test") and s["budget"] in (4,8):totals[s["policy"]]+=s["minimum"]
    best=max(totals[p] for p in POLICIES)
    gate={"totals":dict(totals),"best_symbolic":best,
          "passed":all(totals[f"learned-{s}"]>=best+6 for s in SEEDS) and not any(x["false_accusation"] for x in scores)}
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    (out/"actor-repairs.json").write_text(json.dumps({k:[list(ids) for ids in sorted(v)] for k,v in accepted.items()},indent=2))
    print("GATE",json.dumps(gate),flush=True)
    return accepted

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean");p.add_argument("--skip-native",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fs=list(families());validate(fs)
    (a.out/"families.json").write_text(json.dumps(fs))
    if not a.skip_native:native(fs,a.out,a.lean)
    # Negative controls: a fabricated safe-row witness and an insufficient repair must fail.
    for f in fs:
        for arm in f["arms"]:
            cert={"observed":[0],"lower":1,"repair":None}
            try:verify(f,arm,cert)
            except AssertionError:pass
            else:raise AssertionError("accepted fabricated witness")
            if arm["w"]:
                try:verify(f,arm,{"observed":[],"lower":0,"repair":[]})
                except AssertionError:pass
                else:raise AssertionError("accepted insufficient repair")
    data=teacher(fs)
    print("TRAIN",len(data),sum(len(x[0]) for x in data),flush=True)
    models={};fits=[]
    for seed in SEEDS:
        model,info=train(data,seed,a.out);models[f"learned-{seed}"]=model;fits.append(info)
    (a.out/"training.json").write_text(json.dumps(fits,indent=2))
    accepted=evaluate(fs,models,a.out)
    if not a.skip_native:native(fs,a.out,a.lean,accepted)

if __name__=="__main__":main()
