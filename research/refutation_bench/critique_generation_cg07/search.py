"""CG-07 metered interval search over native theorem advertisements."""
import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"critique_generation_cg06"))
import corpus as C

BUDGETS=(7,15,31,63,127)
SEEDS=(121,127)
POLICIES=("fifo","dfs","slack","volume")
FEATURES=14
NEW_MODULES={"NOR40SingletonBracket","NOR37RandomizedCurvature"}

def key(c):
    return (c["decl"],tuple(c["lo"]),tuple(c["widths"]))

def generate():
    old=C.generate()
    forbidden={key(c) for c in old}
    a,b,c,d=map(C.V,range(4))
    zero=C.K(0)
    new=[
        C.S("test","NOR40SingletonBracket","nonnegative_policy_weight_preserves_singleton_coverage","Nat",3,
            [C.le(b,c)],C.le(C.mul(a,b),C.mul(a,c))),
        C.S("test","NOR37RandomizedCurvature","strict_detection_when_exploration_can_hit","Nat",4,
            [C.lt(zero,a),C.lt(zero,c)],
            C.lt(zero,C.add(C.mul(a,c),C.mul(C.sub(b,a),d))))
    ]
    C.ROOTS=list(C.ROOTS)+new
    C.SEEDS={"train":107701,"cal":107702,"test":107703}
    rows=[]
    for root in C.ROOTS:
        rng=random.Random(C.SEEDS[root["split"]]+int(C.sha(root["decl"])[:8],16))
        count={"train":64,"cal":32,"test":48}[root["split"]]
        quota={False:count//2,True:count//2};seen=set()
        for _ in range(250000):
            if not any(quota.values()):break
            lo=[rng.randint(0,24 if root["decl"]=="epsilon_sixth_clearance" and j==0 else 8) if root["typ"]=="Nat" else rng.randint(-5,5) for j in range(root["n"])]
            widths=[rng.randint(2,4) for _ in lo]
            base={**root,"lo":lo,"widths":widths}
            k=key(base)
            omitted=rng.randrange(len(root["premises"]))
            if k in seen or k in forbidden:continue
            tuples=C.grid(widths)
            if not any(C.valid(base,v) for v in tuples):continue
            weak={**base,"premises":[p for i,p in enumerate(root["premises"]) if i!=omitted]}
            bad=any(C.witness(weak,v) for v in tuples)
            if not quota[bad]:continue
            quota[bad]-=1;seen.add(k)
            pair=C.sha("CG07"+str(k))[:20]
            for arm,case in (("full",base),("weak",weak)):
                rows.append({**case,"pair":pair,"id":C.sha(pair+arm)[:20],"arm":arm})
        assert not any(quota.values()),(root["decl"],quota)
        print("CORPUS_ROOT",root["split"],root["decl"],count,flush=True)
    assert len(rows)==1120 and len({r["id"] for r in rows})==1120
    assert not {key(r) for r in rows}&forbidden
    for split,n in (("train",512),("cal",128),("test",480)):
        assert sum(r["split"]==split for r in rows)==n
    return rows

def numeric(e,box,case):
    op=e[0]
    if op=="v":
        l,h=box[e[1]];base=case["lo"][e[1]]
        return base+l,base+h
    if op=="k":return e[1],e[1]
    a,b=numeric(e[1],box,case);c,d=numeric(e[2],box,case)
    if op=="+":return a+c,b+d
    if op=="-":
        x,y=a-d,b-c
        if case["typ"]=="Nat":x,y=max(0,x),max(0,y)
        return x,y
    if op=="*":
        xs=(a*c,a*d,b*c,b*d)
        return min(xs),max(xs)
    if op=="/":
        assert c==d and c>0 and case["typ"]=="Nat"
        return a//c,b//c
    raise ValueError(op)

def margin(e,box,case):
    """An integer interval enclosing signed truth margin; true iff margin >= 0."""
    op=e[0]
    if op=="not":
        l,h=margin(e[1],box,case)
        return -h-1,-l-1
    assert op in ("le","lt")
    a,b=numeric(e[1],box,case);c,d=numeric(e[2],box,case)
    shift=int(op=="lt")
    return c-b-shift,d-a-shift

def signed(x):
    return math.copysign(math.log1p(abs(x)),x)/8.

def inspect(case,box,depth):
    ranges=[margin(p,box,case) for p in case["premises"]]
    # The conjunction required of a counterexample includes the negated goal.
    gl,gh=margin(case["goal"],box,case)
    ranges.append((-gh-1,-gl-1))
    if any(h<0 for _,h in ranges):return "prune",None
    if all(l>=0 for l,_ in ranges):return "witness",None
    widths=[h-l+1 for l,h in box];vol=math.prod(widths)
    assert vol>1
    ls=[l for l,h in ranges];hs=[h for l,h in ranges]
    unknown=sum(l<0<=h for l,h in ranges)
    f=[unknown/len(ranges),len(ranges)/8,signed(min(ls)),signed(min(hs)),
       sum(signed(x) for x in ls)/len(ls),sum(signed(x) for x in hs)/len(hs),
       math.log1p(vol)/8,max(widths)/4,sum(widths)/(4*len(widths)),
       sum(w==1 for w in widths)/len(widths),depth/16,len(widths)/4,
       float(case["typ"]=="Nat"),signed(ranges[-1][1])]
    assert len(f)==FEATURES
    return "open",f

def rootbox(case):
    return tuple((0,w-1) for w in case["widths"])

def children(box):
    j=max(range(len(box)),key=lambda i:box[i][1]-box[i][0])
    l,h=box[j];assert h>l
    m=(l+h)//2
    left=list(box);right=list(box)
    left[j]=(l,m);right[j]=(m+1,h)
    return tuple(left),tuple(right)

def contains(box,coords):
    return all(l<=x<=h for x,(l,h) in zip(coords,box))

def training_rows(rows):
    xs=[];ys=[]
    for case in rows:
        if case["split"]!="train":continue
        ws=[v for v in C.grid(case["widths"]) if C.witness(case,v)]
        def walk(box,depth):
            here=[v for v in ws if contains(box,v)]
            status,f=inspect(case,box,depth)
            if status=="prune":
                assert not here;return
            if status=="witness":
                assert C.witness(case,tuple(l for l,_ in box));return
            xs.append(f);ys.append(float(bool(here)))
            for b in children(box):walk(b,depth+1)
        walk(rootbox(case),0)
    assert 0<sum(ys)<len(ys)
    return torch.tensor(xs),torch.tensor(ys)

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURES,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def fit(x,y,seed,out):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Ranker();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    weight=(len(y)-y.sum())/y.sum()
    history=[]
    for _ in range(12):
        order=list(range(len(y)));rng.shuffle(order);total=0.
        for off in range(0,len(y),128):
            ids=order[off:off+128]
            opt.zero_grad()
            loss=F.binary_cross_entropy_with_logits(model(x[ids]),y[ids],pos_weight=weight)
            loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        history.append(total/len(y))
    path=out/f"ranker-{seed}.pt";torch.save(model.state_dict(),path)
    info={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
          "train_nodes":len(y),"positive_nodes":int(y.sum()),"loss":history,
          "sha256":C.sha(path.read_bytes())}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

@torch.no_grad()
def search(case,budget,policy,model=None):
    """Each box is charged exactly once, before it enters the frontier."""
    frontier=[];work=0;rank_work=0;pruned=0;trace=[]
    def read(box,depth):
        nonlocal work,pruned
        work+=1
        status,f=inspect(case,box,depth)
        trace.append({"box":box,"depth":depth,"status":status})
        if status=="prune":pruned+=1
        if status=="witness":return tuple(l for l,_ in box)
        if status=="open":frontier.append((box,depth,f))
        return None
    w=read(rootbox(case),0)
    while w is None and frontier and work+2<=budget:
        if model is not None:
            logits=model(torch.tensor([z[2] for z in frontier])).tolist()
            rank_work+=len(frontier)
            idx=max(range(len(frontier)),key=lambda j:logits[j])
        elif policy=="fifo":idx=0
        elif policy=="dfs":idx=len(frontier)-1
        elif policy=="slack":
            rank_work+=len(frontier)
            idx=max(range(len(frontier)),key=lambda j:(frontier[j][2][3],-frontier[j][2][6]))
        elif policy=="volume":
            rank_work+=len(frontier)
            idx=min(range(len(frontier)),key=lambda j:frontier[j][2][6])
        else:raise ValueError(policy)
        box,depth,_=frontier.pop(idx)
        # Always charge and inspect both children; keep the first witness.
        a,b=children(box)
        w1=read(a,depth+1);w2=read(b,depth+1)
        w=w1 if w1 is not None else w2
    action=w if w is not None else "PROVE" if not frontier else None
    assert work<=budget
    return action,{"nodes":work,"rank_evaluations":rank_work,"pruned":pruned,"trace":trace}

def exact_check(case,action,ws):
    if action is None:return False
    if action=="PROVE":return not ws
    return action in ws

def stratum(case):
    if case["split"]!="test":return case["split"]
    return "new_root" if case["module"] in NEW_MODULES else "fresh_box"

def evaluate(rows,models,out):
    scores=[];details=[];accepted={}
    ws={r["id"]:{x for x in C.grid(r["widths"]) if C.witness(r,x)} for r in rows if r["split"]!="train"}
    # The exact truth tables above are used only by the independent scorer.
    for split in ("cal","test"):
        subset=[r for r in rows if r["split"]==split]
        for budget in BUDGETS:
            for name,model in [(p,None) for p in POLICIES]+list(models.items()):
                counts=Counter();strata={};start=time.process_time()
                for r in subset:
                    action,stats=search(r,budget,name,model)
                    witnesses=ws[r["id"]];bad=bool(witnesses)
                    valid=exact_check(r,action,witnesses)
                    assert action is None or valid,(r["id"],action)
                    if valid:accepted.setdefault(r["id"],set()).add(action)
                    st=strata.setdefault(stratum(r),Counter())
                    for dest in (counts,st):
                        dest["n"]+=1;dest["bad_n"]+=bad;dest["supported_n"]+=not bad
                        dest["complete"]+=valid;dest["refuted"]+=valid and bad
                        dest["supported"]+=valid and not bad;dest["abstain"]+=action is None
                        dest["invalid"]+=action is not None and not valid
                        for k in ("nodes","rank_evaluations","pruned"):dest[k]+=stats[k]
                    details.append({"id":r["id"],"split":split,"stratum":stratum(r),"budget":budget,
                                    "policy":name,"action":action,"verified":valid,"bad":bad,**stats})
                row={"split":split,"budget":budget,"policy":name,**counts,
                     "strata":{k:dict(v) for k,v in strata.items()},"cpu_seconds":time.process_time()-start}
                scores.append(row)
                print("SCORE",json.dumps(row),flush=True)
    # Independent complete-search diagnostic; no diagnostic results used in ranking.
    for r in rows:
        if r["split"]=="train":continue
        action,stats=search(r,4096,"fifo")
        assert exact_check(r,action,ws[r["id"]])
    ref=Counter();support=Counter()
    for s in scores:
        if s["split"]=="test" and s["budget"] in (15,31):
            ref[s["policy"]]+=s["refuted"];support[s["policy"]]+=s["supported"]
    best_ref=max(ref[p] for p in POLICIES);best_support=max(support[p] for p in POLICIES)
    gate={"defect_totals":dict(ref),"support_totals":dict(support),
          "best_symbolic_defects":best_ref,"best_symbolic_support":best_support,
          "passed":all(ref[f"learned-{s}"]>=best_ref+4 and support[f"learned-{s}"]>=best_support-4 for s in SEEDS)
                    and not any(s["invalid"] for s in scores)}
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE_PRECHECK",json.dumps(gate),flush=True)
    return {k:sorted(v,key=str) for k,v in accepted.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=generate();native=C.Native(args.source,args.out,args.lean);native.links()
    refs={};inputs=[];gold=[]
    for r in rows:
        ws=[x for x in C.grid(r["widths"]) if C.witness(r,x)]
        ref=ws[0] if ws else "PROVE";refs[r["id"]]=[ref]
        inputs.append({"id":r["id"],"advertised":C.advertised(r),
                       "selected":"MathCan."+r["module"]+"."+r["decl"],
                       "arithmetic_input":{k:r[k] for k in ("lo","widths","typ","premises","goal")}})
        gold.append({"id":r["id"],"split":r["split"],"pair":r["pair"],"reference":ref})
    for name,x in (("inputs",inputs),("gold",gold)):
        (args.out/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in x))
    assert native.replay(rows,refs,"reference")==len(rows)
    x,y=training_rows(rows);models={};fits=[]
    for seed in SEEDS:
        model,info=fit(x,y,seed,args.out);models[f"learned-{seed}"]=model;fits.append(info)
    (args.out/"training.json").write_text(json.dumps(fits,indent=2))
    accepted=evaluate(rows,models,args.out)
    count=native.replay([r for r in rows if r["split"]!="train"],accepted,"actor")
    assert count==sum(map(len,accepted.values()))
    print("NATIVE_GATE",count,"distinct accepted certificates replayed",flush=True)
if __name__=="__main__":main()
