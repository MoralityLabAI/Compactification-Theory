"""CG-06: train tied ground-expression critics and verify generated certificates."""
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import corpus as C

WIDTHS=(32,64)
SEEDS=(111,113)
BUDGETS=(1,2,4,8)
BASELINES=("prove_first","witness_first","random","center","exact")
TOKENS=("PAD","NAT","INT","PREM","GOAL","END","v","k","+","-","*","/","le","lt","not")
T={x:i for i,x in enumerate(TOKENS)}

def serialize(case,indices):
    vs=C.values(case,indices)
    result=[(T[case["typ"].upper()],0.,0.)]
    def visit(e):
        op=e[0]
        if op in ("v","k"):
            val=vs[e[1]] if op=="v" else e[1]
            result.append((T[op],val/32.,math.copysign(math.log1p(abs(val)),val)/4.))
        else:
            result.append((T[op],0.,0.))
            for child in e[1:]:visit(child)
    for p in case["premises"]:
        result.append((T["PREM"],0.,0.));visit(p)
    result.append((T["GOAL"],0.,0.));visit(case["goal"])
    result.append((T["END"],0.,0.))
    return result

def pack(seqs):
    length=torch.tensor([len(s) for s in seqs])
    m=int(length.max());ids=torch.zeros(len(seqs),m,dtype=torch.long)
    vals=torch.zeros(len(seqs),m,2)
    for i,s in enumerate(seqs):
        ids[i,:len(s)]=torch.tensor([x[0] for x in s])
        vals[i,:len(s)]=torch.tensor([x[1:] for x in s])
    return ids,vals,length

class Critic(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.embed=nn.Embedding(len(TOKENS),16,padding_idx=0)
        self.gru=nn.GRU(18,width,batch_first=True)
        self.head=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,ids,vals,length):
        z,_=self.gru(torch.cat((self.embed(ids),vals),-1))
        return self.head(z[torch.arange(len(ids)),length-1]).squeeze(-1)

def train(rows,width,seed,out):
    seq=[];label=[]
    for case in rows:
        if case["split"]!="train":continue
        for x in C.grid(case["widths"]):
            seq.append(serialize(case,x));label.append(float(C.witness(case,x)))
    x,v,l=pack(seq);y=torch.tensor(label)
    assert y.sum()>0 and y.sum()<len(y)
    weight=(len(y)-y.sum())/y.sum()
    torch.manual_seed(seed);rng=random.Random(seed)
    model=Critic(width)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    start=time.process_time();losses=[]
    for epoch in range(12):
        order=list(range(len(y)));rng.shuffle(order);total=0.
        for offset in range(0,len(y),128):
            ids=order[offset:offset+128]
            opt.zero_grad()
            z=model(x[ids],v[ids],l[ids])
            loss=F.binary_cross_entropy_with_logits(z,y[ids],pos_weight=weight)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(y))
        print("EPOCH",width,seed,epoch+1,losses[-1],flush=True)
    tag=f"gru-{width}-{seed}"
    path=out/(tag+".pt");torch.save(model.state_dict(),path)
    info={"model":tag,"parameters":sum(p.numel() for p in model.parameters()),
          "training_tuples":len(y),"positive_tuples":int(y.sum()),"pos_weight":float(weight),
          "loss":losses,"cpu_seconds":time.process_time()-start,
          "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("FIT",json.dumps(info),flush=True)
    return model.eval(),info

@torch.no_grad()
def rank(case,policy,model=None):
    tuples=C.grid(case["widths"])
    if model is not None:
        ids,vals,length=pack([serialize(case,x) for x in tuples])
        scores=model(ids,vals,length).tolist()
        actions=["PROVE",*tuples];logits=[-max(scores),*scores]
        order=sorted(range(len(actions)),key=lambda j:-logits[j])
        return [actions[j] for j in order],len(tuples)
    if policy=="prove_first":return ["PROVE",*tuples],0
    if policy=="witness_first":return [*tuples,"PROVE"],0
    if policy=="random":
        actions=["PROVE",*tuples];random.Random(int(case["id"],16)).shuffle(actions)
        return actions,0
    if policy=="center":
        ordered=sorted(tuples,key=lambda x:sum(abs(2*i-w+1) for i,w in zip(x,case["widths"])))
        return [*ordered,"PROVE"],0
    assert policy=="exact"
    valid=[x for x in tuples if C.witness(case,x)]
    action=valid[0] if valid else "PROVE"
    return [action],len(tuples)

def evaluate(rows,models,out):
    eval_rows=[x for x in rows if x["split"]!="train"]
    accepted={};scores=[];details=[]
    for split in ("cal","test"):
        cases=[x for x in eval_rows if x["split"]==split]
        for policy,model in [(p,None) for p in BASELINES]+list(models.items()):
            counts={k:Counter() for k in BUDGETS};started=time.process_time()
            for case in cases:
                tuples=C.grid(case["widths"])
                witnesses={x for x in tuples if C.witness(case,x)}
                support=not witnesses
                actions,work=rank(case,policy,model)
                good=lambda a:(support if a=="PROVE" else a in witnesses)
                first_bad=(support and actions[0]!="PROVE")
                for budget in BUDGETS:
                    tried=[]
                    selected=None
                    for action in actions[:budget]:
                        tried.append(action)
                        if good(action):
                            selected=action;break
                    passed=selected is not None
                    if passed:accepted.setdefault(case["id"],set()).add(selected)
                    row=counts[budget]
                    row["n"]+=1;row["complete"]+=passed
                    row["supported_n"]+=support;row["refuted_n"]+=not support
                    row["supported"]+=passed and support;row["refuted"]+=passed and not support
                    row["first_false_accusations"]+=first_bad
                    row["rejected"]+=len(tried)-int(passed)
                    row["false_accusation_attempts"]+=sum(support and x!="PROVE" for x in tried)
                    row["proposals"]+=len(tried);row["candidate_evaluations"]+=work
                    details.append({"id":case["id"],"split":split,"module":case["module"],
                                    "policy":policy,"budget":budget,"truth":support,
                                    "attempts":tried,"selected":selected,"verified_precheck":passed})
            cpu=time.process_time()-started
            for budget,row in counts.items():
                s={"split":split,"policy":policy,"budget":budget,**row,"all_budgets_cpu_seconds":cpu}
                scores.append(s);print("SCORE",json.dumps(s),flush=True)
    gate={}
    for width in WIDTHS:
        selected=[x for x in scores if x["split"]=="test" and x["budget"]==4 and x["policy"].startswith(f"gru-{width}-")]
        assert len(selected)==2
        gate[str(width)]=all(100*x["complete"]>=85*x["n"] and 10*x["first_false_accusations"]<=x["supported_n"] for x in selected)
    (out/"scores.json").write_text(json.dumps(scores,indent=2))
    (out/"cases.jsonl").write_text("".join(json.dumps(d)+"\n" for d in details))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE_PRECHECK",json.dumps(gate),flush=True)
    # Stable ordering is used only for independent native replay.
    return {k:sorted(v,key=str) for k,v in accepted.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=C.generate()
    native=C.Native(args.source,args.out,args.lean)
    native.links()
    refs={};prompts=[];gold=[]
    for row in rows:
        ws=[x for x in C.grid(row["widths"]) if C.witness(row,x)]
        action=ws[0] if ws else "PROVE";refs[row["id"]]=[action]
        raw=(args.source/"MathCan"/(row["module"]+".lean")).read_text()
        prompts.append({"id":row["id"],"source_sha256":C.sha(raw),"source":raw,
                        "selected":"MathCan."+row["module"]+"."+row["decl"],
                        "advertised":C.advertised(row),"arithmetic_input":{k:row[k] for k in ("typ","lo","widths","premises","goal")}})
        gold.append({"id":row["id"],"split":row["split"],"pair":row["pair"],"reference":action})
    for name,xs in [("inputs",prompts),("gold",gold)]:
        (args.out/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in xs))
    assert native.replay(rows,refs,"reference")==len(rows)
    models={};infos=[]
    for width in WIDTHS:
        for seed in SEEDS:
            model,info=train(rows,width,seed,args.out)
            models[info["model"]]=model;infos.append(info)
    (args.out/"training.json").write_text(json.dumps(infos,indent=2))
    accepted=evaluate(rows,models,args.out)
    n=native.replay([r for r in rows if r["split"]!="train"],accepted,"actor")
    assert n==sum(map(len,accepted.values()))
    print("NATIVE_GATE",n,"distinct accepted certificates replayed",flush=True)
if __name__=="__main__":main()
