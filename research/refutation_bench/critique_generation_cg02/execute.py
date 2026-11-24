"""RB-CG-02: local, tied certificate execution on the CG-01 raw source grammar."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "critique_generation_cg01"))
import corpus as src

SEEDS=(61,67)
MODES=("learned_both","learned_chain","learned_boolean")
OPS=("not","and","or")
STOP=("NEXT","CLEAN","AX")
MAX=64

def fresh():
    old=list(src.families())
    saved=src.SPLITS
    src.SPLITS={k:(128 if k=="train" else n,101801+i,m,d)
                for i,(k,(n,s,m,d)) in enumerate(saved.items())}
    try:
        new=list(src.families())
    finally:
        src.SPLITS=saved
    assert len(new)==256
    assert len({src.sha(f["source"]) for f in new})==256
    assert not {src.sha(f["source"]) for f in old}&{src.sha(f["source"]) for f in new}
    assert len({a["id"] for f in new for a in f["arms"]})==2048
    for f in new:
        p=src.Parsed(f["source"])
        for a in f["arms"]:
            assert p.symbolic(a["root"],a["target"])==a["gold"]
            assert p.verify(a["root"],a["target"],a["gold"])["complete"]
        assert not re.search(r"\b(sorry|admit|native_decide|sorryAx)\b",f["native"])
    return new

def bits(name):
    n=int(name[1:],16)
    return [float(2*((n>>i)&1)-1) for i in range(32)]

class Syntax:
    """Syntax-only frontend; proof traversal is performed by the actor."""
    def __init__(self,raw):
        self.defs={};self.proofs={};self.bridges={}
        for i,line in enumerate(raw.splitlines(),1):
            m=re.fullmatch(r"def ("+src.NAME+r") \(x y z : Bool\) : Bool := (.+)",line)
            if m:
                self.defs[m[1]]=src.parse_expr(m[2]);continue
            m=re.fullmatch(r"theorem ("+src.NAME+r") : \("+src.TYPE+r"\) → \("+src.TYPE+r"\) := (.+)",line)
            if m:
                self.bridges[(m[2],m[3])]=i;continue
            m=re.fullmatch(r"(axiom|theorem) ("+src.NAME+r") : "+src.TYPE+r"(?: := (.+))?",line)
            assert m,line
            keyword,name,typeid,body=m.groups()
            self.proofs[name]={"line":i,"type":typeid,"keyword":keyword,"body":body or "",
                               "rhs":body if body and re.fullmatch(src.NAME,body) else None}
    def features(self,name):
        p=self.proofs[name];body=p["body"]
        return [float(p["keyword"]=="axiom"),float(bool(body)),
                float(body.startswith("by ")),float(p["rhs"] is not None),
                min(len(body),128)/128]
    def candidates(self,name):
        line=self.proofs[name]["line"]
        return [n for n,p in self.proofs.items() if p["line"]<line]
    def label(self,name):
        p=self.proofs[name]
        return 0 if p["rhs"] is not None else 2 if p["keyword"]=="axiom" else 1

def gate_features(op,a,b):
    return [float(op==k) for k in OPS]+[float(a),float(b)]

def exact_gate(op,a,b):
    if op=="not":return not a
    if op=="and":return a and b
    assert op=="or"
    return a or b

class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.term=nn.Sequential(nn.Linear(5,16),nn.Tanh(),nn.Linear(16,3))
        self.bind=nn.Linear(32,16,bias=False)
        self.boolean=nn.Sequential(nn.Linear(5,16),nn.Tanh(),nn.Linear(16,2))
    def pointer(self,q,c):
        q=F.normalize(self.bind(q),dim=-1)
        c=F.normalize(self.bind(c),dim=-1)
        return 12*(c*q.unsqueeze(-2)).sum(-1)

def train_rows(fs):
    rows=[]
    for f in fs:
        if f["split"]!="train":continue
        x=Syntax(f["source"])
        for name,p in x.proofs.items():
            candidates=x.candidates(name)
            target=x.label(name)
            if target==0:
                assert p["rhs"] in candidates
            rows.append((x.features(name),target,bits(p["rhs"]) if p["rhs"] else [0.]*32,
                         [bits(n) for n in candidates],candidates.index(p["rhs"]) if target==0 else 0))
    assert len(rows)>=2048
    rng=random.Random(8833);rng.shuffle(rows)
    return rows[:2048]

def fit(rows,seed,out):
    torch.manual_seed(seed)
    cell=Cell()
    opt=torch.optim.AdamW(cell.parameters(),lr=.01)
    x=torch.tensor([r[0] for r in rows]);y=torch.tensor([r[1] for r in rows])
    alias=[r for r in rows if r[1]==0]
    q=torch.tensor([r[2] for r in alias]);C=max(len(r[3]) for r in alias)
    candidates=torch.zeros(len(alias),C,32);mask=torch.zeros(len(alias),C,dtype=torch.bool)
    for i,r in enumerate(alias):
        candidates[i,:len(r[3])]=torch.tensor(r[3]);mask[i,:len(r[3])]=True
    ptr=torch.tensor([r[4] for r in alias])
    gates=[(gate_features(op,a,b),int(exact_gate(op,a,b))) for op in OPS for a in (False,True) for b in (False,True)]
    gx=torch.tensor([r[0] for r in gates]);gy=torch.tensor([r[1] for r in gates])
    losses=[]
    for _ in range(120):
        t=cell.term(x)
        p=cell.pointer(q,candidates).masked_fill(~mask,-1e9)
        g=cell.boolean(gx)
        loss=F.cross_entropy(t,y)+F.cross_entropy(p,ptr)+F.cross_entropy(g,gy)
        opt.zero_grad();loss.backward();opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        acc={"terminal":float((cell.term(x).argmax(-1)==y).float().mean()),
             "pointer":float((cell.pointer(q,candidates).masked_fill(~mask,-1e9).argmax(-1)==ptr).float().mean()),
             "gate":float((cell.boolean(gx).argmax(-1)==gy).float().mean())}
    path=out/f"cell-{seed}.pt";torch.save(cell.state_dict(),path)
    info={"seed":seed,"parameters":sum(p.numel() for p in cell.parameters()),
          "train_rows":len(rows),"alias_rows":len(alias),"loss":losses,"primitive_accuracy":acc,
          "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("FIT",json.dumps({k:v for k,v in info.items() if k!="loss"}),flush=True)
    return cell.eval(),info

@torch.no_grad()
def chain(x,root,cell,mode):
    n=root;path=[]
    for _ in range(MAX):
        p=x.proofs[n];path.append("L"+str(p["line"]))
        if mode=="learned":
            action=STOP[int(cell.term(torch.tensor(x.features(n))).argmax())]
        else:
            action=STOP[x.label(n)]
        if action!="NEXT":
            return [action,*path]
        if p["rhs"] is None:return ["FAIL",*path]
        if mode=="exact":
            n=p["rhs"];continue
        cs=x.candidates(n)
        if not cs:return ["FAIL",*path]
        q=torch.tensor(bits(p["rhs"]));c=torch.tensor([bits(z) for z in cs])
        scores=cell.pointer(q,c) if mode=="learned" else c@q
        n=cs[int(scores.argmax())]
    return ["FAIL",*path]

@torch.no_grad()
def semantics(x,root,target,cell,learned):
    def calc(expr,v):
        if isinstance(expr,str):return v[src.VARS.index(expr)]
        op=expr[0]
        if op=="call":return calc(x.defs[expr[1]],v)
        a=calc(expr[1],v);b=False if op=="not" else calc(expr[2],v)
        if learned:
            return bool(int(cell.boolean(torch.tensor(gate_features(op,a,b))).argmax()))
        return exact_gate(op,a,b)
    for v in src.VALUES:
        if not calc(x.defs[target],v):
            return ["COUNTER",*[str(int(z)) for z in v]]
    t=x.proofs[root]["type"]
    if target==t:return ["MATCH"]
    if (t,target) in x.bridges:return ["BRIDGE","L"+str(x.bridges[t,target])]
    return ["UNLINKED"]

def evaluate(fs,cell,policy,out):
    rows=[]
    for split in ("cal","test_depth","test_mechanism"):
        counts=Counter();details=[]
        for f in fs:
            if f["split"]!=split:continue
            x=Syntax(f["source"]);v=src.Parsed(f["source"])
            for a in f["arms"]:
                root,target=a["root"],a["target"]
                if policy=="symbolic":
                    pred=v.symbolic(root,target)
                else:
                    mode="learned" if policy.startswith("learned_both") or policy.startswith("learned_chain") else "dot" if policy=="dot" else "exact"
                    lg=policy.startswith("learned_both") or policy.startswith("learned_boolean")
                    pred=chain(x,root,cell,mode)+["SEP"]+semantics(x,root,target,cell,lg)+["EOS"]
                res=v.verify(root,target,pred)
                expected=v.symbolic(root,target)
                sem=expected[expected.index("SEP")+1]
                supported=expected[0]=="CLEAN" and sem in ("MATCH","BRIDGE")
                counts["n"]+=1
                for k in ("complete","axiom","semantic","support","unlinked"):
                    counts[k]+=res[k]
                counts["malformed"]+=not res["syntax"]
                counts["supported_n"]+=supported
                counts["unlinked_n"]+=sem=="UNLINKED"
                counts["counter_n"]+=sem=="COUNTER"
                counts["verified_counter"]+=res["semantic"] and sem=="COUNTER"
                counts["false_accusation"]+=supported and ("AX" in pred or "COUNTER" in pred)
                counts["tokens"]+=len(pred)
                counts["path_steps"]+=pred.index("SEP")-1
                details.append({"id":a["id"],"generated":pred,**res})
        row={"split":split,"policy":policy,**counts}
        rows.append(row)
        (out/f"{split}-{policy}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
        print("SCORE",json.dumps(row),flush=True)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fs=fresh();src.native(fs,args.out,args.lean)
    src.export(fs,args.out)
    (args.out/"source-hashes.json").write_text(json.dumps({f["id"]:src.sha(f["source"]) for f in fs},indent=2))
    rows=train_rows(fs);scores=[];fits=[]
    for policy in ("symbolic","dot"):
        result=evaluate(fs,None,policy,args.out);scores+=result
        assert all(r["complete"]==r["n"] for r in result)
    for seed in SEEDS:
        cell,info=fit(rows,seed,args.out);fits.append(info)
        for mode in MODES:
            scores+=evaluate(fs,cell,f"{mode}-{seed}",args.out)
    gate={str(seed):all(20*r["complete"]>=19*r["n"] and 20*r["false_accusation"]<=r["supported_n"]
                       for r in scores if r["policy"]==f"learned_both-{seed}" and r["split"].startswith("test")) for seed in SEEDS}
    (args.out/"scores.json").write_text(json.dumps(scores,indent=2))
    (args.out/"training.json").write_text(json.dumps(fits,indent=2))
    (args.out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)

if __name__=="__main__":main()
