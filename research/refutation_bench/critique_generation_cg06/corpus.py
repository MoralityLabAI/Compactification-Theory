"""CG-06 native finite-domain correspondence and witness corpus."""
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from pathlib import Path

PIN="6d02b611c7358216379cb36952512a104e0bdb8e"
ALLOWED={"propext","Classical.choice","Quot.sound"}
def V(i):return ("v",i)
def K(n):return ("k",n)
def B(op,a,b):return (op,a,b)
def add(a,b):return B("+",a,b)
def sub(a,b):return B("-",a,b)
def mul(a,b):return B("*",a,b)
def div(a,b):return B("/",a,b)
def le(a,b):return B("le",a,b)
def lt(a,b):return B("lt",a,b)
def neg(p):return ("not",p)
a,b,c,d=map(V,range(4))
ZERO,ONE,TWO,THREE=map(K,range(4))
def S(split,module,decl,typ,n,premises,goal,args=None):
    return dict(split=split,module=module,decl=decl,typ=typ,n=n,premises=premises,goal=goal,
                args=list(map(V,range(n))) if args is None else args)
summ=add(add(a,b),c)
ROOTS=[
 S("train","MatchedStepMargin","positive_margin_stable","Int",3,[lt(c,a),le(sub(a,b),c)],lt(ZERO,b)),
 S("train","MatchedStepMargin","negative_margin_stable","Int",3,[lt(a,sub(ZERO,c)),le(sub(b,a),c)],lt(b,ZERO)),
 S("train","BlockRemainder","square_mono","Nat",2,[le(a,b)],le(mul(a,a),mul(b,b))),
 S("train","TemporalCutCapacity","margin_three_exceeds_budget","Nat",3,[le(THREE,summ)],neg(le(summ,TWO)),[summ]),
 S("cal","RHWindowExponents","normalized_character_error_26","Nat",3,[le(b,c),le(a,add(mul(K(4),b),mul(K(22),c)))],le(a,mul(K(26),c))),
 S("cal","FiniteHorizonGap","epsilon_sixth_clearance","Nat",2,[le(mul(K(6),b),a)],le(add(a,mul(TWO,b)),mul(TWO,sub(a,mul(TWO,b))))),
 S("test","PrecisionEntropy","quaternary_below_threshold","Nat",2,[le(ONE,a),lt(b,sub(mul(TWO,a),ONE))],lt(div(add(b,ONE),TWO),a)),
 S("test","PrecisionEntropyRobust","separated_clusters_two_sided_noise","Nat",4,[le(add(a,d),b),lt(mul(TWO,c),d)],lt(add(a,c),sub(b,c))),
 S("test","NOR20BigintWidthBounds","streaming_buffer_saving_monotone_in_width","Nat",3,[le(b,c)],le(mul(mul(K(32),a),b),mul(mul(K(32),a),c))),
]
SEEDS={"train":106601,"cal":106602,"test":106603}
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def eval_expr(e,values,typ):
    op=e[0]
    if op=="v":return values[e[1]]
    if op=="k":return e[1]
    if op=="not":return not eval_expr(e[1],values,typ)
    x,y=(eval_expr(t,values,typ) for t in e[1:])
    if op=="le":return x<=y
    if op=="lt":return x<y
    if op=="+":return x+y
    if op=="-":return max(0,x-y) if typ=="Nat" else x-y
    if op=="*":return x*y
    if op=="/":return x//y if y else 0
    raise ValueError(op)
def lean_expr(e,values,typ):
    op=e[0]
    if op=="v":return values[e[1]]
    if op=="k":return f"({e[1]} : {typ})"
    if op=="not":return "(¬ "+lean_expr(e[1],values,typ)+")"
    x,y=(lean_expr(t,values,typ) for t in e[1:])
    return f"({x} {dict(le='≤',lt='<').get(op,op)} {y})"
def grid(widths):return list(itertools.product(*(range(w) for w in widths)))
def values(case,indices):return [lo+i for lo,i in zip(case["lo"],indices)]
def valid(case,indices):
    vs=values(case,indices)
    return all(eval_expr(p,vs,case["typ"]) for p in case["premises"])
def witness(case,indices):
    return valid(case,indices) and not eval_expr(case["goal"],values(case,indices),case["typ"])
def generate():
    rows=[]
    for root in ROOTS:
        rng=random.Random(SEEDS[root["split"]]+int(sha(root["decl"])[:8],16))
        count=32 if root["split"]=="cal" else 64
        quota={False:count//2,True:count//2};seen=set()
        for attempt in range(250000):
            if not any(quota.values()):break
            lo=[rng.randint(0,8) if root["typ"]=="Nat" else rng.randint(-5,5) for _ in range(root["n"])]
            widths=[rng.randint(1,3) for _ in lo]
            omitted=rng.randrange(len(root["premises"]))
            key=tuple(lo+widths)
            if key in seen:continue
            base={**root,"lo":lo,"widths":widths}
            if not any(valid(base,v) for v in grid(widths)):continue
            weakened={**base,"premises":[p for i,p in enumerate(root["premises"]) if i!=omitted]}
            ws=[v for v in grid(widths) if witness(weakened,v)]
            bad=bool(ws)
            if not quota[bad]:continue
            quota[bad]-=1;seen.add(key)
            pair=sha(root["decl"]+str(key))[:20]
            for arm,case in [("full",base),("weak",weakened)]:
                case={**case,"pair":pair,"id":sha(pair+arm)[:20],"arm":arm}
                assert arm!="full" or not any(witness(case,v) for v in grid(widths))
                rows.append(case)
        assert not any(quota.values()),(root["decl"],quota,len(seen))
        print("CORPUS_ROOT",root["split"],root["decl"],count,flush=True)
    assert len(rows)==1024
    assert len({c["id"] for c in rows})==1024
    assert all(sum(x["split"]==s for x in rows)==n for s,n in (("train",512),("cal",128),("test",384)))
    return rows

def coordinate_expr(case,indices=None):
    ty=case["typ"]
    out=[]
    for j,(lo,w) in enumerate(zip(case["lo"],case["widths"])):
        v=f"x{j}.val" if indices is None else f"(({indices[j]} : Fin {w}).val)"
        if ty=="Int":v=f"(Int.ofNat {v})"
        out.append(f"({v} + ({lo} : {ty}))")
    return out
def advertised(case):
    vs=coordinate_expr(case);ty=case["typ"]
    bind=" ".join(f"(x{i} : Fin {w})" for i,w in enumerate(case["widths"]))
    return "∀ "+bind+", "+" → ".join(lean_expr(e,vs,ty) for e in case["premises"]+[case["goal"]])
def cert(case,action):
    prop=advertised(case)
    if action=="PROVE":
        if case["arm"]=="full":
            vs=coordinate_expr(case)
            bind=" ".join(f"x{i}" for i in range(case["n"]))
            hs=" ".join(f"h{i}" for i in range(len(case["premises"])))
            args=" ".join(lean_expr(e,vs,case["typ"]) for e in case["args"])
            body=f"by intro {bind} {hs}; exact MathCan.{case['module']}.{case['decl']} {args} {hs}"
        else:
            body="by decide"
        return prop,body
    vs=coordinate_expr(case,action)
    goal=lean_expr(case["goal"],vs,case["typ"])
    args=" ".join(f"({i} : Fin {w})" for i,w in zip(action,case["widths"]))
    hs=" ".join("(by decide)" for _ in case["premises"])
    return f"¬ ({prop})",f"by intro h; exact (by decide : ¬ {goal}) (h {args} {hs})"
def link(root):
    ty=root["typ"];vs=[f"v{i}" for i in range(root["n"])]
    bind=" ".join(vs);hs=[f"h{i}" for i in range(len(root["premises"]))]
    prop=f"∀ ({bind} : {ty}), "+" → ".join(lean_expr(e,vs,ty) for e in root["premises"]+[root["goal"]])
    args=" ".join(lean_expr(e,vs,ty) for e in root["args"])
    return prop,f"by intro {bind} {' '.join(hs)}; exact MathCan.{root['module']}.{root['decl']} {args} {' '.join(hs)}"

def run_lean(lean,path,cwd,env,log,opts=()):
    proc=subprocess.run([lean,*opts,str(path)],cwd=cwd,env=env,capture_output=True,text=True,timeout=240)
    s=proc.stdout+proc.stderr;log.write_text(s)
    assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",s),s[-10000:]
    return s
def check_axioms(s,expected):
    found={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",s):
        found[m[1]]={x.strip() for x in (m[3] or "").split(",") if x.strip()}
    assert set(found)==set(expected),(set(found)-set(expected),set(expected)-set(found))
    assert all(not (xs-ALLOWED) for xs in found.values()),found
    return {k:sorted(v) for k,v in found.items()}

class Native:
    def __init__(self,source,out,lean):
        self.source=source.resolve();self.out=out.resolve();self.lean=lean
        assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=self.source,text=True).strip()==PIN
        self.cache=self.out/"olean";self.cache.mkdir(parents=True,exist_ok=True)
        self.env=dict(os.environ,LEAN_PATH=str(self.cache));self.done=set()
    def module(self,mod):
        if mod in self.done:return
        rel=Path(*mod.split(".")).with_suffix(".lean")
        raw=(self.source/rel).read_text()
        for dep in re.findall(r"^import (MathCan\.[A-Za-z0-9_]+)\s*$",raw,re.M):
            self.module(dep)
        target=self.cache/rel.with_suffix(".olean");target.parent.mkdir(parents=True,exist_ok=True)
        run_lean(self.lean,rel,self.source,self.env,self.out/(mod+"-source.log"),("-o",str(target)))
        self.done.add(mod)
    def links(self):
        for i,r in enumerate(ROOTS):
            mod="MathCan."+r["module"];self.module(mod)
            prop,body=link(r);name=f"CG06Links.link{i}"
            p=self.out/f"link{i}.lean"
            p.write_text(f"import {mod}\nnamespace CG06Links\n theorem link{i} : {prop} := {body}\n#print axioms {name}\n#print axioms {mod}.{r['decl']}\nend CG06Links\n")
            s=run_lean(self.lean,p,self.out,self.env,p.with_suffix(".log"))
            check_axioms(s,[name,mod+"."+r["decl"]])
        print("NATIVE_LINKS",len(ROOTS),"source modules including dependencies",len(self.done),flush=True)
    def replay(self,rows,actions,tag):
        total=0;receipts=[]
        for mod in sorted({x["module"] for x in rows}):
            group=[(c,action) for c in rows if c["module"]==mod for action in actions.get(c["id"],[])]
            for off in range(0,len(group),24):
                lines=[f"import MathCan.{mod}","set_option maxRecDepth 4096","set_option maxHeartbeats 4000000","namespace CG06Audit"]
                expected=[]
                for j,(case,action) in enumerate(group[off:off+24]):
                    prop,body=cert(case,action);name=f"cert{j}"
                    lines.append(f"theorem {name} : {prop} := {body}")
                    lines.append(f"#print axioms {name}");expected.append("CG06Audit."+name)
                    receipts.append({"id":case["id"],"action":action,"certificate_sha256":sha(prop+" := "+body)})
                lines.append("end CG06Audit")
                path=self.out/f"{tag}-{mod}-{off}.lean";path.write_text("\n".join(lines)+"\n")
                s=run_lean(self.lean,path,self.out,self.env,path.with_suffix(".log"));check_axioms(s,expected)
                total+=len(expected)
        (self.out/f"{tag}-receipts.json").write_text(json.dumps(receipts,indent=2))
        print("NATIVE_REPLAY",tag,total,flush=True)
        return total
