"""CG-10: restricted native proof provenance and allowed Horn entailment."""
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path

SPLITS={"train":(64,110001,(2,4),(3,6)),"cal":(16,110002,(2,4),(3,6)),
        "test_depth":(24,110003,(7,10),(3,6)),"test_clutter":(24,110004,(2,4),(12,20))}
BUDGETS=(8,16,32,64,128)
ALLOWED={"propext","Classical.choice","Quot.sound"}
KINDS=("CLEAN","TAINTED_REPAIRABLE","TAINTED_REQUIRED")

def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def public(c):return {k:c[k] for k in ("n","goal","env","nodes","root")}
def atom(a):return f"p{a}"

def generate(split,i,rng,depths,clutter):
    n=0
    def fresh():
        nonlocal n
        z=n;n+=1;return z
    s0,s1,g=fresh(),fresh(),fresh()
    env=[];nodes=[]
    def add_env(p,h,allowed):
        j=len(env)
        env.append({"prem":list(p),"head":h,"allowed":allowed})
        k=len(nodes);nodes.append({"env":j,"deps":[],"head":h})
        return j,k
    def chain(s,tainted):
        j,previous=add_env([],s,True)
        start=j
        depth=rng.randint(*depths)
        gap=rng.randrange(depth) if tainted else -1
        p=s
        for k in range(depth):
            h=g if k==depth-1 else fresh()
            _,rule=add_env([p],h,k!=gap)
            node=len(nodes)
            nodes.append({"env":None,"deps":[previous,rule],"head":h})
            previous=node;p=h
        return start,previous
    clean_start,clean=chain(s0,False)
    _,taint=chain(s1,True)
    # Decoys can consume any prior atom, but cannot feed either goal chain.
    decoy_env=[]
    for j in range(rng.randint(*clutter)):
        available=list(range(n))
        h=fresh()
        ps=[] if rng.random()<.3 else rng.sample(available,rng.randint(1,min(2,len(available))))
        idx,_=add_env(ps,h,j!=0 and rng.random()<.65)
        decoy_env.append(idx)
    # Keep source proof order topological while randomizing independent nodes.
    atom_order=list(range(n));rng.shuffle(atom_order)
    remap={old:new for new,old in enumerate(atom_order)}
    env_order=list(range(len(env)));rng.shuffle(env_order)
    emap={old:new for new,old in enumerate(env_order)}
    order=[];remaining=set(range(len(nodes)))
    while remaining:
        ready=sorted(j for j in remaining if set(nodes[j]["deps"])<=set(order))
        j=rng.choice(ready);order.append(j);remaining.remove(j)
    nmap={old:new for new,old in enumerate(order)}
    env2=[{"prem":[remap[p] for p in env[j]["prem"]],"head":remap[env[j]["head"]],
           "allowed":env[j]["allowed"]} for j in env_order]
    nodes2=[{"env":None if nodes[j]["env"] is None else emap[nodes[j]["env"]],
             "deps":[nmap[d] for d in nodes[j]["deps"]],"head":remap[nodes[j]["head"]]} for j in order]
    family=f"CG10-{SPLITS[split][1]}-{i}"
    rows=[]
    for arm,chosen,expected in [
        ("clean",clean,"CLEAN"),("repair",taint,"TAINTED_REPAIRABLE"),
        ("required",taint,"TAINTED_REQUIRED"),("decoy",clean,"CLEAN")]:
        es=[dict(x) for x in env2]
        if arm=="required":es[emap[clean_start]]["allowed"]=False
        if arm=="decoy":
            j=emap[decoy_env[0]]
            es[j]["allowed"]=not es[j]["allowed"]
            # Keep a separate unused forbidden declaration in all arms.
            j=emap[decoy_env[1]]
            es[j]["allowed"]=False
        ns=nodes2+[{"env":None,"deps":[nmap[chosen]],"head":remap[g]}]
        cid=sha(family+arm)[:20]
        case={"id":cid,"family":family,"split":split,"n":n,"goal":remap[g],
              "env":es,"nodes":ns,"root":len(ns)-1,"expected":expected}
        ref=reference(public(case))
        assert classify(ref)==expected and verify(public(case),ref)
        rows.append(case)
    rng.shuffle(rows);return rows

def cases():
    rows=[]
    for split,(count,seed,depth,clutter) in SPLITS.items():
        rng=random.Random(seed)
        for i in range(count):rows+=generate(split,i,rng,depth,clutter)
    assert len(rows)==512 and len({c["id"] for c in rows})==512
    assert len({sha(json.dumps(public(c),sort_keys=True)) for c in rows})==512
    return rows

def initial(c):
    return {e["head"] for e in c["env"] if e["allowed"] and not e["prem"]}
def legal(c,known):
    return [i for i,e in enumerate(c["env"]) if e["allowed"] and e["head"] not in known and set(e["prem"])<=known]
def closure(c,start=None):
    known=initial(c) if start is None else set(start)
    trace=[]
    while True:
        a=legal(c,known)
        if not a:return known,trace
        j=a[0];known.add(c["env"][j]["head"]);trace.append(j)
def proof_slice(c,start,trace,target):
    known=set(start);producer={}
    for j in trace:
        e=c["env"][j]
        assert e["allowed"] and e["head"] not in known and set(e["prem"])<=known
        known.add(e["head"]);producer[e["head"]]=j
    used=set()
    def visit(a):
        assert a in known
        if a not in producer:return
        j=producer[a]
        if j in used:return
        used.add(j)
        for p in c["env"][j]["prem"]:visit(p)
    visit(target)
    return [j for j in trace if j in used]
def reachable(c):
    out=set();stack=[c["root"]]
    while stack:
        j=stack.pop()
        if j in out:continue
        out.add(j);stack.extend(c["nodes"][j]["deps"])
    return out
def forbidden(c,j):
    x=c["nodes"][j]["env"]
    return x is not None and not c["env"][x]["allowed"]
def provenance(c):
    stack=[[c["root"]]];seen=set()
    while stack:
        path=stack.pop();j=path[-1]
        if j in seen:continue
        seen.add(j)
        if forbidden(c,j):return {"status":"TAINTED","path":path}
        stack.extend(path+[d] for d in reversed(c["nodes"][j]["deps"]))
    return {"status":"CLEAN","closure":sorted(seen)}
def reference(c):
    known,trace=closure(c)
    a=({"status":"DERIVABLE","trace":proof_slice(c,initial(c),trace,c["goal"])}
       if c["goal"] in known else {"status":"NOT_DERIVABLE","model":sorted(known)})
    return {"provenance":provenance(c),"allowed":a}
def classify(cert):
    p,a=cert["provenance"]["status"],cert["allowed"]["status"]
    if p=="CLEAN":
        assert a=="DERIVABLE";return "CLEAN"
    return "TAINTED_REPAIRABLE" if a=="DERIVABLE" else "TAINTED_REQUIRED"
def verify_prov(c,p):
    if not isinstance(p,dict):return False
    if p.get("status")=="TAINTED":
        path=p.get("path")
        return (type(path) is list and bool(path) and all(type(j) is int and 0<=j<len(c["nodes"]) for j in path)
                and path[0]==c["root"] and all(b in c["nodes"][a]["deps"] for a,b in zip(path,path[1:]))
                and forbidden(c,path[-1]))
    if p.get("status")=="CLEAN":
        xs=p.get("closure")
        return (type(xs) is list and all(type(j) is int and 0<=j<len(c["nodes"]) for j in xs)
                and sorted(reachable(c))==xs and all(not forbidden(c,j) for j in xs))
    return False
def verify_allowed(c,a):
    if not isinstance(a,dict):return False
    if a.get("status")=="DERIVABLE":
        state=initial(c);t=a.get("trace")
        if type(t) is not list:return False
        for j in t:
            if type(j) is not int or not 0<=j<len(c["env"]):return False
            e=c["env"][j]
            if not e["allowed"] or e["head"] in state or not set(e["prem"])<=state:return False
            state.add(e["head"])
        return c["goal"] in state
    if a.get("status")=="NOT_DERIVABLE":
        xs=a.get("model")
        if type(xs) is not list or any(type(x) is not int or not 0<=x<c["n"] for x in xs):return False
        if sorted(set(xs))!=xs:return False
        state=set(xs)
        return (c["goal"] not in state and all(not e["allowed"] or not set(e["prem"])<=state
                or e["head"] in state for e in c["env"]))
    return False
def verify(c,cert):
    return (isinstance(cert,dict) and verify_prov(c,cert.get("provenance"))
            and verify_allowed(c,cert.get("allowed")))

def arrow(p,h):
    return " → ".join([atom(x) for x in p]+[atom(h)])
def raw_source(row):
    c=public(row);ns="CG10_"+row["id"]
    ls=["namespace "+ns]
    for a in range(c["n"]):ls.append(f"axiom {atom(a)} : Prop")
    for j,e in enumerate(c["env"]):
        ls.append(f"axiom e{j} : {arrow(e['prem'],e['head'])}")
    for j,x in enumerate(c["nodes"]):
        # The node aliases for rule axioms have implication types. Preserve
        # exact proof terms; the root and every non-leaf conclude an atom.
        if x["env"] is not None:
            e=c["env"][x["env"]]
            ls.append(f"theorem d{j} : {arrow(e['prem'],e['head'])} := e{x['env']}")
        else:
            ds=x["deps"]
            body=f"d{ds[0]}" if len(ds)==1 else f"d{ds[1]} d{ds[0]}"
            ls.append(f"theorem d{j} : {atom(x['head'])} := {body}")
    ls.extend([f"theorem selected : {atom(c['goal'])} := d{c['root']}","end "+ns,"#print axioms "+ns+".selected"])
    return "\n".join(ls)
def allowed_indices(c):
    return [j for j,e in enumerate(c["env"]) if e["allowed"]]
def allowed_prop(c):
    args=["("+arrow(c["env"][j]["prem"],c["env"][j]["head"])+")" for j in allowed_indices(c)]
    return "∀ ("+" ".join(atom(a) for a in range(c["n"]))+" : Prop), "+" → ".join(args+[atom(c["goal"])])
def render_allowed(c,a,name):
    assert verify_allowed(c,a)
    ids=allowed_indices(c)
    if a["status"]=="NOT_DERIVABLE":
        state=set(a["model"])
        args=["True" if j in state else "False" for j in range(c["n"])]
        for j in ids:
            e=c["env"][j]
            if not e["prem"]:
                assert e["head"] in state;args.append("True.intro");continue
            hs=[f"h{k}" for k in range(len(e["prem"]))]
            body="True.intro" if e["head"] in state else hs[next(k for k,p in enumerate(e["prem"]) if p not in state)]
            args.append("(fun "+" ".join(hs)+" => "+body+")")
        return f"theorem {name} : ¬ ({allowed_prop(c)}) := by\n  intro h\n  exact h "+" ".join(args)
    ls=[f"theorem {name} : {allowed_prop(c)} := by",
        "  intro "+" ".join([atom(a) for a in range(c["n"])]+[f"e{j}" for j in ids])]
    names={c["env"][j]["head"]:f"e{j}" for j in ids if not c["env"][j]["prem"]}
    for k,j in enumerate(a["trace"]):
        e=c["env"][j];h=e["head"]
        term=" ".join([f"e{j}"]+[names[p] for p in e["prem"]])
        ls.append(f"  have t{k} : {atom(h)} := {term}");names[h]=f"t{k}"
    ls.append("  exact "+names[c["goal"]])
    return "\n".join(ls)

def run_lean(lean,p):
    proc=subprocess.run([lean,str(p)],cwd=p.parent,capture_output=True,text=True,timeout=240)
    s=proc.stdout+proc.stderr;p.with_suffix(".log").write_text(s)
    assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",s),s[-12000:]
    return s
def axioms(log):
    out={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
        assert m[1] not in out
        out[m[1]]={a.strip() for a in (m[3] or "").split(",") if a.strip()}
    return out
class Native:
    def __init__(self,out,lean):
        self.out=out.resolve();self.lean=lean
    def source(self,rows):
        receipts=[]
        for off in range(0,len(rows),16):
            sub=rows[off:off+16]
            p=self.out/f"source-{off}.lean"
            p.write_text("import Init\nset_option linter.unusedVariables false\n"+"\n".join(raw_source(r) for r in sub))
            found=axioms(run_lean(self.lean,p))
            assert len(found)==len(sub)
            for row in sub:
                c=public(row);ns="CG10_"+row["id"]
                used={c["nodes"][j]["env"] for j in reachable(c) if c["nodes"][j]["env"] is not None}
                actual=found[ns+".selected"]
                env_names={ns+f".e{j}" for j in range(len(c["env"]))}
                vocab={ns+f".p{j}" for j in range(c["n"])}
                assert actual&env_names=={ns+f".e{j}" for j in used},(row["id"],actual,used)
                assert not actual-(env_names|vocab|ALLOWED)
                receipts.append({"id":row["id"],"source_sha256":sha(raw_source(row)),
                                 "selected_axioms":sorted(actual),"environment_axioms":sorted(used)})
        (self.out/"source-receipts.json").write_text(json.dumps(receipts,indent=2))
        print("NATIVE_SOURCE",len(receipts),flush=True)
    def replay(self,rows,accepted,tag):
        by_id={r["id"]:public(r) for r in rows};receipts=[]
        items=[(cid,cert) for cid,certs in sorted(accepted.items()) for cert in certs]
        for off in range(0,len(items),24):
            sub=items[off:off+24]
            ls=["import Init","set_option linter.unusedVariables false","namespace CG10Audit"]
            expected=set()
            for j,(cid,cert) in enumerate(sub):
                c=by_id[cid];assert verify(c,cert)
                name=f"cert{j}";body=render_allowed(c,cert["allowed"],name)
                ls.extend([body,"#print axioms "+name]);expected.add("CG10Audit."+name)
                receipts.append({"id":cid,"certificate":cert,"native_sha256":sha(body)})
            ls.append("end CG10Audit")
            p=self.out/f"{tag}-{off}.lean";p.write_text("\n".join(ls))
            found=axioms(run_lean(self.lean,p))
            assert set(found)==expected and all(not a-ALLOWED for a in found.values()),found
        (self.out/f"{tag}-receipts.json").write_text(json.dumps(receipts,indent=2))
        print("NATIVE_REPLAY",tag,len(receipts),flush=True)
        return len(receipts)
