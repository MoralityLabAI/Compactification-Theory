"""CG-08: finite Horn correspondence cases tied to the original RH interface."""
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path

PIN="6d02b611c7358216379cb36952512a104e0bdb8e"
MODULE="MathCan.RHMainLogic"
SELECTED=MODULE+".final_rh_route_from_supplied_mathematical_bridges"
ALLOWED={"propext","Classical.choice","Quot.sound"}
SPLITS={"train":(64,108801,(1,3),False),"cal":(16,108802,(1,3),False),
        "test_depth":(24,108803,(4,6),False),"test_bridge":(24,108804,(1,3),True)}
BUDGETS=(8,16,32,64,128,256)
KINDS=("SOURCE_SUPPORTED","INDEPENDENT_WITH_GAP","REFUTED")
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def public(case):return {k:case[k] for k in ("n","facts","rules","roles")}
def generate(split,i,rng,depths,bridge_gap):
    n=0;rules=[]
    def atom():
        nonlocal n
        a=n;n+=1;return a
    def rule(p,h):
        rules.append({"prem":list(dict.fromkeys(p)),"head":h});return len(rules)-1
    seed,decoy_seed=atom(),atom()
    R,C,V,E,H=[atom() for _ in range(5)]
    def path(p,target,depth):
        out=[]
        for j in range(depth):
            h=target if j==depth-1 else atom()
            out.append(rule(p,h));p=[h]
        return out
    core=[path([seed],R,rng.randint(*depths)),path([seed],C,rng.randint(*depths)),
          path([R],V,rng.randint(*depths)),path([R,C,V],E,rng.randint(*depths)),
          path([E],H,rng.randint(*depths))]
    alt=path([seed],H,rng.randint(1,3))
    decoys=[]
    for _ in range(rng.randint(4,8)):
        h=atom()
        candidates=list(range(h))
        p=rng.sample(candidates,rng.randint(1,2))
        decoys.append(rule(p,h))
    gap=rng.choice(core[rng.choice((2,3) if bridge_gap else (0,1))])
    benign=rng.choice(decoys)
    order=list(range(n));rng.shuffle(order)
    remap={k:v for k,v in enumerate(order)}
    rule_order=list(range(len(rules)));rng.shuffle(rule_order)
    family=f"CG08-{SPLITS[split][1]}-{i}"
    result=[]
    for arm,removed,expected in (
        ("full",{alt[-1]},"SOURCE_SUPPORTED"),
        ("benign",{alt[-1],benign},"SOURCE_SUPPORTED"),
        ("gap",{gap},"INDEPENDENT_WITH_GAP"),
        ("false",{gap,alt[-1]},"REFUTED")):
        rr=[{"prem":[remap[x] for x in rules[j]["prem"]],"head":remap[rules[j]["head"]]}
            for j in rule_order if j not in removed]
        case={"id":sha(family+arm)[:20],"family":family,"split":split,
              "n":n,"facts":[remap[seed],remap[decoy_seed]],"rules":rr,
              "roles":[remap[x] for x in (R,C,V,E,H)],"expected":expected}
        ref=reference(public(case))
        assert ref["kind"]==expected,(family,arm,ref,expected)
        assert verify(public(case),ref)
        result.append(case)
    rng.shuffle(result)
    return result
def cases():
    rows=[]
    for split,(count,seed,depths,bridge_gap) in SPLITS.items():
        rng=random.Random(seed)
        for i in range(count):
            rows.extend(generate(split,i,rng,depths,bridge_gap))
    assert len(rows)==512 and len({x["id"] for x in rows})==512
    assert len({sha(json.dumps(public(x),sort_keys=True)) for x in rows})==512
    return rows
def assumptions(case):
    R,C,V,E,H=case["roles"]
    return [(),(R,),(R,C,V),(E,)]
def source_queries(case):
    R,C,V,E,H=case["roles"]
    return [(0,R),(0,C),(1,V),(2,E),(3,H)]
def goals(case):
    R,C,V,E,H=case["roles"]
    return [(R,C,H),(V,),(E,),(H,)]
def initial(case,ctx):
    return set(case["facts"])|set(assumptions(case)[ctx])
def extend(case,known):
    """Deterministic full closure, used for reference and train-only teacher."""
    known=set(known);trace=[]
    while True:
        chosen=None
        for i,r in enumerate(case["rules"]):
            if r["head"] not in known and set(r["prem"])<=known:
                chosen=i;break
        if chosen is None:return known,trace
        known.add(case["rules"][chosen]["head"]);trace.append(chosen)
def needed(case,known,trace,targets):
    """Extract proof-relevant subset of an already supplied trace."""
    known=set(known);producer={}
    for j in trace:
        r=case["rules"][j]
        assert r["head"] not in known and set(r["prem"])<=known
        known.add(r["head"]);producer[r["head"]]=j
    required=set()
    def visit(a):
        assert a in known
        if a not in producer:return
        j=producer[a]
        if j in required:return
        required.add(j)
        for p in case["rules"][j]["prem"]:visit(p)
    for a in targets:visit(a)
    return required
def trim(case,ctx,trace,targets):
    keep=needed(case,initial(case,ctx),trace,targets)
    state=initial(case,ctx);out=[]
    while keep:
        j=min(j for j in keep if set(case["rules"][j]["prem"])<=state)
        state.add(case["rules"][j]["head"]);keep.remove(j);out.append(j)
    return out
def saturated(case,known):
    return all(r["head"] in known or not set(r["prem"])<=known for r in case["rules"])
def certificate(case,known,traces,closed):
    H=case["roles"][-1];queries=source_queries(case)
    if all(a in known[ctx] for ctx,a in queries):
        return {"kind":"SOURCE_SUPPORTED","traces":[trim(case,ctx,traces[ctx],[a for c,a in queries if c==ctx]) for ctx in range(4)]}
    if H in known[0]:
        for ctx,a in queries:
            if a not in known[ctx] and closed[ctx]:
                return {"kind":"INDEPENDENT_WITH_GAP","trace":trim(case,0,traces[0],[H]),
                        "gap":[ctx,a],"model":sorted(known[ctx])}
    elif closed[0]:
        return {"kind":"REFUTED","model":sorted(known[0])}
    return None
def reference(case):
    pairs=[extend(case,initial(case,c)) for c in range(4)]
    cert=certificate(case,[p[0] for p in pairs],[p[1] for p in pairs],[True]*4)
    assert cert is not None
    return cert
def replay_trace(case,ctx,trace):
    if type(trace) is not list:return None
    state=initial(case,ctx)
    for j in trace:
        if type(j) is not int or not 0<=j<len(case["rules"]):return None
        r=case["rules"][j]
        if r["head"] in state or any(p not in state for p in r["prem"]):return None
        state.add(r["head"])
    return state
def model_ok(case,ctx,target,model):
    if type(model) is not list or any(type(x) is not int or not 0<=x<case["n"] for x in model):return False
    if sorted(set(model))!=model:return False
    state=set(model)
    if not initial(case,ctx)<=state or target in state:return False
    return all(any(p not in state for p in r["prem"]) or r["head"] in state for r in case["rules"])
def verify(case,cert):
    if not isinstance(cert,dict):return False
    H=case["roles"][-1];kind=cert.get("kind")
    if kind=="SOURCE_SUPPORTED":
        ts=cert.get("traces")
        if type(ts) is not list or len(ts)!=4:return False
        states=[replay_trace(case,c,t) for c,t in enumerate(ts)]
        return all(s is not None for s in states) and all(a in states[c] for c,a in source_queries(case))
    if kind=="INDEPENDENT_WITH_GAP":
        gap=cert.get("gap")
        if type(gap) is not list or len(gap)!=2 or tuple(gap) not in source_queries(case):return False
        state=replay_trace(case,0,cert.get("trace"))
        return state is not None and H in state and model_ok(case,*gap,cert.get("model"))
    if kind=="REFUTED":
        return model_ok(case,0,H,cert.get("model"))
    return False

ADAPTER="""theorem cg08Bridge (R C V E H : Prop) (hr : R) (hc : C)
    (win : R → V) (hev : R → C → V → E) (last : E → H) : H :=
  MathCan.RHMainLogic.final_rh_route_from_supplied_mathematical_bridges
    hr hc win (fun lit hv => hev lit.caraianiLocalGlobalDual lit.localLanglandsExists hv) last
"""
def p(a):return f"p{a}"
def arrow(xs,goal):return " → ".join([p(x) for x in xs]+[p(goal)])
def prop(case,ctx,goal):
    types=[p(a) for a in case["facts"]]+["("+arrow(r["prem"],r["head"])+")" for r in case["rules"]]
    types += [p(a) for a in assumptions(case)[ctx]]
    return "∀ ("+" ".join(p(a) for a in range(case["n"]))+" : Prop), "+" → ".join(types+[p(goal)])
def binders(case):
    items=["("+" ".join(p(a) for a in range(case["n"]))+" : Prop)"]
    items += [f"(f{j} : {p(a)})" for j,a in enumerate(case["facts"])]
    items += [f"(r{j} : {arrow(r['prem'],r['head'])})" for j,r in enumerate(case["rules"])]
    return " ".join(items)
def derive(case,ctx,trace,indent):
    names={a:f"f{j}" for j,a in enumerate(case["facts"])}
    names.update({a:f"a{ctx}_{j}" for j,a in enumerate(assumptions(case)[ctx])})
    lines=[]
    for j in trace:
        r=case["rules"][j];h=f"q{ctx}_{j}"
        term=" ".join([f"r{j}"]+[names[x] for x in r["prem"]])
        lines.append(indent+f"have {h} : {p(r['head'])} := {term}")
        names[r["head"]]=h
    return lines,names
def negation(case,ctx,target,model,name):
    m=set(model);assert model_ok(case,ctx,target,model)
    args=["True" if a in m else "False" for a in range(case["n"])]
    args += ["True.intro"]*len(case["facts"])
    for r in case["rules"]:
        ns=[f"h{i}" for i in range(len(r["prem"]))]
        body="True.intro" if r["head"] in m else ns[next(i for i,a in enumerate(r["prem"]) if a not in m)]
        args.append("(fun "+" ".join(ns)+" => "+body+")")
    args += ["True.intro"]*len(assumptions(case)[ctx])
    return f"theorem {name} : ¬ ({prop(case,ctx,target)}) := by\n  intro h\n  exact h "+" ".join(args)
def render(case,cert,name):
    assert verify(case,cert)
    R,C,V,E,H=case["roles"];kind=cert["kind"]
    if kind=="REFUTED":
        return [(name,negation(case,0,H,cert["model"],name))]
    if kind=="INDEPENDENT_WITH_GAP":
        lines,ns=derive(case,0,cert["trace"],"  ")
        body="\n".join([f"theorem {name} {binders(case)} : {p(H)} := by",*lines,"  exact "+ns[H]])
        ctx,target=cert["gap"]
        return [(name,body),(name+"Gap",negation(case,ctx,target,cert["model"],name+"Gap"))]
    ts=cert["traces"]
    lines,ns=derive(case,0,ts[0],"  ")
    out=[f"theorem {name} {binders(case)} : {p(H)} := by",*lines]
    for ctx,target,local in [(1,V,"window"),(2,E,"every"),(3,H,"last")]:
        out.append(f"  have {local} : {arrow(assumptions(case)[ctx],target)} := by")
        out.append("    intro "+" ".join(f"a{ctx}_{j}" for j in range(len(assumptions(case)[ctx]))))
        ls,mp=derive(case,ctx,ts[ctx],"    ");out.extend(ls)
        out.append("    exact "+mp[target])
    out.append("  exact cg08Bridge "+" ".join(p(x) for x in (R,C,V,E,H))+" "+ns[R]+" "+ns[C]+" window every last")
    return [(name,"\n".join(out))]

def run_lean(lean,path,cwd,env,log,opts=()):
    proc=subprocess.run([lean,*opts,str(path)],cwd=cwd,env=env,capture_output=True,text=True,timeout=240)
    s=proc.stdout+proc.stderr;log.write_text(s)
    assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",s),s[-10000:]
    return s
def axiom_check(s,expected):
    found={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",s):
        assert m[1] not in found
        found[m[1]]={a.strip() for a in (m[3] or "").split(",") if a.strip()}
    assert set(found)==set(expected),(set(found)-set(expected),set(expected)-set(found))
    assert all(not (xs-ALLOWED) for xs in found.values())
class Native:
    def __init__(self,source,out,lean):
        self.source=source.resolve();self.out=out.resolve();self.lean=lean
        assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=self.source,text=True).strip()==PIN
        cache=self.out/"olean";(cache/"MathCan").mkdir(parents=True,exist_ok=True)
        self.env=dict(os.environ,LEAN_PATH=str(cache))
        rel=Path("MathCan/RHMainLogic.lean")
        raw=(self.source/rel).read_bytes()
        self.raw=raw.decode();self.source_hash=sha(raw)
        run_lean(lean,rel,self.source,self.env,self.out/"original-source.log",("-o",str(cache/rel.with_suffix(".olean"))))
        path=self.out/"adapter.lean";path.write_text("import "+MODULE+"\nnamespace CG08Audit\n"+ADAPTER+"\n#print axioms "+SELECTED+"\n#print axioms cg08Bridge\nend CG08Audit\n")
        s=run_lean(lean,path,self.out,self.env,path.with_suffix(".log"))
        axiom_check(s,[SELECTED,"CG08Audit.cg08Bridge"])
        print("NATIVE_ADAPTER",self.source_hash,flush=True)
    def replay(self,rows,accepted,tag):
        receipt=[];declarations=0
        cases={x["id"]:public(x) for x in rows}
        items=[(cid,cert) for cid,cs in sorted(accepted.items()) for cert in cs]
        for off in range(0,len(items),16):
            lines=["import "+MODULE,"set_option linter.unusedVariables false","namespace CG08Audit",ADAPTER]
            expected=[]
            for j,(cid,cert) in enumerate(items[off:off+16]):
                bodies=render(cases[cid],cert,f"cert{j}")
                for name,body in bodies:
                    lines.extend([body,"#print axioms "+name])
                    expected.append("CG08Audit."+name)
                receipt.append({"id":cid,"kind":cert["kind"],"certificate":cert,
                                "native_sha256":sha("\n".join(b for _,b in bodies))})
            lines.append("end CG08Audit")
            path=self.out/f"{tag}-{off}.lean";path.write_text("\n".join(lines)+"\n")
            s=run_lean(self.lean,path,self.out,self.env,path.with_suffix(".log"))
            axiom_check(s,expected);declarations+=len(expected)
        (self.out/f"{tag}-receipts.json").write_text(json.dumps(receipt,indent=2))
        print("NATIVE_REPLAY",tag,len(items),declarations,flush=True)
        return len(items)
