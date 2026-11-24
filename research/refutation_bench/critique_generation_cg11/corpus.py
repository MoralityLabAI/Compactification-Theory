"""CG-11: generated surface traps; actual dependency edges come only from Lean."""
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

SPLITS={"train":(64,111101,(1,3),False),"cal":(16,111102,(1,3),False),
        "test_depth":(24,111103,(6,9),False),"test_syntax":(24,111104,(1,3),True)}
BUDGETS=(8,16,32,64,128)
TRUSTED={"propext","Classical.choice","Quot.sound"}
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()

def families():
    for split,(count,seed,depth,syntax) in SPLITS.items():
        rng=random.Random(seed)
        for i in range(count):
            ns=f"CG11_{seed}_{i}"
            lines={};used=set();spec={}
            def name():
                while True:
                    x=f"n{rng.getrandbits(48):012x}"
                    if x not in used:used.add(x);return x
            def add(kind,typ,body=None,deps=()):
                n=name()
                lines[n]=f"{kind} {n} : {typ}"+("" if body is None else f" := {body}")
                spec[n]=set(deps)
                return n
            p=add("axiom","Prop")
            good=add("axiom",p,deps=[p]);bad=add("axiom",p,deps=[p])
            nat0=add("axiom","Nat");nat1=add("axiom","Nat")
            eq0=add("def","Prop",f"{nat0} = {nat0}",[nat0])
            eq1=add("def","Prop",f"{nat1} = {nat1}",[nat1])
            w0=add("axiom",eq0,deps=[eq0]);w1=add("axiom",eq1,deps=[eq1])
            def chain(base,typ,decoy=None):
                cur=base
                for _ in range(rng.randint(*depth)):
                    if decoy is None:body=f"by exact {cur}";deps=[typ,cur]
                    elif syntax:
                        if rng.random()<.7:
                            body=f"by let {decoy} : {typ} := {cur}; exact {decoy}"
                        else:
                            body=f"by exact (show {typ} from {cur})"
                        deps=[typ,cur,decoy]
                    elif rng.random()<.6:
                        body=f"by first | exact {cur} | exact {decoy}"
                        deps=[typ,cur,decoy]
                    else:body=cur;deps=[typ,cur]
                    cur=add("theorem",typ,body,deps)
                return cur
            left=chain(good,p,bad);right=chain(good,p,bad);taint=chain(bad,p,good)
            q0=chain(w0,eq0);q1=chain(w1,eq1)
            # Independent dead declarations retain misleading surface evidence.
            for _ in range(rng.randint(3,6)):
                chain(rng.choice([good,bad]),p)
            clean=add("theorem",f"{p} ∧ {p}",f"And.intro {left} {right}",[p,left,right])
            dirty=add("theorem",f"{p} ∧ {p}",f"And.intro {left} {taint}",[p,left,taint])
            clean_type=add("theorem",eq0,q0,[eq0,q0]);dirty_type=add("theorem",eq1,q1,[eq1,q1])
            ordered=[];todo=set(lines)
            while todo:
                ready=sorted(n for n in todo if spec[n]<=set(ordered))
                assert ready
                n=rng.choice(ready);ordered.append(n);todo.remove(n)
            raw="namespace "+ns+"\n"+"\n".join(lines[n] for n in ordered)+"\nend "+ns+"\n"
            forbid=[ns+"."+x for x in (bad,nat1)]
            roots=[(clean,False,"value"),(dirty,True,"value"),(clean_type,False,"type"),(dirty_type,True,"type")]
            cases=[]
            for root,tainted,mechanism in roots:
                fq=ns+"."+root
                cases.append({"id":sha(fq)[:24],"family":ns,"split":split,"root":fq,
                              "forbidden":forbid,"expected":tainted,"mechanism":mechanism})
            yield {"namespace":ns,"split":split,"source":raw,"cases":cases}

def parse(log):
    graphs={};root_ax={};prints={}
    def names(s):return sorted(set(x.strip() for x in s.split(",") if x.strip()))
    for line in log.splitlines():
        if "CG11_ROOT|" in line:
            r,a=line.split("CG11_ROOT|",1)[1].split("|",1)
            assert r not in root_ax
            root_ax[r]=names(a)
        if "CG11_NODE|" in line:
            r,n,k,t,v,ax=line.split("CG11_NODE|",1)[1].split("|",5)
            g=graphs.setdefault(r,{})
            assert n not in g,(r,n)
            g[n]={"kind":k,"type":names(t),"value":names(v),"axioms":names(ax)}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
        assert m[1] not in prints
        prints[m[1]]=names(m[3] or "")
    assert set(graphs)==set(root_ax)==set(prints),(set(graphs),set(root_ax),set(prints))
    return graphs,root_ax,prints

def edges(node):return sorted(set(node["type"]+node["value"]))
def node_axioms(name,node):
    return {name} if node["kind"]=="axiom" else set(node["axioms"]) if node["kind"]=="boundary" else set()
def reachable(graph,root):
    stack=[root];seen=set()
    while stack:
        n=stack.pop()
        if n in seen:continue
        assert n in graph,n
        seen.add(n);stack.extend(edges(graph[n]))
    return seen
def is_bad(name,node,forbidden):
    return bool(node_axioms(name,node)&set(forbidden))
def verify(graph,case,cert):
    if not isinstance(cert,dict):return False
    root=case["root"];forbidden=case["forbidden"]
    if cert.get("status")=="TAINTED":
        path=cert.get("path")
        return (type(path) is list and bool(path) and path[0]==root and all(n in graph for n in path)
                and all(b in edges(graph[a]) for a,b in zip(path,path[1:]))
                and is_bad(path[-1],graph[path[-1]],forbidden))
    if cert.get("status")=="CLEAN":
        rs=reachable(graph,root)
        return cert.get("closure")==sorted(rs) and all(not is_bad(n,graph[n],forbidden) for n in rs)
    return False

def surface(source,namespace):
    """Deliberately naive lexical dependency graph: includes shadowed names and unused branches."""
    declarations={}
    for line in source.splitlines():
        m=re.match(r"(axiom|def|theorem|opaque) (n[0-9a-f]+) : (.+)",line)
        if m:declarations[m[2]]={"kind":m[1],"body":m[3],"line":line}
    known=set(declarations);g={}
    for n,x in declarations.items():
        refs=set(re.findall(r"\bn[0-9a-f]+\b",x["body"]))&known
        g[namespace+"."+n]={"refs":sorted(namespace+"."+r for r in refs),
                             "kind":x["kind"],"length":len(x["line"]),
                             "tactic": "by " in x["body"],"alternative":"first |" in x["body"],
                             "shadow":"let " in x["body"],"show":"show " in x["body"]}
    return g

def extract(families,lean,out,extractor):
    header=extractor.read_text()
    assert header.startswith("import Lean")
    receipts=[];rows=[];graphs={}
    for off in range(0,len(families),4):
        chunk=families[off:off+4];lines=[header,"set_option linter.unusedVariables false","set_option pp.fullNames true"]
        for f in chunk:
            lines.append(f["source"])
            for c in f["cases"]:
                lines.extend([f"#cg11_deps {c['root']} within {f['namespace']}",f"#print axioms {c['root']}"])
        path=out/f"native-{off}.lean";path.write_text("\n".join(lines))
        p=subprocess.run([lean,str(path)],cwd=out,capture_output=True,text=True,timeout=240)
        log=p.stdout+p.stderr;path.with_suffix(".log").write_text(log)
        assert p.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",log),log[-10000:]
        gs,axs,printed=parse(log)
        assert set(gs)=={c["root"] for f in chunk for c in f["cases"]}
        for f in chunk:
            lex=surface(f["source"],f["namespace"])
            local_axioms={n for n,x in lex.items() if x["kind"]=="axiom"}
            for c in f["cases"]:
                root=c["root"];g=gs[root]
                rs=reachable(g,root)
                assert rs==set(g)
                found=set().union(*(node_axioms(n,g[n]) for n in rs))
                assert found==set(axs[root])==set(printed[root]),(root,found,axs[root],printed[root])
                assert found<=local_axioms|TRUSTED,(root,found-local_axioms-TRUSTED)
                label=bool(found&set(c["forbidden"]))
                assert label==c["expected"],(root,label,c)
                receipt={"id":c["id"],"root":root,"axioms":sorted(found),"nodes":len(g),
                         "source_sha256":sha(f["source"]),"graph_sha256":sha(json.dumps(g,sort_keys=True))}
                receipts.append(receipt)
                rows.append({**c,"source":f["source"],"surface":lex})
                graphs[c["id"]]=g
        print("NATIVE_BATCH",off,len(receipts),flush=True)
    assert len(receipts)==512 and len({x["id"] for x in rows})==512
    (out/"native-receipts.json").write_text(json.dumps(receipts,indent=2))
    (out/"graphs.json").write_text(json.dumps(graphs,indent=2))
    print("NATIVE_GRAPH",len(rows),sum(x["nodes"] for x in receipts),flush=True)
    return rows,graphs
