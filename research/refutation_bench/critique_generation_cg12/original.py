"""CG-12: original-source dependency comparison and matched hypothetical policies."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"critique_generation_cg11"))
import corpus as C

PIN="6d02b611c7358216379cb36952512a104e0bdb8e"
ROOTS={
 "ObservableQuotient":("factor_iff_fiber","factor_unique"),
 "AuthorizationQuotient":("classifier_exists_iff_fiber","collision_forces_error","coarse_compatible_implies_fine"),
 "NOR19StreamedVerifier":("streamed_exactly_iff_materialized_equal","streamed_rejection_iff_unequal"),
 "MatchedStepMargin":("positive_margin_stable","negative_margin_stable"),
}
POLICIES={
 "classical":[],
 "choice_free":["Classical.choice"],
}
def native(lean,path,cwd,env,opts=()):
    p=subprocess.run([lean,*opts,str(path)],cwd=cwd,env=env,capture_output=True,text=True,timeout=240)
    log=p.stdout+p.stderr
    return p.returncode,log
def assert_native(result,logpath):
    code,log=result
    logpath.write_text(log)
    assert code==0 and not re.search(r"error:|sorryAx|uses 'sorry'",log),log[-10000:]
    return log
def axiom_reports(log):
    result={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
        assert m[1] not in result
        result[m[1]]=sorted(set(x.strip() for x in (m[3] or "").split(",") if x.strip()))
    return result
def certificate(graph,case):
    root=case["root"];seen=set();frontier=[[root]]
    while frontier:
        path=frontier.pop(0);node=path[-1]
        if node in seen:continue
        seen.add(node)
        if C.is_bad(node,graph[node],case["forbidden"]):
            return {"status":"TAINTED","path":path}
        frontier.extend(path+[child] for child in C.edges(graph[node]) if child not in seen)
    return {"status":"CLEAN","closure":sorted(seen)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    a=p.parse_args()
    source=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()==PIN
    assert (source/"lean-toolchain").read_text().strip()=="leanprover/lean4:v4.35.0"
    cache=out/"olean";(cache/"MathCan").mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LEAN_PATH=str(cache))
    extractor=(HERE.parent/"critique_generation_cg11"/"Extract.lean").read_text()
    assert extractor.startswith("import Lean\n")
    command=extractor.removeprefix("import Lean\n")
    receipts=[];case_receipts=[];graphs={}
    for mod,roots in ROOTS.items():
        scope="MathCan."+mod
        rel=Path("MathCan")/(mod+".lean")
        raw=(source/rel).read_bytes()
        text=raw.decode();assert text.startswith("import Init\n")
        assert re.findall(r"^import (.+)$",text,re.M)==["Init"]
        # Compile untouched source, then separately query imported roots.
        assert_native(native(a.lean,rel,source,env,("-o",str(cache/rel.with_suffix(".olean")))),out/(mod+"-original.log"))
        qualified=[scope+"."+r for r in roots]
        audit=out/(mod+"-imported.lean")
        audit.write_text("import "+scope+"\nset_option pp.fullNames true\n"+"\n".join("#print axioms "+r for r in qualified)+"\n")
        original=axiom_reports(assert_native(native(a.lean,audit,out,env),audit.with_suffix(".log")))
        assert set(original)==set(qualified)
        # Additional import is explicit; original file is never altered.
        instrumented="import Lean\n"+text+"\n"+command+"\nset_option pp.fullNames true\n"
        for r in qualified:
            instrumented+=f"#cg11_deps {r} within {scope}\n#print axioms {r}\n"
        ip=out/(mod+"-instrumented.lean");ip.write_text(instrumented)
        log=assert_native(native(a.lean,ip,out,env),ip.with_suffix(".log"))
        # Earlier #print reports in the original source are outside this window.
        audit_log=log[log.index("CG11_ROOT|"):]
        gs,axs,printed=C.parse(audit_log)
        assert set(gs)==set(qualified)
        for r in qualified:
            g=gs[r];rs=C.reachable(g,r)
            assert rs==set(g)
            observed=sorted(set().union(*(C.node_axioms(n,g[n]) for n in rs)))
            assert observed==axs[r]==printed[r]==original[r],(r,observed,axs[r],printed[r],original[r])
            assert set(observed)<=C.TRUSTED,(r,observed)
            graph_hash=C.sha(json.dumps(g,sort_keys=True));graphs[r]=g
            row={"module":mod,"root":r,"original_source_sha256":C.sha(raw),
                 "instrumented_source_sha256":C.sha(instrumented),"graph_sha256":graph_hash,
                 "original_axioms":original[r],"nodes":len(g),
                 "boundaries":sum(x["kind"]=="boundary" for x in g.values())}
            receipts.append(row)
            for policy,forbidden in POLICIES.items():
                case={"root":r,"forbidden":forbidden};cert=certificate(g,case)
                assert C.verify(g,case,cert)
                expected=bool(set(observed)&set(forbidden))
                assert (cert["status"]=="TAINTED")==expected
                case_receipts.append({"id":C.sha(r+policy)[:24],"policy":policy,**row,"forbidden":forbidden,
                                      "certificate":cert,"verified":True})
            print("ROOT",json.dumps(row),flush=True)
    assert len(receipts)==9 and len(case_receipts)==18
    for name,data in (("native-roots",receipts),("policy-cases",case_receipts),("graphs",graphs)):
        (out/(name+".json")).write_text(json.dumps(data,indent=2))
    counts={p:{s:sum(x["policy"]==p and x["certificate"]["status"]==s for x in case_receipts)
               for s in ("CLEAN","TAINTED")} for p in POLICIES}
    print("NATIVE_ORIGINAL",len(ROOTS),len(receipts),len(case_receipts),json.dumps(counts),flush=True)
if __name__=="__main__":main()
