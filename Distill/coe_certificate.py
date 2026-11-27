"""Source-pinned native COE mathematical certificate reuse.

For an ordinary binary observer (margin 1), distinct words on edges provide
an upper bound. A clique of size c forces at least ceil(log2(c)) bits. When
the bounds agree, the certificate proves the minimum width of that graph.
Transport rechecks both witnesses on the target; a hash alone proves nothing.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import json
import time
from pathlib import Path
from .core import digest, require, object_keys, write_json
from .native import _load, PINS, BASE
from .registry import ROOT
from .instances import Rng

SCHEMA="distill.coe_certificate.v1"
NATIVE_PATH,NATIVE_BLOB,_,_=PINS[("COE-01","cycle_observer")]
SOURCE={"repository":"MoralityLabAI/Compactification-Theory","commit":BASE,"path":NATIVE_PATH,"blob":NATIVE_BLOB}

def graph(n,edges):
    require(type(n) is int and 1<=n<=8,"graph size outside native scope")
    clean=set()
    for edge in edges:
        require(isinstance(edge,(list,tuple)) and len(edge)==2,"invalid edge")
        u,v=edge
        require(type(u) is int and type(v) is int and 0<=u<n and 0<=v<n and u!=v,"invalid vertices")
        clean.add(tuple(sorted((u,v))))
    return {"n":n,"edges":[list(e) for e in sorted(clean)]}

def verify_graph(g):
    object_keys(g,{"n","edges"},"graph")
    require(isinstance(g["edges"],list) and g==graph(g["n"],g["edges"]),"graph must be canonical")
    return g

def clique_bound(g):
    """Independent finite clique witness; sufficient but not complete."""
    n=g["n"]
    ee={tuple(e) for e in g["edges"]}
    for k in range(n,0,-1):
        for c in combinations(range(n),k):
            if all(e in ee for e in combinations(c,2)):
                return list(c)
    raise ValueError("empty graph")

def _certificate(g,width,words,clique):
    body={"schema_version":SCHEMA,"source":SOURCE,"graph":g,"width":width,
          "words":list(words),"clique":list(clique),"margin":1,
          "scope":"ordinary finite binary observer minimum width"}
    return {**body,"certificate_id":digest(body)}

def verify_certificate(c):
    object_keys(c,{"schema_version","source","graph","width","words","clique","margin","scope","certificate_id"},"certificate")
    require(c["schema_version"]==SCHEMA and c["source"]==SOURCE and type(c["margin"]) is int and c["margin"]==1
            and c["scope"]=="ordinary finite binary observer minimum width","certificate scope mismatch")
    require(c["certificate_id"]==digest({k:v for k,v in c.items() if k!="certificate_id"}),"certificate digest mismatch")
    g=verify_graph(c["graph"])
    n,k,words,clique=g["n"],c["width"],c["words"],c["clique"]
    require(type(k) is int and 0<=k<=3,"bad width")
    require(isinstance(words,list) and len(words)==n and all(type(w) is int and 0<=w<1<<k for w in words),"bad codewords")
    require(all(words[u]!=words[v] for u,v in g["edges"]),"upper bound witness invalid")
    require(isinstance(clique,list) and clique and all(type(i) is int and 0<=i<n for i in clique)
            and clique==sorted(set(clique)),"bad clique")
    ee={tuple(e) for e in g["edges"]}
    require(all(e in ee for e in combinations(clique,2)),"lower bound witness invalid")
    require((len(clique)-1).bit_length()==k,"clique lower bound does not meet upper bound")
    return k

class Native:
    def __init__(self,root=ROOT):
        self.module=_load(root,NATIVE_PATH,NATIVE_BLOB)
        self.solves=0
    def cold(self,g):
        verify_graph(g)
        self.solves+=1
        k,words=self.module.minimum_observer(g["n"],[tuple(e) for e in g["edges"]],margin=1)
        require(self.module.validate_code(g["n"],[tuple(e) for e in g["edges"]],k,tuple(words),margin=1),"native solver witness invalid")
        c=_certificate(g,k,words,clique_bound(g))
        verify_certificate(c)  # may reject if this lower-bound certificate family is incomplete
        return c

def transport(source,target,mapping):
    verify_certificate(source)
    verify_graph(target)
    n=source["graph"]["n"]
    require(target["n"]==n and isinstance(mapping,list) and len(mapping)==n and
            all(type(i) is int for i in mapping) and sorted(mapping)==list(range(n)),"bad vertex bijection")
    words=[0]*n
    for u,v in enumerate(mapping):
        words[v]=source["words"][u]
    clique=sorted(mapping[u] for u in source["clique"])
    c=_certificate(target,source["width"],words,clique)
    verify_certificate(c)
    receipt={"source_certificate_id":source["certificate_id"],"target_certificate_id":c["certificate_id"],
             "mapping":mapping,"target_graph_sha256":digest(target),
             "scope":"both upper and lower witnesses rechecked on target; no general isomorphism assumption"}
    return c,{**receipt,"receipt_id":digest(receipt)}

def shuffled(r,n):
    xs=list(range(n))
    for i in range(n-1,0,-1):
        j=r.below(i+1)
        xs[i],xs[j]=xs[j],xs[i]
    return xs

def root_graph(seed,ordinal):
    r=Rng(["distill.coe_reuse.v1",seed,ordinal])
    n=7+r.below(2)
    k=3+2*r.below(2)
    colors=list(range(k))+[r.below(k) for _ in range(n-k)]
    ee=[(u,v) for u,v in combinations(range(n),2)
        if colors[u]!=colors[v] and (v<k or r.below(4)!=0)]
    anchor=graph(n,ee)
    targets=[]
    for _ in range(2):
        mapping=shuffled(r,n)
        targets.append((graph(n,[(mapping[u],mapping[v]) for u,v in ee]),mapping))
    # A complete graph has no repeated codewords, so it invalidates a planted
    # 3/5-color certificate (native branch-and-bound uses at most k colors).
    targets.append((graph(n,combinations(range(n),2)),list(range(n))))
    return anchor,targets

def benchmark(out,*,seed=2026112835,roots=24):
    require(type(seed) is int and 0<=seed<2**63 and type(roots) is int and 1<=roots<=64,"invalid benchmark")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    native=Native()
    rows=[]
    seen=set()
    for ordinal in range(roots):
        anchor,targets=root_graph(seed,ordinal)
        rid=digest({"anchor":anchor,"targets":targets})
        require(rid not in seen,"duplicate graph root")
        seen.add(rid)
        cold0=native.solves
        start=time.process_time_ns()
        cold=[native.cold(g) for g in [anchor]+[g for g,_ in targets]]
        cold_ns=time.process_time_ns()-start
        cold_solves=native.solves-cold0
        start=time.process_time_ns()
        reuse0=native.solves
        base=native.cold(anchor)
        transported=[]
        failures=0
        outputs=[base]
        for g,mapping in targets:
            try:
                c,receipt=transport(base,g,mapping)
                transported.append(receipt)
            except ValueError:
                failures+=1
                c=native.cold(g)
            outputs.append(c)
        reuse_ns=time.process_time_ns()-start
        reuse_solves=native.solves-reuse0
        require(len(transported)==2 and failures==1,"registered reuse/shift fixture mismatch")
        require([verify_certificate(c) for c in cold]==[verify_certificate(c) for c in outputs],"cold/transport exact width mismatch")
        rows.append({"root_id":rid,"cold_solves":cold_solves,"reuse_solves":reuse_solves,
                     "transport_success":len(transported),"transport_reject":failures,
                     "cold_cpu_ns":cold_ns,"reuse_cpu_ns":reuse_ns,
                     "certificates":outputs,"transport_receipts":transported})
    totals={k:sum(r[k] for r in rows) for k in ("cold_solves","reuse_solves","transport_success",
            "transport_reject","cold_cpu_ns","reuse_cpu_ns")}
    result={"schema_version":"distill.native_certificate_reuse.v1","source":SOURCE,"seed":seed,"roots":roots,
            "checked_target_widths":4*roots,"totals":totals,"rows":rows,"training_admission":False,
            "scope":"native finite COE solver and independently checked clique/code witnesses; CPU descriptive, no Lean proof or trained controller"}
    result["receipt_id"]=digest(result)
    write_json(root/"receipt.json",result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=2026112835)
    p.add_argument("--roots",type=int,default=24)
    a=p.parse_args()
    r=benchmark(a.out,seed=a.seed,roots=a.roots)
    print(json.dumps({k:v for k,v in r.items() if k!="rows"},sort_keys=True))

if __name__=="__main__":
    main()
