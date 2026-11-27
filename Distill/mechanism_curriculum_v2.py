"""DISTILL-12: independently specified finite mechanism curriculum v2.

V1 remains frozen reference evidence. These toy models use only generated
mathematical inputs and exact labels. No provider outputs or training admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache, reduce
from itertools import combinations, product
import json
from math import gcd
from pathlib import Path

from .core import canonical, digest, require, write_json
from .instances import Rng

SCHEMA="distill.mechanism_curriculum.v2"
SPLITS=(("train",2026113011,96,False),("calibration",2026113012,24,False),
        ("heldout_roots",2026113013,48,False),("heldout_structural",2026113014,48,True))
ACTIONS=("defer","recommend0","recommend1","direct","transport")

def frac(x):
    x=F(x)
    return f"{x.numerator}/{x.denominator}"

def state(b,anchor,law):
    return {"budget":b,"anchor_receipt":anchor,"law_receipt":law}

def project(p="1/2",cert=11,correct=6,wrong=-12,direct=3,transport=2,scope=True):
    return {"p":frac(p),"cert":cert,"correct":correct,"wrong":wrong,
            "direct":direct,"transport":transport,"scope":scope}

def normalize(kind,d):
    """Canonicalize registered nuisance symmetries, excluding row metadata.

    This is not a general isomorphism solver. Project order and loss units
    remain significant. Pair swap, atom order, repeated identical atoms and
    contexts, rational representation, and common weight scaling do not.
    """
    if kind=="receipt_continuation":
        projects=[{**p,"p":frac(p["p"])} for p in d["projects"]]
        states=sorted(d["states"],key=lambda s:(s["budget"],s["anchor_receipt"],s["law_receipt"]))
        return {"projects":projects,"states":states}
    require(kind=="multi_context_quantization","unknown mechanism")
    atoms=Counter()
    for a in d["atoms"]:
        atoms[F(a["p"])]+=a["w"]
    g=reduce(gcd,atoms.values())
    ctx=Counter()
    for c in d["contexts"]:
        key=(c["fp"],c["fn"],None if c["defer"] is None else frac(c["defer"]))
        ctx[key]+=c["w"]
    h=reduce(gcd,ctx.values())
    return {"bins":d["bins"],"atoms":[{"p":frac(p),"w":w//g} for p,w in sorted(atoms.items())],
            "contexts":[{"fp":k[0],"fn":k[1],"defer":k[2],"w":w//h}
                        for k,w in sorted(ctx.items(),key=lambda kv:(kv[0][0],kv[0][1],str(kv[0][2])))]}

def content_id(kind,d):
    return digest({"mechanism":kind,"instance":normalize(kind,d)})

def options(p,b,eligible):
    q=F(p["p"])
    result={"defer":(0,F(0)),
            "recommend0":(0,(1-q)*p["correct"]+q*p["wrong"]),
            "recommend1":(0,q*p["correct"]+(1-q)*p["wrong"])}
    if b>=p["direct"]:
        result["direct"]=(p["direct"],F(p["cert"]-p["direct"]))
    if eligible and p["scope"] and b>=p["transport"]:
        result["transport"]=(p["transport"],F(p["cert"]-p["transport"]))
    return result

def quotient_dp(projects,s):
    """All-action deterministic expected-reward signature and optimal value.

    One decision per project, receipts fixed, no observations or new receipt
    acquisition. Costs consume shared budget; project expected rewards add.
    A class preserves each legal action label, cost, immediate expected
    reward and successor class. It does not claim stochastic equivalence.
    """
    eligible=s["anchor_receipt"] and s["law_receipt"]
    @lru_cache(None)
    def solve(i,b):
        if i==len(projects):
            return (),F(0)
        children=[]
        values=[]
        for a,(c,r) in sorted(options(projects[i],b,eligible).items()):
            sig,v=solve(i+1,b-c)
            children.append((a,c,frac(r),sig))
            values.append(r+v)
        return tuple(children),max(values)
    return solve(0,s["budget"])

def quotient_traces(projects,s):
    """Independent non-DP oracle: enumerate whole action strings.

    Compute feasibility and reward directly from the raw project schedule,
    without calling options or quotient_dp. Every prefix has a defer extension.
    """
    out={}
    for path in product(ACTIONS,repeat=len(projects)):
        remaining=s["budget"]
        receipts=[]
        for p,a in zip(projects,path):
            prob=F(p["p"])
            if a=="defer":
                c,r=0,F(0)
            elif a in ("recommend0","recommend1"):
                success=prob if a=="recommend1" else 1-prob
                c,r=0,success*p["correct"]+(1-success)*p["wrong"]
            elif a=="direct":
                c=p["direct"]
                r=F(p["cert"])-c
            else:
                if not (p["scope"] and s["anchor_receipt"] and s["law_receipt"]):
                    break
                c=p["transport"]
                r=F(p["cert"])-c
            if c>remaining:
                break
            remaining-=c
            receipts.append((c,r))
        else:
            out[path]=tuple(receipts)
    return out

def capped_key(projects,s):
    flags=tuple(bool(p["scope"] and s["anchor_receipt"] and s["law_receipt"]) for p in projects)
    cap=sum(max(p["direct"],p["transport"]) if f else p["direct"] for p,f in zip(projects,flags))
    return (flags,min(s["budget"],cap))

def assess_quotient(d):
    ps=d["projects"]
    a,b=d["states"]
    sa,va=quotient_dp(ps,a)
    sb,vb=quotient_dp(ps,b)
    ta,tb=quotient_traces(ps,a),quotient_traces(ps,b)
    ea=max(sum((r for c,r in path),F(0)) for path in ta.values())
    eb=max(sum((r for c,r in path),F(0)) for path in tb.values())
    equivalent=sa==sb
    require((va,vb)==(ea,eb) and equivalent==(ta==tb),"independent quotient oracle mismatch")
    cheap=capped_key(ps,a)==capped_key(ps,b)
    require(not cheap or equivalent,"saturation key unsound")
    # Distinguish value-only equivalence from the stronger all-action contract.
    gap=abs(va-vb)
    require(not equivalent or gap==0,"equivalent states have different value")
    def loss(merge):
        return F(1) if merge and not equivalent else F(1,20) if not merge and equivalent else F(0)
    witness=None
    if not equivalent:
        path=next(k for k in sorted(ta.keys()|tb.keys()) if ta.get(k)!=tb.get(k))
        def row(t):
            return [[c,frac(r)] for c,r in t[path]] if path in t else None
        witness={"actions":list(path),"a":row(ta),"b":row(tb)}
    return {"equivalent":equivalent,"value_equal":va==vb,"value_a":frac(va),"value_b":frac(vb),
            "value_gap":frac(gap),"equal_value_unsafe_merge":va==vb and not equivalent,
            "witness":witness,"optimal_action":"merge" if equivalent else "separate",
            "action_regrets":{"merge":frac(loss(True)),"separate":frac(loss(False))},
            "baseline_regrets":{"always_merge":frac(loss(True)),"always_separate":frac(loss(False)),
               "value_only":frac(loss(va==vb)),"capped_eligibility":frac(loss(cheap))},
            "independent_traces":len(ta)+len(tb)}

def losses(p,c):
    values=[p*c["fn"],(1-p)*c["fp"]]
    if c["defer"] is not None:
        values.append(F(c["defer"]))
    return values

def segment(d,i,j):
    atoms=d["atoms"]
    contexts=d["contexts"]
    W=sum(a["w"] for a in atoms)
    C=sum(c["w"] for c in contexts)
    part=atoms[i:j]
    w=sum(a["w"] for a in part)
    mean=sum(a["w"]*F(a["p"]) for a in part)/w
    brier=sum(a["w"]*(F(a["p"])-mean)**2 for a in part)/W
    regret=F(0)
    for c in contexts:
        scores=losses(mean,c)
        act=min(range(len(scores)),key=lambda k:(scores[k],k))
        for a in part:
            v=losses(F(a["p"]),c)
            regret+=F(a["w"],W)*F(c["w"],C)*(v[act]-min(v))
    return regret,brier

def partition_dp(d,objective="decision"):
    """Optimal contiguous k-bin partition by additive segment dynamic program."""
    n,k=len(d["atoms"]),d["bins"]
    require(1<=k<=n and objective in ("decision","brier"),"bad partition")
    @lru_cache(None)
    def score(i,j):
        return segment(d,i,j)
    @lru_cache(None)
    def best(end,bins):
        if bins==1:
            r,b=score(0,end)
            return r,b,()
        rows=[]
        for cut in range(bins-1,end):
            r,b,cuts=best(cut,bins-1)
            rr,bb=score(cut,end)
            rows.append((r+rr,b+bb,cuts+(cut,)))
        return min(rows,key=lambda x:x if objective=="decision" else (x[1],x[0],x[2]))
    return best(n,k)

def partition_enumeration(d):
    """Independent exhaustive cuts; use direct bin losses and pairwise variance."""
    atoms=d["atoms"]
    contexts=d["contexts"]
    W=sum(a["w"] for a in atoms)
    C=sum(c["w"] for c in contexts)
    # Explicit loss vectors avoid reuse of losses/segment/partition_dp.
    def raw(p,c):
        v=[p*c["fn"],(1-p)*c["fp"]]
        return v if c["defer"] is None else v+[F(c["defer"])]
    best_atom=sum(F(a["w"],W)*F(c["w"],C)*min(raw(F(a["p"]),c))
                  for a in atoms for c in contexts)
    result={}
    for cuts in combinations(range(1,len(atoms)),d["bins"]-1):
        total_loss=F(0)
        brier=F(0)
        for i,j in zip((0,)+cuts,cuts+(len(atoms),)):
            part=atoms[i:j]
            mass=sum(a["w"] for a in part)
            for c in contexts:
                action_count=2+int(c["defer"] is not None)
                best=min(sum(a["w"]*raw(F(a["p"]),c)[k] for a in part)
                         for k in range(action_count))
                total_loss+=F(c["w"],C)*best/W
            brier+=sum((F(a["w"]*b["w"],mass*W)*(F(a["p"])-F(b["p"]))**2
                        for a,b in combinations(part,2)),F(0))
        result[cuts]=(total_loss-best_atom,brier)
    return result

def assess_quantization(d):
    all_scores=partition_enumeration(d)
    decision=partition_dp(d)
    brier=partition_dp(d,"brier")
    e=min((r,b,k) for k,(r,b) in all_scores.items())
    eb=min((r,b,k) for k,(r,b) in all_scores.items() if b==min(x[1] for x in all_scores.values()))
    # Verify every candidate against the independent loss computation, not only winners.
    for cuts,(r,b) in all_scores.items():
        parts=[segment(d,i,j) for i,j in zip((0,)+cuts,cuts+(len(d["atoms"]),))]
        require((r,b)==tuple(sum((x[k] for x in parts),F(0)) for k in (0,1)),"segment oracle mismatch")
    require(decision==e and brier==eb,"partition oracle mismatch")
    def key(cuts):
        return "cuts_"+ "_".join(map(str,cuts))
    def balanced(cuts):
        bounds=(0,)+cuts+(len(d["atoms"]),)
        W=sum(a["w"] for a in d["atoms"])
        return (sum((d["bins"]*sum(a["w"] for a in d["atoms"][i:j])-W)**2
                    for i,j in zip(bounds,bounds[1:])),cuts)
    balanced_cut=min(all_scores,key=balanced)
    rstar=decision[0]
    return {"optimal_action":key(decision[2]),"optimal_regret":frac(rstar),
            "optimal_brier":frac(decision[1]),"brier_action":key(brier[2]),
            "brier_decision_gap":frac(brier[0]-rstar),
            "action_regrets":{key(k):frac(v[0]-rstar) for k,v in all_scores.items()},
            "action_metrics":{key(k):{"decision_regret":frac(v[0]),"brier_excess":frac(v[1])}
                              for k,v in all_scores.items()},
            "baseline_regrets":{"balanced_bins":frac(all_scores[balanced_cut][0]-rstar),
                               "brier_dp":frac(brier[0]-rstar)},
            "independent_partitions":len(all_scores)}

def make_quotient(r,i,shift):
    # Ordinary: homogeneous two-project horizon. Structural: heterogeneous
    # three/four-project horizon with out-of-scope transport and varied costs.
    H=r.between(3,4) if shift else 2
    t=r.between(1,2)
    D=t+r.between(1,2)
    base=project(p=F(r.between(3,7),10),cert=r.between(9,15),
                 correct=r.between(4,7),wrong=-r.between(10,15),direct=D,transport=t)
    ps=[dict(base) for _ in range(H)]
    if shift:
        for j in range(1,H):
            ps[j]=project(p=F(r.between(2,8),10),cert=r.between(9,16),correct=r.between(4,8),
                          wrong=-r.between(10,15),direct=r.between(3,5),
                          transport=r.between(1,3),scope=j!=H-1)
    mode=i%5
    if mode==0:
        states=[state(t,True,True),state(2*t,True,True)]
    elif mode==1:
        states=[state(D,True,True),state(D,False,True)]
    elif mode==2:
        cap=sum(max(p["direct"],p["transport"]) for p in ps)
        states=[state(cap,True,True),state(cap+r.between(1,4),True,True)]
    elif mode==3:
        # Equal certificate value but different available actions.
        for p in ps:
            p["transport"]=p["direct"]
        cap=sum(p["direct"] for p in ps)
        states=[state(cap,True,True),state(cap,False,True)]
    else:
        states=[state(r.between(0,8),bool(r.below(2)),bool(r.below(2))) for _ in range(2)]
    return normalize("receipt_continuation",{"projects":ps,"states":states})

def make_quantization(r,i,shift):
    # Ordinary: 6-8 atoms, 2 contexts, 2 bins. Structural: variable bin count,
    # 3-4 contexts, and an additional abstention decision in every context.
    n=r.between(7,9) if shift else r.between(6,8)
    xs=set()
    while len(xs)<n:
        xs.add(r.between(1,39))
    atoms=[{"p":frac(F(x,40)),"w":r.between(1,7)} for x in sorted(xs)]
    contexts=[]
    for j in range(r.between(3,4) if shift else 2):
        fp,fn=(r.between(5,10),r.between(1,4)) if j%2 else (r.between(1,4),r.between(5,10))
        contexts.append({"fp":fp,"fn":fn,"w":r.between(1,5),
                         "defer":frac(F(r.between(1,4),4)) if shift else None})
    return normalize("multi_context_quantization",{"atoms":atoms,"contexts":contexts,"bins":r.between(2,3) if shift else 2})

def export(out):
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    seen=set()
    splits=[]
    for name,seed,n,shift in SPLITS:
        rng=Rng([SCHEMA,seed])
        public=[]
        referee=[]
        counters=Counter()
        sums={}
        denominators=Counter()
        attempts=0
        while len(public)<n:
            attempts+=1
            require(attempts<=20*n,"duplicate rejection exhausted")
            i=len(public)
            kind="receipt_continuation" if i%2==0 else "multi_context_quantization"
            d=make_quotient(rng,i//2,shift) if i%2==0 else make_quantization(rng,i//2,shift)
            rid=content_id(kind,d)
            if rid in seen:
                continue
            seen.add(rid)
            label=assess_quotient(d) if i%2==0 else assess_quantization(d)
            p={"row_id":rid,"split":name,"ordinal":i,"mechanism":kind,"instance":d,
               "legal_actions":sorted(label["action_regrets"])}
            r={"row_id":rid,"public_sha256":digest(p),**label}
            public.append(p)
            referee.append(r)
            counters[kind]+=1
            if kind=="receipt_continuation":
                counters["equal_value_unsafe_merge"]+=int(label["equal_value_unsafe_merge"])
                counters["equivalent"]+=int(label["equivalent"])
                counters["positive_value_gap"]+=int(F(label["value_gap"])>0)
                counters["independent_traces"]+=label["independent_traces"]
            else:
                counters["positive_optimal_regret"]+=int(F(label["optimal_regret"])>0)
                counters["brier_decision_disagreement"]+=int(F(label["brier_decision_gap"])>0)
                counters["independent_partitions"]+=label["independent_partitions"]
            for k,v in label["baseline_regrets"].items():
                key=kind+":"+k
                sums[key]=sums.get(key,F(0))+F(v)
                denominators[key]+=1
        for suffix,rows in (("public",public),("referee",referee)):
            with (root/(name+"."+suffix+".jsonl")).open("xb") as f:
                for row in rows:
                    f.write(canonical(row)+b"\n")
        splits.append({"split":name,"seed":seed,"structural_shift":shift,"rows":n,
                       "counts":dict(counters),"public_sha256":digest(public),"referee_sha256":digest(referee),
                       "baseline_mean_regret":{k:frac(v/denominators[k]) for k,v in sorted(sums.items())}})
    manifest={"schema_version":SCHEMA,"source_kind":"synthetic_oracle_only","provider_outputs_admitted":False,
              "training_admission":False,"total_rows":len(seen),"splits":splits,
              "scope":"independent exact toy oracles; normalized content disjoint within registered symmetries; public development"}
    manifest["manifest_id"]=digest(manifest)
    write_json(root/"manifest.json",manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    print(json.dumps(export(parser.parse_args().out),sort_keys=True,indent=2))

if __name__=="__main__":
    main()
