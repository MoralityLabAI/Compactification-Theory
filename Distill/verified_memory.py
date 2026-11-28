"""DISTILL-14: bounded within-root certificate retrieval, exact verification.

A learned embedding proposes candidates; only exact policy-relative equality
permits reuse. Direct quotient indexing is a required reference. All query
states are public, and support/query selection never reads referee labels.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import random
import time
import numpy as np

from .core import digest, require, write_json
from .research_program import Model
from .program_student import ACTIONS, Meter
from .program_opportunity import incumbent
from .program_cancellation import immediate, children, Auditor
from .program_quotient import signature
from .student_campaign import PRICE
from .history_memory import (CTX, EVENT, histories, replay, context,
                             root_data, verify, specs)
from .history_memory_train import RNN, arrays, WIDTH, EPOCHS, BATCH, SEEDS
from .opportunity_campaign import make_spec

CAPS=(8,32,128)
QUERY_COUNT=128
PROBES=2
EVAL_SPLITS=(("ordinary",2026113011,"shared_control"),
             ("shifted",2026113012,"shifted_control"),
             ("scarce",2026113013,"shifted_control"))
TRAIN_ARMS=(("history_pair",False,2.),("history_no_pair",False,0.),
            ("bag_pair",True,2.))


def cold_tail(m,s):
    """Metered exact solve; one fresh literal memo per query or bank entry."""
    meter=Meter(10**8)
    memo={}
    visits=0
    def tail(t):
        nonlocal visits
        if t.stage==4:
            return 0,0
        meter.charge(16)
        if t in memo:
            return memo[t]
        start=meter.used
        a=incumbent(m,t,meter)
        selection=meter.used-start
        v,r,M=immediate(m,t,a,meter)
        v-=selection*M
        for child in children(m,t,a,meter):
            x,y=tail(child)
            v+=x
            r+=y
        meter.charge(16)
        memo[t]=(v,r)
        visits+=1
        return v,r
    v,r=tail(s)
    meter.charge(64+16)
    M=m.mass(s.mask)
    return (Fraction(v,PRICE*M),Fraction(r,M)),meter.used,visits


def key_for(m,s,mode="marginal"):
    meter=Meter(100000)
    key,_,_=signature(m,s,mode,meter)
    return key,meter.used


def key_work(key):
    return 8+(len(key[-1]) if isinstance(key[-1],tuple) else 1)


def encode(model,spec,h,bag=False):
    """Unpadded encoder, same recurrence as DISTILL-13; no output heads."""
    x=np.array(context(spec),dtype=np.float32)
    p=model.p
    z=np.tanh(x@p["init"]+p["ib"])
    events=[(ACTIONS.index(a),0 if b is None else b+1) for a,b in h]
    if bag:
        events.sort()
    for a,b in events:
        z=np.tanh(z@p["rec"]+p["event"][a]+p["event"][len(ACTIONS)+b]+p["hb"])
    # Dense prior projection, sparse event rows, recurrent MAC and activations.
    work=CTX+WIDTH*(CTX+2)+len(events)*(WIDTH*WIDTH+5*WIDTH+4)
    if bag:
        work+=len(events)**2  # declared sorting allowance
    return z,work


def counts(h):
    x=np.zeros(len(ACTIONS)*3,dtype=np.int32)
    for a,b in h:
        x[3*ACTIONS.index(a)+(0 if b is None else b+1)]+=1
    return x, len(x)+2*len(h)


def train_models(n=24,epochs=EPOCHS,seeds=SEEDS,limit=48):
    train=[]
    for spec in specs("train",n):
        p,r=root_data(spec,limit)
        verify(p,r)
        train.append((digest(spec),p,r))
    models={}
    fits={}
    for name,bag,pair in TRAIN_ARMS:
        data=arrays(train,bag)
        for seed in seeds:
            rng=np.random.default_rng(seed+100)
            model=RNN(seed)
            start=time.process_time_ns()
            for ep in range(epochs):
                order=rng.permutation(len(data["trip"]))
                for i in range(0,len(order),BATCH):
                    model.train_batch(data,order[i:i+BATCH],pair)
            key=name+"-"+str(seed)
            models[key]=(model,bag)
            fits[key]={"updates":model.step,"params":sum(v.size for v in model.p.values()),
                       "cpu_ns":time.process_time_ns()-start}
    return models,fits,{rid for rid,_,_ in train}


def records(spec,models,capacity,query_count):
    m=Model(spec)
    paths=histories(m)
    # BFS order deterministic; sample literal states uniformly without class labels.
    rng=random.Random(int(digest(spec)[:16],16))
    selected=rng.sample(list(paths),min(len(paths),capacity+query_count))
    require(len(selected)>capacity,"too few query states")
    ref=Auditor(m)
    data=[]
    for s in selected:
        h=paths[s]
        require(replay(m,h)==s,"public history mismatch")
        qkey,qcost=key_for(m,s)
        lkey,lcost=key_for(m,s,"literal")
        value,cold,visits=cold_tail(m,s)
        v,r=ref.tail(s)
        M=m.mass(s.mask)
        require(value==(Fraction(v,PRICE*M),Fraction(r,M)),"cold exact mismatch")
        embeddings={k:encode(model,spec,h,bag) for k,(model,bag) in models.items()}
        embeddings["event_counts"]=counts(h)
        data.append({"stage":s.stage,"qkey":qkey,"qcost":qcost,
                     "lkey":lkey,"lcost":lcost,"value":value,"cold":cold,
                     "visits":visits,"emb":embeddings,"history":h})
    ref.close()
    return data[:capacity],data[capacity:]


def rank(bank,q,name):
    # All approximate methods use the same public stage filter and at most 2
    # verification probes. No key/value access in selection.
    work=0
    if name=="recent":
        choices=[]
        for i in range(len(bank)-1,-1,-1):
            work+=1
            if bank[i]["stage"]==q["stage"]:
                choices.append(i)
                if len(choices)==PROBES:
                    break
        return choices,work
    z,cost=q["emb"][name]
    work+=cost
    best=[]
    for i,b in enumerate(bank):
        work+=1
        if b["stage"]!=q["stage"]:
            continue
        bz,_=b["emb"][name]
        work+=3*len(z)+4+2*PROBES
        d=float(np.sum((z-bz)**2))
        best.append((d,i))
        best.sort()
        del best[PROBES:]
    return [i for _,i in best],work


def evaluate_bank(bank,queries,models):
    arms=("cold","literal_index","quotient_index","recent","event_counts",*models)
    cover=sum(any(b["qkey"]==q["qkey"] for b in bank) for q in queries)
    result={}
    for name in arms:
        cold=name=="cold"
        literal=name=="literal_index"
        direct=name in ("literal_index","quotient_index")
        key="lkey" if literal else "qkey"
        cost="lcost" if literal else "qcost"
        setup=0
        index={}
        if not cold:
            for i,b in enumerate(bank):
                setup+=b["cold"]+b[cost]+16  # solved value, key, bounded entry write
                if direct:
                    setup+=key_work(b[key])
                    index.setdefault(b[key],i)
                elif name!="recent":
                    setup+=b["emb"][name][1]
        stats=defaultdict(int)
        for q in queries:
            work=0
            found=None
            if direct:
                work+=q[cost]+key_work(q[key])
                found=index.get(q[key])
                if found is not None:
                    stats["probes"]+=1
            elif not cold:
                picks,ranking=rank(bank,q,name)
                work+=ranking
                if picks:
                    work+=q["qcost"]
                for i in picks:
                    work+=key_work(q["qkey"])
                    stats["probes"]+=1
                    if bank[i]["qkey"]==q["qkey"]:
                        found=i
                        break
                    stats["rejected_proposals"]+=1
            if found is None:
                work+=q["cold"]
                value=q["value"]
                stats["fallbacks"]+=1
                stats["fallback_visits"]+=q["visits"]
            else:
                require(bank[found][key]==q[key],"unverified retrieval")
                value=bank[found]["value"]
                work+=16 # read exact normalized value/risk
                stats["hits"]+=1
            require(value==q["value"],"incorrect reuse")
            stats["query_work"]+=work
        n=len(queries)
        result[name]={"queries":n,"coverage":cover,"hits":stats["hits"],
                      "fallbacks":stats["fallbacks"],"probes":stats["probes"],
                      "rejected_proposals":stats["rejected_proposals"],
                      "fallback_visits":stats["fallback_visits"],
                      "query_work":stats["query_work"],"setup_work":setup,
                      "total_work":setup+stats["query_work"],"incorrect_reuses":0}
    require(result["quotient_index"]["hits"]==cover,"exact index coverage mismatch")
    require(result["literal_index"]["hits"]==0,"literal support/query leak")
    return result


def summary(rows):
    keys=tuple(rows[0]["arms"])
    out={}
    for name in keys:
        arms=[r["arms"][name] for r in rows]
        base=[r["arms"]["cold"] for r in rows]
        totals={k:sum(x[k] for x in arms) for k in arms[0]}
        totals["hit_rate"]=totals["hits"]/totals["queries"]
        totals["covered_hit_rate"]=(totals["hits"]/totals["coverage"]
                                   if totals["coverage"] else None)
        totals["query_work_saved"]=1-totals["query_work"]/sum(x["query_work"] for x in base)
        totals["total_work_saved"]=1-totals["total_work"]/sum(x["total_work"] for x in base)
        totals["query_work_winning_roots"]=sum(x["query_work"]<b["query_work"] for x,b in zip(arms,base))
        totals["total_work_winning_roots"]=sum(x["total_work"]<b["total_work"] for x,b in zip(arms,base))
        out[name]=totals
    return out


def campaign(out,n=12,train_n=24,epochs=EPOCHS,seeds=SEEDS,
             limit=48,caps=CAPS,query_count=QUERY_COUNT):
    require(caps and min(caps)>0 and query_count>0,"bad memory settings")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"rows").mkdir()
    models,fits,train_ids=train_models(train_n,epochs,seeds,limit)
    seen=set(train_ids)
    rows=[]
    index=[]
    for split,seed,family in EVAL_SPLITS:
        for i in range(n):
            spec=make_spec(seed,i,family,split=="scarce")
            rid=digest(spec)
            require(rid not in seen,"split overlap")
            seen.add(rid)
            bank,queries=records(spec,models,max(caps),query_count)
            for cap in caps:
                arms=evaluate_bank(bank[:cap],queries,models)
                row={"root":rid,"spec":spec,"split":split,"capacity":cap,"arms":arms}
                name="rows/"+rid+"-"+str(cap)+".json"
                write_json(out/name,row)
                index.append({"path":name,"sha256":digest(row)})
                rows.append(row)
    report={"schema_version":"distill.verified_memory.v1",
            "source_kind":"synthetic_finite_model","training_admission":False,
            "scope":"within-root exact incumbent continuation reuse; no policy action or learned certificate",
            "bank_capacities":list(caps),"query_count":query_count,"probes":PROBES,
            "fits":fits,"train_roots":sorted(train_ids),"row_index":index,
            "summary":{split+"/"+str(cap):summary([r for r in rows if r["split"]==split and r["capacity"]==cap])
                       for split,_,_ in EVAL_SPLITS for cap in caps}}
    write_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    print(json.dumps(campaign(args.out),sort_keys=True))


if __name__=="__main__":
    main()
