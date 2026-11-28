"""DISTILL-15: lazy, bounded, exact research continuation memoization.

No precomputed bank and no learned proposals. A persistent cache is populated
only by work performed in the current query stream. Each query has the same
literal local memo as the independent cold baseline. Persistent keys are scoped
to one Model and its frozen incumbent; normalized integer values preserve mass.
"""
import argparse
from collections import OrderedDict, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import random
import time

from .core import digest, require, write_json
from .research_program import Model
from .program_student import Meter
from .program_opportunity import incumbent
from .program_cancellation import immediate, children, Auditor
from .program_quotient import signature, exact_div
from .student_campaign import PRICE
from .history_memory import histories
from .opportunity_campaign import make_spec
from .program_oracle import actual_transition

CAPS=(8,32,128)
QUERY_COUNT=128
SPLITS=(("ordinary",2026113021,"shared_control"),
        ("shifted",2026113022,"shifted_control"),
        ("scarce",2026113023,"shifted_control"))
STREAMS=("distinct","rollout")
ARMS=("cold","root_literal","root_quotient","recursive_literal",
      "recursive_quotient","recursive_hybrid","selective_hybrid")
STATS=("requests","root_hits","recursive_hits","different_state_hits",
       "different_mask_hits","local_hits","misses","writes","evictions",
       "skipped_writes","expansions","key_work","admin_work","work")


def key_charge(key):
    return 8+(len(key[-1]) if isinstance(key[-1],tuple) else 1)


class Memo:
    def __init__(self,model,arm,capacity):
        require(arm in ARMS,"invalid memo arm")
        require(type(capacity) is int and capacity>0,"invalid capacity")
        self.m=model
        self.arm=arm
        self.capacity=capacity
        self.cache=OrderedDict()
        self.stats=defaultdict(int)
        self.peak=0
        self.local_peak=0

    def mode(self,s,top):
        if self.arm=="cold" or (self.arm.startswith("root_") and not top):
            return None
        if self.arm.endswith("literal"):
            return "literal"
        if self.arm.endswith("quotient"):
            return "marginal"
        # Frozen hybrid: avoid expensive early-stage marginal projections.
        return "literal" if s.stage<2 else "marginal"

    def query(self,s):
        require(s.stage<4,"terminal query")
        m=self.m
        meter=Meter(10**8)
        local={}
        self.stats["requests"]+=1

        def admin(n):
            meter.charge(n)
            self.stats["admin_work"]+=n

        def tail(t,top=False):
            if t.stage==4:
                return 0,0
            meter.charge(16)
            if t in local:
                self.stats["local_hits"]+=1
                return local[t]
            mode=self.mode(t,top)
            key=None
            before_key=meter.used
            if mode is not None:
                key,g,N=signature(m,t,mode,meter)
                # Tags preserve mode identity even for hybrid variants.
                key=(mode,key)
                meter.charge(key_charge(key[1]))
                self.stats["key_work"]+=meter.used-before_key
                key_cost=meter.used-before_key
                if key in self.cache:
                    admin(16)
                    cv,cr,source=self.cache[key]
                    self.cache.move_to_end(key)
                    pair=(cv*g,cr*g)
                    self.stats["root_hits" if top else "recursive_hits"]+=1
                    self.stats["different_state_hits"]+=int(source!=t)
                    self.stats["different_mask_hits"]+=int(source.mask!=t.mask)
                    # Keep common local memo semantics for reused subproblems.
                    meter.charge(16)
                    local[t]=pair
                    self.local_peak=max(self.local_peak,len(local))
                    return pair
                self.stats["misses"]+=1

            solve_start=meter.used
            a=incumbent(m,t,meter)
            selection=meter.used-solve_start
            v,r,M=immediate(m,t,a,meter)
            v-=selection*M
            for child in children(m,t,a,meter):
                x,y=tail(child)
                v+=x
                r+=y
            self.stats["expansions"]+=1
            meter.charge(16)
            local[t]=(v,r)
            self.local_peak=max(self.local_peak,len(local))
            if key is not None:
                spent=meter.used-solve_start
                # The selective arm writes only expensive-enough completed
                # computations. The gate uses realized metered solve work.
                if self.arm=="selective_hybrid":
                    admin(4)
                    admit=spent>=2*key_cost+32
                else:
                    admit=True
                if admit:
                    admin(32+key_charge(key[1]))
                    cv,cr=exact_div(v,g),exact_div(r,g)
                    if key in self.cache:
                        # A descendant can share a key only if its exact
                        # normalized value agrees; never silently overwrite.
                        old=self.cache[key]
                        require((cv,cr)==old[:2],"inconsistent cache value")
                        self.cache.move_to_end(key)
                    else:
                        if len(self.cache)>=self.capacity:
                            admin(8)
                            self.cache.popitem(last=False)
                            self.stats["evictions"]+=1
                        self.cache[key]=(cv,cr,t)
                        self.stats["writes"]+=1
                    self.peak=max(self.peak,len(self.cache))
                    require(self.peak<=self.capacity,"capacity exceeded")
                else:
                    self.stats["skipped_writes"]+=1
            return v,r

        v,r=tail(s,True)
        meter.charge(64+16)
        M=m.mass(s.mask)
        self.stats["work"]+=meter.used
        return v,r,M

    def metrics(self):
        return {**{k:self.stats[k] for k in STATS},
                "persistent_peak":self.peak,"local_peak":self.local_peak}


def streams(m,count,seed):
    require(type(count) is int and count>0,"invalid query count")
    rng=random.Random(seed)
    # Distinct off-policy public states, with no class/utility sampling.
    pool=[s for s in histories(m) if 1<=s.stage<4]
    require(len(pool)>=count,"not enough distinct states")
    distinct=rng.sample(pool,count)
    # Separate repeated trace workload. The referee draws a world only to
    # generate legal public histories under the incumbent. No world enters
    # the cache or the continuation solver. Exclude the trivial initial stage.
    rollout=[]
    weights=[w.mass for w in m.worlds]
    while len(rollout)<count:
        i=rng.choices(range(len(weights)),weights=weights,k=1)[0]
        s=m.initial()
        while s.stage<4:
            if s.stage>=1:
                rollout.append(s)
                if len(rollout)==count:
                    break
            a=incumbent(m,s,Meter(100000))
            _,s,_=actual_transition(m,s,a,i)
    return {"distinct":distinct,"rollout":rollout}


def run_arm(m,queries,arm,capacity,reference):
    mem=Memo(m,arm,capacity)
    work_trace=[]
    start=time.process_time_ns()
    for i,s in enumerate(queries):
        before=mem.stats["work"]
        v,r,M=mem.query(s)
        require((v,r,M)==reference[i],"incorrect continuation reuse")
        work_trace.append(mem.stats["work"]-before)
    elapsed=time.process_time_ns()-start
    n=len(queries)
    require(mem.stats["requests"]==n,"query accounting")
    return {**mem.metrics(),"incorrect_reuses":0,"cpu_ns":elapsed,
            "prefix_work":{str(k):sum(work_trace[:k])
                           for k in sorted({1,min(16,n),min(64,n),n})}}


def root_result(spec,count,caps):
    m=Model(spec)
    qs=streams(m,count,int(digest(spec)[:16],16))
    audit=Auditor(m)
    rows=[]
    for name,queries in qs.items():
        reference=[(*audit.tail(s),m.mass(s.mask)) for s in queries]
        # The exact oracle is a referee, not a cache entry source.
        state_count=len(set(queries))
        classes=len({signature(m,s,"marginal",Meter(100000))[0] for s in queries})
        for cap in caps:
            arms={a:run_arm(m,queries,a,cap,reference) for a in ARMS}
            rows.append({"stream":name,"capacity":cap,"states":state_count,
                         "classes":classes,"queries":len(queries),"arms":arms})
    audit.close()
    return rows


def summarize(rows):
    base=sum(r["arms"]["cold"]["work"] for r in rows)
    out={}
    for a in ARMS:
        rr=[r["arms"][a] for r in rows]
        totals={k:sum(x[k] for x in rr) for k in (*STATS,"incorrect_reuses","cpu_ns")}
        totals.update(work_saved=1-totals["work"]/base,
                      winning_roots=sum(r["arms"][a]["work"]<r["arms"]["cold"]["work"] for r in rows),
                      tied_roots=sum(r["arms"][a]["work"]==r["arms"]["cold"]["work"] for r in rows),
                      root_hit_rate=totals["root_hits"]/totals["requests"],
                      persistent_peak=max(x["persistent_peak"] for x in rr),
                      local_peak=max(x["local_peak"] for x in rr),
                      prefix_work={k:sum(x["prefix_work"][k] for x in rr)
                                   for k in rr[0]["prefix_work"]})
        out[a]=totals
    return {"roots":len(rows),"states":sum(r["states"] for r in rows),
            "classes":sum(r["classes"] for r in rows),"arms":out}


def campaign(out,n=12,count=QUERY_COUNT,caps=CAPS):
    require(n>0 and count>0 and caps and min(caps)>0,"invalid campaign")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"rows").mkdir()
    rows=[]
    index=[]
    seen=set()
    for split,seed,family in SPLITS:
        for i in range(n):
            spec=make_spec(seed,i,family,split=="scarce")
            rid=digest(spec)
            require(rid not in seen,"root overlap")
            seen.add(rid)
            for r in root_result(spec,count,caps):
                row={**r,"root":rid,"split":split,"spec":spec}
                path="rows/"+rid+"-"+r["stream"]+"-"+str(r["capacity"])+".json"
                write_json(out/path,row)
                index.append({"path":path,"sha256":digest(row)})
                rows.append(row)
    summary={split+"/"+stream+"/"+str(cap):summarize([
                r for r in rows if r["split"]==split and r["stream"]==stream and r["capacity"]==cap])
             for split,_,_ in SPLITS for stream in STREAMS for cap in caps}
    report={"schema_version":"distill.lazy_certificate_memory.v1",
            "source_kind":"synthetic_finite_model","training_admission":False,
            "scope":"within-root, frozen incumbent exact continuation workload",
            "caps":list(caps),"queries_per_stream":count,"row_index":index,
            "summary":summary}
    write_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    report=campaign(args.out)
    print(json.dumps({k:v for k,v in report.items() if k!="row_index"},sort_keys=True))


if __name__=="__main__":
    main()
