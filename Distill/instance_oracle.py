"""Exact finite case grading, separate from native gym scoring."""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import json
from .core import INSTANCE_VERSION, _packet, canonical, evaluate, require, split_quiz
from .instances import compile_instance, parity

def chromatic(n,edges):
    adj=[set() for _ in range(n)]
    for u,v in edges:
        adj[u].add(v)
        adj[v].add(u)
    order=sorted(range(n),key=lambda i:(-len(adj[i]),i))
    colors=[-1]*n
    def fit(t,k):
        if t==n:
            return True
        v=order[t]
        used={colors[u] for u in adj[v]}
        for c in range(k):
            if c not in used:
                colors[v]=c
                if fit(t+1,k):
                    return True
                colors[v]=-1
        return False
    for k in range(1,n+1):
        if fit(0,k):
            return k
    raise AssertionError("finite graph must be colorable")

def chromatic_partition(n,edges):
    masks=[(1<<u)|(1<<v) for u,v in edges]
    @lru_cache(None)
    def go(s):
        if not s:
            return 0
        low=s&-s
        best=n
        t=s
        while t:
            if t&low and all((t&m)!=m for m in masks):
                best=min(best,1+go(s^t))
            t=(t-1)&s
        return best
    return go((1<<n)-1)

def unique(values):
    values=set(values)
    require(bool(values),"inconsistent constraints")
    return next(iter(values)) if len(values)==1 else "abstain"

def exact_value(case):
    """Finite reference computation for bounded generated inputs."""
    kind,d=case["kind"],case["data"]
    if kind=="equal_weight_fronts":
        n,D,c,s=(d[k] for k in ("fronts","deadline","command_ticks","switch_ticks"))
        k=max(k for k in range(n+1) if not k or k*c+(k-1)*s<=D)
        require(k==min(n,(D+s)//(c+s)),"deadline cross-check")
        return k
    if kind=="graph_observer":
        n,edges=d["vertices"],d["edges"]
        k=chromatic(n,edges)
        require(k==chromatic_partition(n,edges),"chromatic cross-check")
        return (k-1).bit_length()
    if kind=="arithmetic_trace":
        states=[d["initial"]]
        for row in d["instructions"]:
            x,o=states[-1],row["operand"]
            op=row["op"]
            if op=="add":
                y=x+o
            elif op=="subtract":
                y=x-o
            elif op=="multiply":
                y=x*o
            elif op=="if_even_add":
                y=x+o if x%2==0 else x
            else:
                raise ValueError("unknown instruction")
            states.append(y)
        return states
    if kind=="modular_trace":
        p,x=d["modulus"],d["initial"]
        states=[x]
        A,B=1,0
        for a,b in d["operations"]:
            states.append((a*states[-1]+b)%p)
            A,B=(a*A)%p,(a*B+b)%p
            require(states[-1]==(A*x+B)%p,"affine cross-check")
        return states
    if kind=="formal_reciprocal":
        f=d["coefficients"]
        g=[1]
        for k in range(1,len(f)):
            g.append(-sum(f[i]*g[k-i] for i in range(1,k+1)))
        require([sum(f[i]*g[k-i] for i in range(k+1)) for k in range(len(f))]==[1]+[0]*(len(f)-1),"convolution certificate")
        return g
    if kind=="lagrange_identity":
        u,v=d["u"],d["v"]
        cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        result=sum(x*x for x in cross)
        require(result==sum(x*x for x in u)*sum(x*x for x in v)-sum(x*y for x,y in zip(u,v))**2,"Lagrange cross-check")
        return result
    if kind=="prefix_recurrence":
        x,parents,p=d["input"],d["parents"],d["modulus"]
        n=len(x)
        depth,values=[],[]
        affected=set(d["changed"])
        for i,row in enumerate(parents):
            depth.append(1+max((depth[j] for j in row),default=0))
            values.append((x[i]+sum(values[j] for j in row))%p)
            if any(j in affected for j in row):
                affected.add(i)
        state=list(d["initial"])
        for _ in range(max(depth)):
            state=[(x[i]+sum(state[j] for j in row))%p for i,row in enumerate(parents)]
        require(state==values,"synchronous cross-check")
        closure=set(d["changed"])
        for _ in range(n):
            closure|={i for i,row in enumerate(parents) if closure.intersection(row)}
        require(closure==affected,"invalidation cross-check")
        return {"depth":max(depth),"values":values,"affected":sorted(affected)}
    if kind=="regime_counts":
        a,b=d["counts"]
        prob=Fraction(max(a,b),a+b)
        return {"action":int(b>a),"probability":f"{prob.numerator}/{prob.denominator}"}
    if kind=="anchored_parity":
        n,edges=d["vertices"],d["edges"]
        anchor,bit=d["anchor"]
        known={anchor:bit}
        for _ in range(n):
            for u,v,b in edges:
                if u in known and v in known:
                    require(known[u]^known[v]==b,"inconsistent parity")
                elif u in known:
                    known[v]=known[u]^b
                elif v in known:
                    known[u]=known[v]^b
        value=known.get(d["target"],"abstain")
        other=unique((x>>d["target"])&1 for x in range(1<<n) if ((x>>anchor)&1)==bit and all((((x>>u)^(x>>v))&1)==b for u,v,b in edges))
        require(value==other,"parity enumeration cross-check")
        return value
    if kind=="gf2_query":
        n,rows,q=d["n"],d["equations"],d["query_mask"]
        value=unique(parity(x&q) for x in range(1<<n) if all(parity(x&m)==b for m,b in rows))
        span={0:0}
        for m,b in rows:
            for s,v in list(span.items()):
                if s^m in span:
                    require(span[s^m]==v^b,"inconsistent row span")
                span[s^m]=v^b
        require(value==span.get(q,"abstain"),"row-span cross-check")
        return value
    raise ValueError("unsupported case")

def grade_instance(packet,response,*,adapter=None):
    _packet(packet)
    require("instance" in packet and any(q["id"]=="case.solve" for q in packet["questions"]),"no generated case answer")
    if adapter is None:
        from .registry import load_registry
        adapter=load_registry()[packet["gym_id"]]
    ins=packet["instance"]
    parent=compile_instance(adapter,seed=ins["seed"],ordinal=ins["ordinal"],difficulty=packet["difficulty"],profile=packet["profile"])
    require(packet==parent or any(packet==part for k in (1,2,3) for part in split_quiz(parent,max_questions=k)),"generated packet does not match reconstructed source")
    def check(case,fields):
        expected=exact_value(case)
        try:
            return canonical(json.loads(fields["answer"]))==canonical(expected)
        except (ValueError,TypeError,OverflowError):
            return False
    receipt=evaluate(packet,response,checkers={"case.solve":check})
    receipt["procedural_verifier"]={"generator":INSTANCE_VERSION,"content_sha256":ins["content_sha256"],"scope":"finite synthetic exact answer; explanation review pending"}
    return receipt
