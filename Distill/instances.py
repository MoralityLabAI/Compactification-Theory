"""Deterministic public mathematical exercises, independent of model execution."""
from __future__ import annotations
import hashlib
from .core import (CASE, EXPLAIN, DIFFICULTIES, INSTANCE_VERSION, _packet,
                   canonical, compile_quiz, digest, require)

FAMILIES = {
    "IRC-PG-01": ("equal_weight_fronts", "The first front is selected. Each distinct front needs one command; moving to the next costs one switch. Find the greatest number completed by the deadline, including equality. Give one integer."),
    "COE-01": ("graph_observer", "Find the minimum ordinary binary observer width for the supplied undirected graph. Each edge requires distinct endpoint codewords. Give one integer and show a lower bound and feasible code."),
    "RLT-ROLLOUT": ("arithmetic_trace", "Execute instructions in order. if_even_add adds its operand iff the current value is even. Give a JSON list including the initial state and every intermediate state."),
    "AUX-GYM": ("modular_trace", "Apply x <- (a*x+b) mod modulus for each [a,b] in operations. Give a JSON list including the initial state and each least nonnegative residue."),
    "ARG-01-ORACLE": ("formal_reciprocal", "For F(q) given by coefficients with constant term one, compute 1/F(q) modulo q^(len(coefficients)). Give the JSON list of integer coefficients in increasing degree."),
    "BSAR-02": ("lagrange_identity", "For the two supplied integer vectors in R^3, compute squared norm of the cross product. Give one integer and verify the dot-product Lagrange identity. Explain the scope of this pointwise identity."),
    "MOLT-HYBRID-01": ("prefix_recurrence", "Each node updates synchronously to (input[i]+sum of previous states of its parents) mod modulus. Compute the structural longest-node-chain bound, topological fixed point, and structural invalidation set including changed nodes. Give JSON object with depth, values and affected (sorted indices)."),
    "ASM-01": ("regime_counts", "The observation cell has counts for regimes 0 and 1. Action 0 succeeds exactly in regime 0, action 1 exactly in regime 1. Choose maximal expected success, ties choose action 0. Give JSON object with action and probability (reduced fraction string including denominator)."),
    "BROKEN-VERIFIER": ("anchored_parity", "All edges are exact parity constraints [u,v,b] meaning z[u] XOR z[v]=b; anchor [vertex,bit] is exact. Infer target if determined. Give JSON 0 or 1, or JSON string \"abstain\" if ambiguous."),
    "HG-07": ("gf2_query", "Unknown n-bit vector x satisfies equations [mask,bit] where bit=popcount(mask AND x) mod 2. Infer parity of query_mask if uniquely determined. Give JSON 0 or 1, or JSON string \"abstain\". Explain row span or rank."),
}
LEVELS = {"foundation":0,"standard":1,"frontier":2}

class Rng:
    def __init__(self,key):
        self.key=canonical(key)
        self.counter=0
    def below(self,n):
        require(type(n) is int and 1<=n<=2**64,"invalid range")
        cap=2**256-(2**256%n)
        while True:
            x=int.from_bytes(hashlib.sha256(self.key+self.counter.to_bytes(8,"big")).digest(),"big")
            self.counter+=1
            if x<cap:
                return x%n
    def between(self,a,b):
        return a+self.below(b-a+1)
    def choice(self,items):
        return items[self.below(len(items))]

def parity(mask):
    return mask.bit_count()&1

def make_case(gym_id,seed,ordinal,difficulty):
    require(gym_id in FAMILIES and difficulty in LEVELS,"unknown instance family or difficulty")
    require(type(seed) is int and 0<=seed<2**63 and type(ordinal) is int and 0<=ordinal<10**7,"invalid seed or ordinal")
    r=Rng([INSTANCE_VERSION,gym_id,seed,ordinal,difficulty])
    l=LEVELS[difficulty]
    kind=FAMILIES[gym_id][0]
    if kind=="equal_weight_fronts":
        d={"fronts":r.between(3+l,6+2*l),"deadline":r.between(4,18+16*l),"command_ticks":r.between(1,5+l),"switch_ticks":r.between(0,5+l)}
    elif kind=="graph_observer":
        n=4+l+r.below(2)
        density=r.between(2,8)
        edges=[[i,j] for i in range(n) for j in range(i+1,n) if r.below(10)<density]
        d={"vertices":n,"edges":edges or [[0,1]]}
    elif kind=="arithmetic_trace":
        ops=("add","subtract","multiply","if_even_add")
        d={"initial":r.between(-12,12),"instructions":[{"op":r.choice(ops),"operand":r.between(-4,4)} for _ in range(3+2*l+r.below(3))]}
    elif kind=="modular_trace":
        p=r.choice((5,7,11,13,17))
        d={"modulus":p,"initial":r.below(p),"operations":[[r.below(p),r.below(p)] for _ in range(2+2*l+r.below(3))]}
    elif kind=="formal_reciprocal":
        d={"coefficients":[1]+[r.between(-3-l,3+l) for _ in range(3+2*l+r.below(2))]}
    elif kind=="lagrange_identity":
        d={"u":[r.between(-4-l,4+l) for _ in range(3)],"v":[r.between(-4-l,4+l) for _ in range(3)]}
    elif kind=="prefix_recurrence":
        n=4+2*l+r.below(2)
        p=r.choice((5,7,11,17))
        d={"input":[r.below(p) for _ in range(n)],"modulus":p,"parents":[[j for j in range(i) if r.below(3)==0] for i in range(n)],"initial":[r.below(p) for _ in range(n)],"changed":[r.below(n)]}
    elif kind=="regime_counts":
        d={"counts":[r.between(1,30+70*l),r.between(1,30+70*l)]}
    elif kind=="anchored_parity":
        n=5+2*l
        bits=[r.below(2) for _ in range(n)]
        edges=[[i,j,bits[i]^bits[j]] for i in range(n) for j in range(i+1,n) if r.below(10)<(6 if j==i+1 else 1)]
        a=r.below(n)
        d={"vertices":n,"edges":edges,"anchor":[a,bits[a]],"target":(a+1+r.below(n-1))%n}
    elif kind=="gf2_query":
        n=5+2*l
        x=r.below(1<<n)
        rows=[]
        for _ in range(2+l+r.below(2)):
            m=r.between(1,(1<<n)-1)
            if m not in rows:
                rows.append(m)
        span={0}
        for m in rows:
            span|={s^m for s in span}
        q=r.choice(sorted(span-{0})) if r.below(2)==0 else r.choice([i for i in range(1,1<<n) if i not in span])
        d={"n":n,"equations":[[m,parity(m&x)] for m in rows],"query_mask":q}
    else:
        raise ValueError("unknown kind")
    content=digest({"kind":kind,"data":d})
    return {"classification":"public_development","case_id":"proc-v1-"+content,"kind":kind,"data":d},content

def compile_instance(adapter,*,seed,ordinal,difficulty="standard",profile="full"):
    case,h=make_case(adapter["gym_id"],seed,ordinal,difficulty)
    packet=compile_quiz(adapter,difficulty=difficulty,profile=profile)
    packet.pop("packet_id")
    packet["public_case"]=case
    packet["instance"]={"generator":INSTANCE_VERSION,"seed":seed,"ordinal":ordinal,"content_sha256":h}
    packet["questions"]=[
        {"id":"case.solve","category":"worked_case","prompt":FAMILIES[adapter["gym_id"]][1]+" Put only the requested JSON value in answer; give the derivation and verification in the other fields.","fields":list(CASE)},
        {"id":"case.mechanism","category":"mechanism","prompt":"Derive the general algorithm or identity used here. Explain its invariants and why it terminates or applies.","fields":list(EXPLAIN)},
        {"id":"case.boundary","category":"evidence","prompt":"Modify one premise to create a different outcome, ambiguity or failure of the method. Give a small checkable witness and describe the scope of the original claim.","fields":list(EXPLAIN)},
    ]
    packet["packet_id"]=digest(packet)
    _packet(packet)
    return packet

def generate_instances(adapter,*,seed,count,difficulties=("foundation","standard","frontier"),profile="full"):
    require(type(count) is int and 0<=count<=512,"invalid count")
    require(difficulties and len(set(difficulties))==len(difficulties) and set(difficulties)<=DIFFICULTIES.keys(),"invalid difficulties")
    seen=set()
    example=adapter["example_case"]
    if example is not None:
        seen.add(digest({"kind":example["kind"],"data":example["data"]}))
    result=[]
    ordinal=0
    while len(result)<count:
        require(ordinal<max(100,count*128),"unique case generation exhausted")
        p=compile_instance(adapter,seed=seed,ordinal=ordinal,difficulty=difficulties[len(result)%len(difficulties)],profile=profile)
        ordinal+=1
        h=p["instance"]["content_sha256"]
        if h not in seen:
            seen.add(h)
            result.append(p)
    return result
