"""CG-03 generated conditional Lean claims and exact independent critique verifier."""
import hashlib
import itertools
import random
from collections import Counter

SPLITS={"train":(96,103301,(2,4),(8,16)),
        "cal":(24,103302,(2,4),(8,16)),
        "test_depth":(36,103303,(5,6),(8,16)),
        "test_clutter":(36,103304,(2,4),(30,48))}
BUDGETS=(4,8,16,32,64,128)
ALLOWED={"propext","Classical.choice","Quot.sound"}

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def satisfied(clause,values):return any(values[abs(l)-1]==(l>0) for l in clause)
def counterexample(cs,goal,values,n):
    return (type(values) is list and len(values)==n and all(type(v) is bool for v in values)
            and all(satisfied(c,values) for c in cs) and not satisfied([goal],values))

class Masks:
    def __init__(self,n):
        N=1<<n
        self.full=(1<<N)-1
        self.var=[]
        for i in range(n):
            block=1<<i
            pattern=((1<<block)-1)<<block
            self.var.append(pattern*(self.full//((1<<(2*block))-1)))
    def lit(self,l):
        z=self.var[abs(l)-1]
        return z if l>0 else self.full^z
    def clause(self,c):
        z=0
        for l in c:z|=self.lit(l)
        return z
    def cnf(self,cs):
        z=self.full
        for c in cs:z&=self.clause(c)
        return z

def generate(split,i,rng,depths,clutter):
    d=rng.randint(*depths);n=2*d+3
    ids=list(range(1,n+1));rng.shuffle(ids)
    p=ids[:d+1];aux=ids[d+1:2*d+1]
    core=[(p[0],)]
    for j in range(d):
        core.extend([(-p[j],p[j+1],aux[j]),(-p[j],p[j+1],-aux[j])])
    gaps=rng.sample(range(d),2)
    removed=[1+2*g+rng.randrange(2) for g in gaps]
    def plant(k=None):
        values=[bool(rng.getrandbits(1)) for _ in range(n)]
        if k is None:
            for v in p:values[v-1]=True
        else:
            g=(k-1)//2
            for j,v in enumerate(p):values[v-1]=j<=g
            values[aux[g]-1]=bool((k-1)%2)
        return values
    full=plant();ws=[plant(k) for k in removed]
    assert all(satisfied(c,full) for c in core)
    for k,w in zip(removed,ws):
        assert not satisfied(core[k],w)
        assert all(satisfied(c,w) for j,c in enumerate(core) if j!=k)
        assert not w[p[-1]-1]
    canon=lambda c:tuple(sorted(c,key=abs))
    decoys=[];seen={canon(c) for c in core}
    target=rng.randint(*clutter)
    while len(decoys)<target:
        c=canon(tuple(v if rng.random()<.5 else -v for v in rng.sample(ids,3)))
        if c not in seen and all(satisfied(c,w) for w in [full,*ws]):
            seen.add(c);decoys.append(c)
    clauses=core+decoys
    order=list(range(len(clauses)));rng.shuffle(order)
    inverse={j:k for k,j in enumerate(order)}
    clauses=[list(clauses[j]) for j in order]
    core_ids=[inverse[j] for j in range(len(core))]
    decoy_ids=[inverse[j] for j in range(len(core),len(clauses))]
    rng.shuffle(decoy_ids)
    ns=f"CG{SPLITS[split][1]}_{i}"
    arms=[]
    for arm,k in enumerate([None,None,*removed]):
        omit=[] if arm==0 else rng.sample(decoy_ids,rng.randint(2,min(6,len(decoys))))
        if k is not None:omit.append(inverse[k])
        kept=sorted(set(range(len(clauses)))-set(omit))
        arms.append({"id":digest(ns+str(arm))[:20],"kept":kept,"omitted":sorted(omit),
                     "bad":k is not None,"ref_repair":None if k is None else inverse[k],
                     "ref_witness":None if k is None else ws[arm-2]})
    rng.shuffle(arms)
    f={"id":ns,"split":split,"n":n,"clauses":clauses,"goal":p[-1],"p":p,"aux":aux,
       "core":core_ids,"source_witness":full,"arms":arms}
    m=Masks(n)
    assert m.cnf(clauses)!=0 and not (m.cnf(clauses)&m.lit(-f["goal"]))
    for a in arms:
        cs=[clauses[j] for j in a["kept"]]
        targetmask=m.cnf(cs)&m.lit(-f["goal"])
        assert bool(targetmask)==a["bad"]
        if a["bad"]:
            assert counterexample(cs,f["goal"],a["ref_witness"],n)
            assert not targetmask&m.clause(clauses[a["ref_repair"]])
    return f

def families():
    for split,(count,seed,depths,clutter) in SPLITS.items():
        rng=random.Random(seed)
        for i in range(count):yield generate(split,i,rng,depths,clutter)

def clause_prop(c,values=None):
    parts=[]
    for l in c:
        p=f"x{abs(l)}" if values is None else ("True" if values[abs(l)-1] else "False")
        parts.append(p if l>0 else f"¬ {p}")
    return " ∨ ".join(parts)

STEP="""theorem cgStep (p q a : Prop) (h1 : ¬ p ∨ q ∨ a) (h2 : ¬ p ∨ q ∨ ¬ a) (hp : p) : q :=
  h1.elim (fun hn => False.elim (hn hp)) (fun h =>
    h.elim id (fun ha => h2.elim (fun hn => False.elim (hn hp))
      (fun h' => h'.elim id (fun hna => False.elim (hna ha)))))
"""

def proof_source(f,name,kept):
    core=f["core"]
    assert set(core)<=set(kept)
    variables=" ".join(f"x{i}" for i in range(1,f["n"]+1))
    hs=" ".join(f"(h{j} : {clause_prop(f['clauses'][j])})" for j in kept)
    lines=[f"theorem {name} ({variables} : Prop) {hs} : x{f['goal']} := by",
           f"  have hp0 : x{f['p'][0]} := h{core[0]}"]
    for j,a in enumerate(f["aux"]):
        lines.append(f"  have hp{j+1} : x{f['p'][j+1]} := cgStep _ _ x{a} h{core[1+2*j]} h{core[2+2*j]} hp{j}")
    lines.append(f"  exact hp{len(f['aux'])}")
    return "\n".join(lines)

def concrete(cs,values,goal=None):
    props=["("+clause_prop(c,values)+")" for c in cs]
    if goal is not None:props.append("¬ "+("True" if values[goal-1] else "False"))
    # Preserve all concrete clauses, but keep instance-search depth logarithmic.
    def balanced(xs):
        if not xs: return "True"
        if len(xs)==1: return xs[0]
        j=len(xs)//2
        return "(" + balanced(xs[:j]) + " ∧ " + balanced(xs[j:]) + ")"
    return balanced(props)

def concrete_proof(cs, values, goal=None):
    """Construct every true clause explicitly, then mirror concrete's balanced tree."""
    def literal(l):
        assert values[abs(l)-1] == (l>0)
        return "True.intro" if l>0 else "(fun h : False => h)"
    def disjunction(c):
        assert satisfied(c,values)
        if len(c)==1:
            return literal(c[0])
        if values[abs(c[0])-1] == (c[0]>0):
            return "(Or.inl " + literal(c[0]) + ")"
        return "(Or.inr " + disjunction(c[1:]) + ")"
    proofs=[disjunction(c) for c in cs]
    if goal is not None:
        assert not values[goal-1]
        proofs.append("(fun h : False => h)")
    def balanced(xs):
        if not xs: return "True.intro"
        if len(xs)==1: return xs[0]
        j=len(xs)//2
        return "⟨" + balanced(xs[:j]) + ", " + balanced(xs[j:]) + "⟩"
    return balanced(proofs)

def native_source(f):
    lines=["namespace "+f["id"]]
    expected=[]
    def add(name,body):
        lines.append(body);lines.append("#print axioms "+name)
        expected.append(f["id"]+"."+name)
    all_ids=list(range(len(f["clauses"])))
    add("submitted",proof_source(f,"submitted",all_ids))
    add("nonvacuous",f"theorem nonvacuous : {concrete(f['clauses'],f['source_witness'])} := {concrete_proof(f['clauses'],f['source_witness'])}")
    for j,a in enumerate(f["arms"]):
        name=f"claim{j}"
        if a["bad"]:
            cs=[f["clauses"][i] for i in a["kept"]]
            add(name,f"theorem {name} : {concrete(cs,a['ref_witness'],f['goal'])} := {concrete_proof(cs,a['ref_witness'],f['goal'])}")
            name=f"repair{j}"
            add(name,proof_source(f,name,sorted(a["kept"]+[a["ref_repair"]])))
        else:add(name,proof_source(f,name,a["kept"]))
    lines.append("end "+f["id"])
    return "\n".join(lines)+"\n",expected

# Independent verifier deliberately has its own scanning and propagation loop.
def verify_tree(cs,n,tree):
    def visit(t,assn):
        while True:
            forced=None
            for clause in cs:
                undecided=[];sat=False
                for l in clause:
                    if abs(l) not in assn:undecided.append(l)
                    elif assn[abs(l)]==(l>0):sat=True;break
                if sat:continue
                if not undecided:return t=={"conflict":True}
                if len(undecided)==1:
                    forced=undecided[0];break
            if forced is None:break
            assn[abs(forced)]=forced>0
        if type(t) is not dict or set(t)!={"var","false","true"}:return False
        v=t["var"]
        if type(v) is not int or v<1 or v>n or v in assn:return False
        return visit(t["false"],dict(assn,**{})|{v:False}) and visit(t["true"],dict(assn,**{})|{v:True})
    return visit(tree,{})

def verify(f,a,cert):
    cs=[f["clauses"][j] for j in a["kept"]]
    goal=f["goal"];n=f["n"]
    v=cert.get("verdict")
    if v=="ABSTAIN":return {"verdict":False,"repair":False,"complete":False}
    if v=="SUPPORTED":
        ok=verify_tree(cs+[[-goal]],n,cert.get("proof"))
        return {"verdict":ok,"repair":False,"complete":ok}
    if v=="REFUTED":
        ok=counterexample(cs,goal,cert.get("witness"),n)
        j=cert.get("repair_clause")
        repair=(ok and type(j) is int and j in a["omitted"] and
                verify_tree(cs+[f["clauses"][j],[-goal]],n,cert.get("repair_proof")))
        return {"verdict":ok,"repair":repair,"complete":ok and repair}
    return {"verdict":False,"repair":False,"complete":False}
