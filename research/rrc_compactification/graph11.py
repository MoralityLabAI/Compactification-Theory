"""RRC-11: exact finite formal-series fork/join graph induction.

Graph (left,right,post) means post(left(F)*right(F)); all three nodes
share the same input. Public candidate interpreter is separate from the learner.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import random

SEED=2026111011
DEGREE=12
PREVIEW=4
HORIZON=32
OPS=('I','N','S','R')
GRAPHS=tuple(itertools.product(OPS,repeat=3))
TRAIN_GRAPHS=tuple((a,b,'I') for a in ('I','N','R') for b in ('I','N','R'))
TEST_GRAPHS=tuple(g for g in GRAPHS if g not in TRAIN_GRAPHS and ('S' in g or g[2]=='R'))


def truncate(f, degree):
    if len(f)<degree+1:raise ValueError('series too short')
    return tuple(f[:degree+1])


def mul(f,g):
    if len(f)!=len(g):raise ValueError('length mismatch')
    n=len(f)-1
    return tuple(sum(f[k]*g[i-k] for k in range(i+1)) for i in range(n+1))


def inv(f):
    if not f or f[0]!=1:raise ValueError('unit required')
    r=[1]
    for i in range(1,len(f)):
        r.append(-sum(f[k]*r[i-k] for k in range(1,i+1)))
    return tuple(r)


def unary(f,op):
    if not f or f[0]!=1:raise ValueError('unit required')
    if op=='I':return tuple(f)
    if op=='N':return tuple(v*(-1 if i%2 else 1) for i,v in enumerate(f))
    if op=='S':return tuple(f[i//2] if i%2==0 else 0 for i in range(len(f)))
    if op=='R':return inv(f)
    raise ValueError('unknown operator')


def execute(f,graph):
    if len(graph)!=3 or any(x not in OPS for x in graph):raise ValueError('invalid graph')
    return unary(mul(unary(f,graph[0]),unary(f,graph[1])),graph[2])


@dataclass(frozen=True)
class Case:
    identifier:str
    family:str
    demos:tuple
    query:tuple
    future:tuple
    graph:tuple|None
    pair:str|None=None

    @property
    def visible(self):
        return tuple((f[:DEGREE+1],g[:DEGREE+1]) for f,g in self.demos)


def source(rng):
    return (1,)+tuple(rng.randint(-2,2) for _ in range(HORIZON))


def data(*,include_test=True,train_count=160,dev_count=32):
    out={'train':[],'dev':[]}
    if include_test:out.update({'heldout_graph':[],'out_of_grammar':[],'late_collision':[]})
    known=set()
    def add(group,seed,graph,mutate=None,pair=None,duplicate=False):
        rng=random.Random(seed)
        demos=tuple((f:=source(rng),execute(f,graph)) for _ in range(2))
        if mutate=='visible':
            f,g=demos[1];z=list(g);z[7]+=1;demos=(demos[0],(f,tuple(z)))
        x=source(rng);future=execute(x,graph)
        if mutate=='future':
            z=list(future);z[DEGREE+3]+=1;future=tuple(z)
        sig=tuple((f[:DEGREE+1],g[:DEGREE+1]) for f,g in demos)
        if sig in known and not duplicate:return False
        known.add(sig)
        out[group].append(Case(f'{group}-{len(out[group]):03d}',group,demos,x,future,graph,pair))
        return True
    def fill(group,count,graphs,seed):
        rng=random.Random(seed);tries=0
        while len(out[group])<count:
            tries+=1
            if tries>count*200:raise RuntimeError('uniqueness exhausted')
            graph=graphs[rng.randrange(len(graphs))]
            add(group,seed*100000+tries,graph)
    fill('train',train_count,TRAIN_GRAPHS,SEED+1)
    fill('dev',dev_count,TRAIN_GRAPHS,SEED+2)
    if not include_test:return out
    fill('heldout_graph',80,TEST_GRAPHS,SEED+3)
    rng=random.Random(SEED+4)
    i=0
    while len(out['out_of_grammar'])<20:
        i+=1
        graph=TEST_GRAPHS[rng.randrange(len(TEST_GRAPHS))]
        before=len(out['out_of_grammar'])
        if not add('out_of_grammar',SEED*100000+i,graph,mutate='visible'):continue
        case=out['out_of_grammar'][-1]
        if search(case)['graph'] is not None:
            out['out_of_grammar'].pop()
        if i>10000:raise RuntimeError('unrepresentable case generation failed')
    for i in range(8):
        graph=TEST_GRAPHS[i*3%len(TEST_GRAPHS)]
        if not add('late_collision',SEED*1000000+i,graph,pair=f'pair-{i}'):raise AssertionError
        base=out['late_collision'][-1]
        mutated=list(base.future);mutated[DEGREE+3]+=1
        out['late_collision'].append(Case(f'late_collision-{len(out["late_collision"]):03d}','late_collision',base.demos,base.query,tuple(mutated),base.graph,base.pair))
    return out


def search(case,order=GRAPHS,*,budget=None):
    attempts=0;calls=0;work=0
    for graph in order:
        if budget is not None and attempts>=budget:break
        attempts+=1
        for f,g in case.demos:
            predicted=execute(truncate(f,DEGREE),graph)
            calls+=4;work+=3*(DEGREE+1)**2
            if predicted!=truncate(g,DEGREE):break
        else:return {'graph':graph,'attempts':attempts,'primitive_calls':calls,'work_proxy':work}
    return {'graph':None,'attempts':attempts,'primitive_calls':calls,'work_proxy':work}


def preview_score(case,graph):
    diffs=[]
    for f,g in case.demos:
        p=execute(truncate(f,PREVIEW),graph)
        diffs.extend(abs(a-b) for a,b in zip(p[1:],g[1:PREVIEW+1]))
    first=next((i+1 for i,x in enumerate(diffs) if x),len(diffs)+1)
    return (sum(x==0 for x in diffs),first,-sum(min(x,10**6) for x in diffs),-max(diffs))


def order_by_preview(case,weights=None):
    rows=[]
    for i,g in enumerate(GRAPHS):
        s=preview_score(case,g)
        if weights is None: rank=tuple(-x for x in s)+(i,)
        else:
            # Learned score includes a deterministic exact-match component as a public feature.
            rank=(-sum(w*x for w,x in zip(weights,s)),i)
        rows.append((rank,g))
    return tuple(g for _,g in sorted(rows))


def guided(case,weights=None,budget=4):
    order=order_by_preview(case,weights)
    v=search(case,order,budget=budget)
    v['preview_calls']=len(GRAPHS)*2*4
    v['preview_work_proxy']=len(GRAPHS)*2*3*(PREVIEW+1)**2
    v['work_proxy']+=v['preview_work_proxy']
    v['primitive_calls']+=v['preview_calls']
    return v


def score(case,pred):
    graph=pred['graph']
    accepted=int(graph is not None)
    correct=int(accepted and execute(case.query,graph)==case.future)
    return {'accepted':accepted,'future_correct':correct,'false_accept':accepted-correct,
            'attempts':pred['attempts'],'work_proxy':pred['work_proxy'],
            'primitive_calls':pred['primitive_calls']}


def digest(cases):
    payload=[(c.identifier,c.family,c.demos,c.query,c.future,c.graph,c.pair) for c in cases]
    return sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
