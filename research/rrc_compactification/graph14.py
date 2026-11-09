"""RRC-14: finite reversible edit graph and fresh, split-safe series roster.

A graph is (left,right,post) over I,N,S,R. One edit changes exactly one
operation; its reverse is always legal. Feedback comes solely from executing
an evaluated graph on public demonstration prefixes, degree 12.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random

try:
    from . import graph11 as g, graph12 as b12, graph13 as b13
except ImportError:
    import graph11 as g
    import graph12 as b12
    import graph13 as b13

SEED=2026111014
DEGREE=g.DEGREE
HORIZON=g.HORIZON
OPS=g.OPS
TRAIN_GRAPHS=b12.TRAIN_GRAPHS
TEST_GRAPHS=b12.TEST_GRAPHS
TRAIN_N=960
DEV_N=128
COUNTS={'familiar':80,'unseen_composition':160,'source_shift':64,
        'out_of_grammar':20,'late_collision':16}
QUERY_BUDGET=12


@dataclass(frozen=True)
class Case:
    identifier:str
    family:str
    demos:tuple
    query:tuple
    future:tuple
    hidden_graph:tuple|None
    pair:str|None=None

    @property
    def visible(self):
        return tuple((f[:DEGREE+1],o[:DEGREE+1]) for f,o in self.demos)


def make(rng,graph,family,identifier,*,product=False,mutate=False,pair=None):
    source=b13._product_source if product else g.source
    demos=tuple((f:=source(rng),g.execute(f,graph)) for _ in range(2))
    if mutate:
        f,o=demos[1]
        arr=list(o);arr[7]+=rng.choice((1,2,3))
        demos=(demos[0],(f,tuple(arr)))
    query=source(rng)
    return Case(identifier,family,demos,query,g.execute(query,graph),graph,pair)


def dataset(*,include_test=True,train_count=TRAIN_N,dev_count=DEV_N):
    if train_count<1 or dev_count<1:raise ValueError('positive counts required')
    output={'train':[],'dev':[]}
    if include_test:output.update({k:[] for k in COUNTS})
    seen=set()
    def append(group,case,duplicate=False):
        if case.visible in seen and not duplicate:return False
        seen.add(case.visible);output[group].append(case);return True
    def fill(group,num,graphs,seed,*,product=False):
        rng=random.Random(seed)
        for attempt in range(num*300):
            if len(output[group])==num:break
            graph=rng.choice(graphs)
            case=make(rng,graph,group,f'{group}-{len(output[group]):04d}',product=product)
            append(group,case)
        if len(output[group])!=num:raise RuntimeError(f'cannot fill {group}')
    fill('train',train_count,TRAIN_GRAPHS,SEED+101)
    fill('dev',dev_count,TRAIN_GRAPHS,SEED+202)
    if not include_test:return output
    fill('familiar',COUNTS['familiar'],TRAIN_GRAPHS,SEED+303)
    fill('unseen_composition',COUNTS['unseen_composition'],TEST_GRAPHS,SEED+404)
    fill('source_shift',COUNTS['source_shift'],TEST_GRAPHS,SEED+505,product=True)
    rng=random.Random(SEED+606)
    attempt=0
    while len(output['out_of_grammar'])<COUNTS['out_of_grammar']:
        attempt+=1
        if attempt>30000:raise RuntimeError('negative exhaustion')
        gr=rng.choice(TEST_GRAPHS)
        c=make(rng,gr,'out_of_grammar',f'out_of_grammar-{len(output["out_of_grammar"]):04d}',mutate=True)
        if c.visible in seen:continue
        if g.search(c)['graph'] is None:append('out_of_grammar',c)
    rng=random.Random(SEED+707)
    for i in range(COUNTS['late_collision']//2):
        gr=TEST_GRAPHS[i%len(TEST_GRAPHS)]
        c=make(rng,gr,'late_collision',f'late_collision-{2*i:04d}',pair=f'collision-{i:02d}')
        if not append('late_collision',c):raise AssertionError('collision alias')
        mutated=list(c.future);mutated[DEGREE+3]+=1 if i%2 else -1
        alt=Case(f'late_collision-{2*i+1:04d}','late_collision',c.demos,c.query,
                 tuple(mutated),c.hidden_graph,c.pair)
        assert append('late_collision',alt,duplicate=True)
    return output


def digest(cases):
    return sha256(json.dumps([(c.identifier,c.family,c.demos,c.query,c.future,c.hidden_graph,c.pair)
                               for c in cases],separators=(',',':')).encode()).hexdigest()


def raw(case):
    """A flattened public 104-value observation, independent of the query future."""
    return tuple(v for row in b12.features(case) for v in row)


def neighborhood(graph):
    if graph not in g.GRAPHS:raise ValueError('invalid graph')
    result=[]
    for pos in range(3):
        for op in OPS:
            if op==graph[pos]:continue
            arr=list(graph);arr[pos]=op
            result.append((tuple(arr),pos,op))
    assert len(result)==9
    return tuple(result)


def reverse_edge(before,after):
    """Unique inverse edit exists exactly for Hamming-one neighbors."""
    return sum(x!=y for x,y in zip(before,after))==1 and len(before)==len(after)==3


def graph_onehot(graph):
    return tuple(float(op==graph[j]) for j in range(3) for op in OPS)


def feedback(case,graph):
    """One candidate query evaluates both public demos, returning residuals.

    The target future and graph label are neither accessed nor cached.
    """
    if graph not in g.GRAPHS:raise ValueError('invalid graph')
    residual=[];firsts=[];total=0;matches=0;valid=True
    for f,target in case.demos:
        pred=g.execute(g.truncate(f,DEGREE),graph)
        truth=g.truncate(target,DEGREE)
        diffs=[pred[n]-truth[n] for n in range(1,DEGREE+1)]
        first=next((n for n,v in enumerate(diffs,1) if v),DEGREE+1)
        firsts.append(first)
        total+=sum(min(abs(v),1000000) for v in diffs)
        matches+=sum(v==0 for v in diffs)
        valid &= first==DEGREE+1
        residual.extend((first/(DEGREE+1),sum(v==0 for v in diffs)/DEGREE,
                         math.log1p(sum(abs(v) for v in diffs))/12.0))
        residual.extend(b12.signed_log(v) for v in diffs[:5])
    return {'residual':tuple(residual),'accepted':bool(valid),
            'first_fault':min(firsts),'matched':matches,
            'capped_residual':total,
            'rank':(min(firsts),matches,-total)}


def action_pairs(graph):
    return tuple((new,pos,op) for new,pos,op in neighborhood(graph))


def action_features(case,graph,result):
    """Public data only; 132 = raw 104 + graph one-hot 12 + feedback 16."""
    return raw(case)+graph_onehot(graph)+result['residual']


def teacher(case,graph,cache=None):
    """Training-only privileged best-neighbor label from nine executions.

    Ties use a canonical action index. The current demonstration residual
    is free of hidden labels; teacher tests other candidate graphs off-line.
    """
    cache={} if cache is None else cache
    best=None
    for i,(candidate,_,_) in enumerate(neighborhood(graph)):
        r=cache.get(candidate)
        if r is None:
            r=feedback(case,candidate);cache[candidate]=r
        score=r['rank']
        if best is None or score>best[0]:best=(score,i,candidate)
    return best[1],best[2]


def cost(queries):
    """One query = two degree-12 graph executions, including final verification."""
    if queries<0:raise ValueError('negative queries')
    return {'graph_queries':queries,'demo_executions':2*queries,
            'primitive_calls':8*queries,'integer_loop_proxy':2*queries*3*(DEGREE+1)**2}


def verify_future(case,graph):
    if graph is None:return {'accepted':0,'future_correct':0,'false_accept':0}
    accepted=int(all(g.execute(g.truncate(f,DEGREE),graph)==g.truncate(y,DEGREE)
                     for f,y in case.demos))
    future=int(accepted and g.execute(case.query,graph)==case.future)
    return {'accepted':accepted,'future_correct':future,'false_accept':accepted-future}
