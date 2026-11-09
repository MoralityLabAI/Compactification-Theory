"""RRC-13: fresh graph roster and bounded partial-program feedback.

One candidate partial graph is executed after selecting each of the first two
nodes. Feedback is computed exclusively against the PUBLIC observed demos.
The hidden source graph, query future and full search do not enter a policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random

try:
    from . import graph11 as g, graph12 as b
except ImportError:
    import graph11 as g
    import graph12 as b

SEED=2026111013
TRAIN_N=1200
DEV_N=160
TEST_COUNTS={'familiar':64,'unseen_composition':120,'source_shift':48,
             'out_of_grammar':16,'late_collision':16}
TRAIN_GRAPHS=b.TRAIN_GRAPHS
TEST_GRAPHS=b.TEST_GRAPHS
OPS=g.OPS
DEGREE=g.DEGREE
HORIZON=g.HORIZON
FEEDBACK_SIZE=16


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


def _product_source(rng):
    """Test-only family: unrestricted arithmetic part sets."""
    modulus=rng.choice((5,7,9,11,13))
    mask={i for i in range(modulus) if rng.random()<.5}
    if not mask:mask={1}
    f=[1]+[0]*HORIZON
    for k in range(1,HORIZON+1):
        if k%modulus in mask:
            for n in range(k,HORIZON+1):f[n]+=f[n-k]
    return tuple(f)


def _make(rng,graph,family,ident,*,product=False,mutation=False,pair=None):
    source=_product_source if product else g.source
    demos=tuple((f:=source(rng),g.execute(f,graph)) for _ in range(2))
    if mutation:
        x,y=demos[1];edited=list(y);edited[7]+=rng.choice((1,2,3))
        demos=(demos[0],(x,tuple(edited)))
    query=source(rng)
    return Case(ident,family,demos,query,g.execute(query,graph),graph,pair)


def dataset(*,include_test=True,train_count=TRAIN_N,dev_count=DEV_N):
    if train_count<1 or dev_count<1:raise ValueError('positive train/dev required')
    groups={'train':[],'dev':[]}
    if include_test:groups.update({k:[] for k in TEST_COUNTS})
    seen=set()

    def append(name,c,*,duplicate=False):
        if c.visible in seen and not duplicate:return False
        seen.add(c.visible);groups[name].append(c)
        return True

    def fill(name,num,graphs,seed,*,product=False):
        rng=random.Random(seed)
        tries=0
        while len(groups[name])<num:
            tries+=1
            if tries>300*num:raise RuntimeError('source uniqueness exhausted: '+name)
            gr=rng.choice(graphs)
            append(name,_make(rng,gr,name,f'{name}-{len(groups[name]):04d}',product=product))

    fill('train',train_count,TRAIN_GRAPHS,SEED+101)
    fill('dev',dev_count,TRAIN_GRAPHS,SEED+202)
    if not include_test:return groups
    fill('familiar',TEST_COUNTS['familiar'],TRAIN_GRAPHS,SEED+303)
    fill('unseen_composition',TEST_COUNTS['unseen_composition'],TEST_GRAPHS,SEED+404)
    fill('source_shift',TEST_COUNTS['source_shift'],TEST_GRAPHS,SEED+505,product=True)
    rng=random.Random(SEED+606)
    attempts=0
    while len(groups['out_of_grammar'])<TEST_COUNTS['out_of_grammar']:
        attempts+=1
        if attempts>20000:raise RuntimeError('negative exhaustion')
        gr=rng.choice(TEST_GRAPHS)
        c=_make(rng,gr,'out_of_grammar',f'out_of_grammar-{len(groups["out_of_grammar"]):04d}',mutation=True)
        if c.visible in seen:continue
        if g.search(c)['graph'] is None:append('out_of_grammar',c)
    rng=random.Random(SEED+707)
    for i in range(TEST_COUNTS['late_collision']//2):
        gr=TEST_GRAPHS[i%len(TEST_GRAPHS)]
        base=_make(rng,gr,'late_collision',f'late_collision-{2*i:04d}',pair=f'collision-{i:02d}')
        if not append('late_collision',base):raise AssertionError('collision duplication')
        mutated=list(base.future);mutated[DEGREE+3]+=1 if i%2 else -1
        alt=Case(f'late_collision-{2*i+1:04d}','late_collision',base.demos,
                 base.query,tuple(mutated),base.hidden_graph,base.pair)
        assert append('late_collision',alt,duplicate=True)
    return groups


def digest(cases):
    return sha256(json.dumps([(c.identifier,c.family,c.demos,c.query,c.future,
                                c.hidden_graph,c.pair) for c in cases],
                             separators=(',',':')).encode()).hexdigest()


def raw_features(case):
    """104 fixed public features, independent of proposed graph and future."""
    return [x for row in b.features(case) for x in row]


def residual_feedback(case,left,right=None):
    """One exact partial-program query on BOTH observed demonstrations.

    Returns 16 normalized observed residual coordinates. A partial query is
    graph (left,I,I), or (left,right,I), with a known neutral suffix.
    No matching/selection over other candidate operations occurs here.
    """
    if left not in OPS or (right is not None and right not in OPS):raise ValueError('bad operation')
    partial=(left,right or 'I','I')
    out=[]
    for f,target in case.demos:
        proposed=g.execute(g.truncate(f,DEGREE),partial)
        expected=g.truncate(target,DEGREE)
        d=[proposed[n]-expected[n] for n in range(1,DEGREE+1)]
        first=next((n for n,v in enumerate(d,1) if v),DEGREE+1)
        out.extend((first/(DEGREE+1),sum(v==0 for v in d)/DEGREE,
                    math.log1p(sum(abs(x) for x in d))/12))
        out.extend(b.signed_log(v) for v in d[:5])
    if len(out)!=FEEDBACK_SIZE:raise AssertionError('feedback size')
    return tuple(out)


def greedy_construct(case):
    """Untrained stagewise active comparator: four candidate previews per node.

    Uses public target and interpreter; 4*3*2 partial demonstrations, then
    one independent full certificate check. Costs are reported separately.
    """
    chosen=[]
    for j in range(3):
        best=None
        for candidate in OPS:
            words=tuple(chosen+[candidate]+['I']*(2-j))
            error=0
            for f,target in case.demos:
                got=g.execute(g.truncate(f,DEGREE),words)
                expected=g.truncate(target,DEGREE)
                # Lexicographic: correct prefix length, then capped L1 error.
                diffs=[abs(got[n]-expected[n]) for n in range(1,DEGREE+1)]
                first=next((n for n,v in enumerate(diffs,1) if v),DEGREE+1)
                error += 100*(DEGREE+1-first)+sum(min(d,1000) for d in diffs)
            row=(error,OPS.index(candidate),candidate)
            if best is None or row<best:best=row
        chosen.append(best[-1])
    return tuple(chosen)


def policy_work(partial_queries:int=0,full_verifications:int=1):
    """Nominal RRC-11 cost convention; every candidate has 2 demonstrations.

    One partial query executes one graph on both demonstrations. Each graph
    execution charges four primitive calls and 3*(T+1)^2 loop units.
    """
    executions=2*(partial_queries+full_verifications)
    return {'partial_queries':partial_queries,'full_verifications':full_verifications,
            'primitive_calls':4*executions,
            'integer_loop_proxy':executions*3*(DEGREE+1)**2}


def certificate(case,graph):
    """Independent exact full-prefix verifier; future used for scoring only."""
    r=b.verify(case,graph)
    return {**r,'proposed_graph':list(graph)}
