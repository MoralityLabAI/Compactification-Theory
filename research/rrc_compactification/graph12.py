"""RRC-12: fresh, disjoint graph-composition benchmark, no test oracle in learner.

The training program list excludes every algebraic equivalence class whose
canonical graph contains both S (degree doubling) and R (formal reciprocal).
Every primitive appears in the training syntax, including each output slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random

try:
    from . import graph11 as g
except ImportError:
    import graph11 as g

SEED = 2026111012
DEGREE = g.DEGREE
HORIZON = g.HORIZON
OPS = g.OPS
TRAIN_N = 2040
DEV_N = 320
TEST_COUNTS = {'familiar':68,'unseen_composition':160,
               'source_shift':64,'out_of_grammar':20,'late_collision':16}


def canonical_graphs():
    """Group syntax by exact outputs on eight fixed generic formal series.

    The grouping is a finite design convention. Its equivalences are checked
    again on fresh independent inputs, with the algebraic forms documented.
    """
    rng = random.Random(100932)
    probes = [g.source(rng)[:DEGREE+1] for _ in range(8)]
    representations = {}
    for graph in g.GRAPHS:
        signature = tuple(g.execute(f,graph) for f in probes)
        representations.setdefault(signature, graph)
    return tuple(representations.values())


CANONICAL = canonical_graphs()
TRAIN_GRAPHS = tuple(x for x in CANONICAL if not ('S' in x and 'R' in x))
TEST_GRAPHS = tuple(x for x in CANONICAL if 'S' in x and 'R' in x)
assert len(CANONICAL)==25 and len(TRAIN_GRAPHS)==17 and len(TEST_GRAPHS)==8
assert all(x not in TRAIN_GRAPHS for x in TEST_GRAPHS)


@dataclass(frozen=True)
class Case:
    identifier: str
    family: str
    demos: tuple
    query: tuple
    future: tuple
    hidden_graph: tuple | None
    pair: str | None = None

    @property
    def visible(self):
        return tuple((f[:DEGREE+1],o[:DEGREE+1]) for f,o in self.demos)


def _product_source(rng):
    """Test-only product generating functions, with exact integer DP."""
    modulus = rng.choice((5,7,9,11))
    allowed = {i for i in range(modulus) if rng.random()<0.45}
    if not allowed:allowed.add(1)
    coeff = [1] + [0]*HORIZON
    for p in range(1,HORIZON+1):
        if p%modulus in allowed:
            for n in range(p,HORIZON+1):coeff[n]+=coeff[n-p]
    return tuple(coeff)


def _make(rng,graph,family,identifier,*,product=False,mutation=None,pair=None):
    src = _product_source if product else g.source
    demos = tuple((f:=src(rng),g.execute(f,graph)) for _ in range(2))
    if mutation=='visible':
        f,o = demos[1]
        arr=list(o);arr[7]+=rng.choice((1,2,3))
        demos=(demos[0],(f,tuple(arr)))
    query=src(rng)
    future=g.execute(query,graph)
    return Case(identifier,family,demos,query,future,graph,pair)


def dataset(*,include_test=True,train_count=TRAIN_N,dev_count=DEV_N):
    if train_count<1 or dev_count<1:raise ValueError('positive training and development sizes required')
    splits = {'train':[], 'dev':[]}
    if include_test:splits.update({k:[] for k in TEST_COUNTS})
    seen=set()

    def append(split,case,*,duplicate=False):
        sig=case.visible
        if sig in seen and not duplicate:return False
        seen.add(sig)
        splits[split].append(case)
        return True

    def fill(split,n,graphs,seed,*,product=False):
        rng=random.Random(seed)
        for attempts in range(n*300):
            if len(splits[split])>=n:break
            pg=graphs[rng.randrange(len(graphs))]
            candidate=_make(rng,pg,split,f'{split}-{len(splits[split]):04d}',product=product)
            append(split,candidate)
        if len(splits[split])!=n:raise AssertionError(f'{split} roster exhausted')

    fill('train',train_count,TRAIN_GRAPHS,SEED+100)
    fill('dev',dev_count,TRAIN_GRAPHS,SEED+200)
    if not include_test:return splits
    fill('familiar',TEST_COUNTS['familiar'],TRAIN_GRAPHS,SEED+300)
    fill('unseen_composition',TEST_COUNTS['unseen_composition'],TEST_GRAPHS,SEED+400)
    fill('source_shift',TEST_COUNTS['source_shift'],TEST_GRAPHS,SEED+500,product=True)

    rng=random.Random(SEED+600)
    tries=0
    while len(splits['out_of_grammar'])<TEST_COUNTS['out_of_grammar']:
        tries+=1
        if tries>20000:raise RuntimeError('cannot produce negatives')
        graph=rng.choice(TEST_GRAPHS)
        case=_make(rng,graph,'out_of_grammar',f'out_of_grammar-{len(splits["out_of_grammar"]):04d}',mutation='visible')
        if case.visible in seen:continue
        # Independent 64-syntax bounded negative witness, no model access.
        if g.search(case)['graph'] is None:append('out_of_grammar',case)

    rng=random.Random(SEED+700)
    for i in range(8):
        candidate=_make(rng,TEST_GRAPHS[i], 'late_collision', f'late_collision-{2*i:04d}',pair=f'collision-{i}')
        if not append('late_collision',candidate):raise AssertionError('collision source duplicate')
        future=list(candidate.future)
        future[DEGREE+3]+=1 if i%2==0 else -1
        second=Case(f'late_collision-{2*i+1:04d}', 'late_collision', candidate.demos,
                    candidate.query, tuple(future), candidate.hidden_graph,candidate.pair)
        assert append('late_collision',second,duplicate=True)
    return splits


def roster_digest(cases):
    data=[(c.identifier,c.family,c.demos,c.query,c.future,c.hidden_graph,c.pair) for c in cases]
    return sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()


def signed_log(v):
    return math.copysign(min(30.0,math.log1p(abs(v)))/10.0,v) if v else 0.0


def features(case):
    """Two raw observed input/output series + public degree and demo index.

    No candidate graph execution, inferred derivative, or hidden label enters.
    """
    result=[]
    for j,(f,out) in enumerate(case.demos):
        for n in range(DEGREE+1):
            result.append((signed_log(f[n]),signed_log(out[n]),n/DEGREE,float(j)))
    return result


def verify(case,graph):
    if graph is None:return {'accepted':0,'correct_future':0,'false_accept':0}
    if graph not in g.GRAPHS:raise ValueError('bad graph proposal')
    accepted=int(all(g.execute(g.truncate(f,DEGREE),graph)==g.truncate(o,DEGREE)
                     for f,o in case.demos))
    correct=int(accepted and g.execute(case.query,graph)==case.future)
    return {'accepted':accepted,'correct_future':correct,'false_accept':accepted-correct}
