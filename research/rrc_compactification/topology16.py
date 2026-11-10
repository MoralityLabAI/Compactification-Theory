"""RRC-16: multiple training wirings and a held-out Hadamard-series join.

Every attempted candidate is a complete three-token graph on two public
formal-series demonstrations. Source graph/future are grading-only after
inference. All arithmetic is exact integer arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random

try:
    from . import graph11 as g, graph13 as b13, graph14 as b14, topology15 as b15
except ImportError:
    import graph11 as g
    import graph13 as b13
    import graph14 as b14
    import topology15 as b15

SEED = 2026111016
PREFIX = g.DEGREE
HORIZON = g.HORIZON
BUDGET = 12
TRAIN_TOPOLOGIES = ('fork', 'cascade', 'additive')
HELDOUT_TOPOLOGY = 'hadamard'
TOPOLOGIES = TRAIN_TOPOLOGIES + (HELDOUT_TOPOLOGY,)
TRAIN_PER_TOPOLOGY = 120
DEV_PER_TOPOLOGY = 30
COUNTS = {'seen_wirings': 60, 'new_join_signed': 80, 'new_join_products': 40,
          'new_join_negative': 12, 'new_join_collision': 12}


@dataclass(frozen=True)
class Case:
    identifier: str
    family: str
    topology: str
    demos: tuple
    query: tuple
    future: tuple
    hidden_graph: tuple | None
    pair: str | None = None

    @property
    def visible(self):
        return (self.topology, tuple((f[:PREFIX + 1], y[:PREFIX + 1]) for f, y in self.demos))


def unit_hadamard(left, right):
    """Coefficientwise Hadamard product in 1 + q Z[[q]]."""
    if len(left) != len(right) or not left or left[0] != 1 or right[0] != 1:
        raise ValueError('normalized unit series of equal length required')
    return tuple(x * y for x, y in zip(left, right))


def execute(f, graph, topology):
    if topology not in TOPOLOGIES or graph not in g.GRAPHS:
        raise ValueError('bad graph or topology')
    if topology in TRAIN_TOPOLOGIES:
        return b15.execute(f, graph, topology)
    a, b, c = graph
    left = g.unary(f, a)
    right = g.unary(f, b)
    return g.unary(unit_hadamard(left, right), c)


def topology_descriptor(topology):
    """Public compositional topology information, shared by all learned arms.

    Coordinates: has cross-degree binary coupling; join is nonlinear; right
    operand receives left's output. The held-out wiring combines descriptor
    properties encountered separately in the training topologies.
    """
    if topology not in TOPOLOGIES: raise ValueError('bad topology')
    return (float(topology in ('fork','cascade')),
            float(topology in ('fork','cascade','hadamard')),
            float(topology == 'cascade'))


def graph_cost(graph, topology, degree=PREFIX):
    if graph not in g.GRAPHS or topology not in TOPOLOGIES or degree < 0:
        raise ValueError('invalid arithmetic work request')
    if topology != 'hadamard': return b15.graph_cost(graph, topology, degree)
    triangle = degree * (degree + 1) // 2
    unary = lambda op: triangle if op == 'R' else degree + 1
    return sum(unary(op) for op in graph) + degree + 1


def feedback(case, graph):
    """Exactly one charged candidate query, evaluating both public demos."""
    if graph not in g.GRAPHS: raise ValueError('bad candidate')
    features=[]; firsts=[]; matched=0; cap=0
    for f, target in case.demos:
        candidate=execute(f[:PREFIX+1],graph,case.topology)
        differences=[candidate[n]-target[n] for n in range(1,PREFIX+1)]
        first=next((n for n,d in enumerate(differences,1) if d),PREFIX+1)
        firsts.append(first)
        matched+=sum(d==0 for d in differences)
        cap+=sum(min(abs(d),1_000_000) for d in differences)
        features.extend((first/(PREFIX+1),sum(d==0 for d in differences)/PREFIX,
                         math.log1p(sum(abs(d) for d in differences))/12))
        features.extend(b14.b12.signed_log(d) for d in differences[:5])
    return {'residual':tuple(features),'accepted':all(x==PREFIX+1 for x in firsts),
            'first_fault':min(firsts),'matched':matched,'capped_residual':cap,
            'rank':(min(firsts),matched,-cap)}


def matching_graphs(case, topology=None):
    topology=case.topology if topology is None else topology
    return tuple(graph for graph in g.GRAPHS if all(
        execute(f[:PREFIX+1],graph,topology)==y[:PREFIX+1] for f,y in case.demos))


def certificate(case, graph):
    if graph is None:return {'accepted':0,'future_correct':0,'false_accept':0}
    accepted=int(all(execute(f[:PREFIX+1],graph,case.topology)==y[:PREFIX+1]
                     for f,y in case.demos))
    correct=int(bool(accepted) and execute(case.query,graph,case.topology)==case.future)
    return {'accepted':accepted,'future_correct':correct,'false_accept':accepted-correct}


def features(case, graph, signal):
    """132 public observed graph features + 3 published topology features."""
    return tuple(b14.action_features(case,graph,signal)) + topology_descriptor(case.topology)


def _source(rng, product=False):
    if not product:return g.source(rng)
    for _ in range(500):
        f=b13._product_source(rng)
        if f[1] or f[2]: return f
    raise RuntimeError('unhelpful product source')


def _case(rng, graph, topology, family, ident, *, product=False, mutation=False, pair=None):
    source=lambda: _source(rng, product)
    demos=tuple((f:=source(),execute(f,graph,topology)) for _ in range(2))
    if mutation:
        f,y=demos[1]
        bad=list(y);bad[7]+=rng.choice((-3,-2,-1,1,2,3))
        demos=(demos[0],(f,tuple(bad)))
    query=source()
    return Case(ident,family,topology,demos,query,execute(query,graph,topology),graph,pair)


def roster(*, include_test=True, train_per=TRAIN_PER_TOPOLOGY, dev_per=DEV_PER_TOPOLOGY):
    if train_per < 1 or dev_per < 1:raise ValueError('positive train/dev sizes')
    groups={'train':[], 'dev':[]}
    if include_test:groups.update({name:[] for name in COUNTS})
    observed=set()
    def append(split,item,allow_duplicate=False):
        if item.visible in observed and not allow_duplicate:return False
        observed.add(item.visible);groups[split].append(item);return True
    for split,amount,offset in (('train',train_per,100),('dev',dev_per,200)):
        rng=random.Random(SEED+offset)
        desired={topology:amount for topology in TRAIN_TOPOLOGIES}
        for i in range(40*amount*len(TRAIN_TOPOLOGIES)):
            if not any(desired.values()):break
            topology=TRAIN_TOPOLOGIES[i%len(TRAIN_TOPOLOGIES)]
            if desired[topology]==0:continue
            graph=rng.choice(g.GRAPHS)
            item=_case(rng,graph,topology,split,f'{split}-{len(groups[split]):04d}')
            if append(split,item):desired[topology]-=1
        if any(desired.values()):raise RuntimeError('cannot fill '+split)
    if not include_test:return groups
    rng=random.Random(SEED+303)
    per={t:COUNTS['seen_wirings']//3 for t in TRAIN_TOPOLOGIES}
    for _ in range(COUNTS['seen_wirings']*200):
        if not any(per.values()):break
        topology=rng.choice([t for t,n in per.items() if n])
        item=_case(rng,rng.choice(g.GRAPHS),topology,'seen_wirings',f'seen_wirings-{len(groups["seen_wirings"]):04d}')
        if append('seen_wirings',item):per[topology]-=1
    if any(per.values()):raise RuntimeError('seen topology fill failed')
    for split,count,seed,product in (('new_join_signed',COUNTS['new_join_signed'],404,False),
                                     ('new_join_products',COUNTS['new_join_products'],505,True)):
        rng=random.Random(SEED+seed)
        for _ in range(count*1000):
            if len(groups[split])>=count:break
            item=_case(rng,rng.choice(g.GRAPHS),'hadamard',split,f'{split}-{len(groups[split]):04d}',product=product)
            if item.visible in observed:continue
            # Eligibility depends solely on observed coefficients; this
            # discards simple aliases in ANY familiar topology.
            if any(matching_graphs(item,old) for old in TRAIN_TOPOLOGIES):continue
            if not matching_graphs(item):raise AssertionError('source graph unrepresentable')
            append(split,item)
        if len(groups[split])!=count:raise RuntimeError('new join sampling exhausted '+split)
    rng=random.Random(SEED+606)
    for _ in range(10000):
        if len(groups['new_join_negative'])==COUNTS['new_join_negative']:break
        item=_case(rng,rng.choice(g.GRAPHS),'hadamard','new_join_negative',
                   f'new_join_negative-{len(groups["new_join_negative"]):04d}',mutation=True)
        if item.visible in observed or matching_graphs(item):continue
        append('new_join_negative',item)
    if len(groups['new_join_negative'])!=COUNTS['new_join_negative']:
        raise RuntimeError('negative sampling exhausted')
    rng=random.Random(SEED+707)
    for i in range(COUNTS['new_join_collision']//2):
        for _ in range(10000):
            base=_case(rng,rng.choice(g.GRAPHS),'hadamard','new_join_collision',
                f'new_join_collision-{2*i:04d}',pair=f'pair-{i}')
            if any(matching_graphs(base,old) for old in TRAIN_TOPOLOGIES):continue
            if append('new_join_collision',base):break
        else:raise RuntimeError('collision sampling exhausted')
        future=list(base.future)
        future[PREFIX+3]+=1 if i%2 else -1
        duplicate=Case(f'new_join_collision-{2*i+1:04d}',base.family,base.topology,
                       base.demos,base.query,tuple(future),None,base.pair)
        assert append('new_join_collision',duplicate,allow_duplicate=True)
    return groups


def digest(cases):
    return sha256(json.dumps([(c.identifier,c.family,c.topology,c.demos,c.query,c.future,
                               c.hidden_graph,c.pair) for c in cases],separators=(',',':')).encode()).hexdigest()
