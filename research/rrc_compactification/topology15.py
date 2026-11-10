"""RRC-15 frozen-controller transfer to NEW algebraic graph wiring.

Topology and the public source/target demonstrations are observable. Graph
identities and withheld future outputs are excluded from policy inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random

try:
    from . import graph11 as old, graph13 as prior, graph14 as edit, graph12 as raw12
except ImportError:
    import graph11 as old
    import graph13 as prior
    import graph14 as edit
    import graph12 as raw12

SEED = 2026111015
DEGREE = old.DEGREE
HORIZON = old.HORIZON
OPS = old.OPS
BUDGET = 12
TOPOLOGIES = ('fork', 'cascade', 'additive')
COUNTS = {
    'fork_anchor': 32,
    'cascade_novel': 48,
    'cascade_alias': 24,
    'cascade_product': 24,
    'additive_novel': 48,
    'additive_product': 24,
    'cascade_negative': 10,
    'additive_negative': 10,
    'cascade_collision': 12,
    'additive_collision': 12,
}
SPLIT_TOPOLOGY = {name: ('fork' if name.startswith('fork') else
                         'cascade' if name.startswith('cascade') else 'additive')
                  for name in COUNTS}


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
        return (self.topology, tuple((f[:DEGREE+1], y[:DEGREE+1]) for f, y in self.demos))


def unit_add(f, h):
    if len(f) != len(h) or f[0] != 1 or h[0] != 1:
        raise ValueError('unit series of equal length required')
    return tuple([1] + [f[n] + h[n] for n in range(1,len(f))])


def execute(f, graph, topology):
    """Three operations and one binary join; same six-bit graph syntax."""
    if topology not in TOPOLOGIES or graph not in old.GRAPHS:
        raise ValueError('unknown topology or graph')
    a, b, c = graph
    left = old.unary(f, a)
    right = old.unary(left if topology == 'cascade' else f, b)
    joint = unit_add(left, right) if topology == 'additive' else old.mul(left, right)
    return old.unary(joint, c)


def graph_cost(graph, topology, degree=DEGREE):
    """Nominal exact coefficient work, independent of source/output values."""
    if graph not in old.GRAPHS or topology not in TOPOLOGIES or degree < 0:
        raise ValueError('invalid work request')
    triangular = degree * (degree+1)//2
    def unary(op):
        return triangular if op == 'R' else degree+1
    join = degree+1 if topology == 'additive' else (degree+1)*(degree+2)//2
    return unary(graph[0]) + unary(graph[1]) + unary(graph[2]) + join


def matching_graphs(case, topology=None):
    """Complete finite exact candidate search, used for dataset audit and baseline."""
    kind = case.topology if topology is None else topology
    return tuple(gr for gr in old.GRAPHS if all(
        execute(f[:DEGREE+1], gr, kind) == y[:DEGREE+1]
        for f,y in case.demos))


def feedback(case, graph):
    """One charged graph query on both degree-12 public demonstrations."""
    residual=[]; firsts=[]; total=0; matches=0
    for f,y in case.demos:
        pred=execute(f[:DEGREE+1],graph,case.topology)
        diffs=[pred[n]-y[n] for n in range(1,DEGREE+1)]
        first=next((n for n,v in enumerate(diffs,1) if v), DEGREE+1)
        firsts.append(first)
        total += sum(min(abs(v),1000000) for v in diffs)
        matches += sum(v==0 for v in diffs)
        residual.extend((first/(DEGREE+1),sum(v==0 for v in diffs)/DEGREE,
                         math.log1p(sum(abs(v) for v in diffs))/12.0))
        residual.extend(raw12.signed_log(v) for v in diffs[:5])
    return {'residual':tuple(residual),'accepted':all(k==DEGREE+1 for k in firsts),
            'first_fault':min(firsts),'matched':matches,
            'capped_residual':total,'rank':(min(firsts),matches,-total)}


def certify_future(case, graph):
    if graph is None:
        return {'accepted':0,'correct_future':0,'false_accept':0}
    accepted=int(all(execute(f[:DEGREE+1],graph,case.topology)==y[:DEGREE+1]
                     for f,y in case.demos))
    future=int(bool(accepted) and execute(case.query,graph,case.topology)==case.future)
    return {'accepted':accepted,'correct_future':future,'false_accept':accepted-future}


def source(rng,product):
    if not product: return old.source(rng)
    # Require at least one small available part to expose a useful prefix.
    for _ in range(200):
        f=prior._product_source(rng)
        if f[1] != 0 or f[2] != 0: return f
    raise RuntimeError('failed to sample nontrivial part source')


def make(rng,graph,topology,family,identifier,*,product=False,pair=None,mutated=False):
    demos=tuple((f:=source(rng,product),execute(f,graph,topology)) for _ in range(2))
    if mutated:
        f,y=demos[1]
        z=list(y);z[7]+=rng.choice((-3,-2,-1,1,2,3))
        demos=(demos[0],(f,tuple(z)))
    q=source(rng,product)
    return Case(identifier,family,topology,demos,q,execute(q,graph,topology),graph,pair)


def roster():
    """Fresh fixed seed; no model, policy score or future grade used to select cases.

    Novel strata are selected ONLY by absence of an old-fork program that
    matches the public observed demonstrations. The alias stratum is kept.
    """
    groups={name:[] for name in COUNTS};seen=set()
    def add(family,c,allow_pair=False):
        if c.visible in seen and not allow_pair:return False
        seen.add(c.visible);groups[family].append(c);return True
    seeds={'fork_anchor':101,'cascade_novel':202,'cascade_alias':303,
           'cascade_product':404,'additive_novel':505,'additive_product':606,
           'cascade_negative':707,'additive_negative':808,
           'cascade_collision':909,'additive_collision':1010}
    for family in COUNTS:
        topology=SPLIT_TOPOLOGY[family]
        rng=random.Random(SEED+seeds[family]);tries=0
        if family.endswith('collision'):
            for pair_id in range(COUNTS[family]//2):
                for _ in range(5000):
                    gr=rng.choice(old.GRAPHS)
                    c=make(rng,gr,topology,family,f'{family}-{len(groups[family]):03d}',pair=f'{family}-{pair_id}')
                    if add(family,c):break
                else:raise RuntimeError('collision source exhaustion')
                future=list(c.future);future[DEGREE+3]+=1 if pair_id%2 else -1
                alternate=Case(f'{family}-{len(groups[family]):03d}',family,topology,
                               c.demos,c.query,tuple(future),None,c.pair)
                assert add(family,alternate,allow_pair=True)
            continue
        while len(groups[family])<COUNTS[family]:
            tries+=1
            if tries>120000:raise RuntimeError('roster exhaustion '+family)
            gr=rng.choice(old.GRAPHS)
            c=make(rng,gr,topology,family,f'{family}-{len(groups[family]):03d}',
                   product='product' in family,mutated=family.endswith('negative'))
            if c.visible in seen:continue
            if family.endswith('negative'):
                if matching_graphs(c):continue
            else:
                old_alias=bool(matching_graphs(c,'fork'))
                if family=='cascade_novel' and old_alias:continue
                if family=='cascade_alias' and not old_alias:continue
                if family in ('cascade_product','additive_product','additive_novel') and old_alias:continue
                if not matching_graphs(c):raise AssertionError('generated program not in new language')
            add(family,c)
    return groups


def digest(cases):
    values=[(c.identifier,c.family,c.topology,c.demos,c.query,c.future,c.hidden_graph,c.pair) for c in cases]
    return sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
