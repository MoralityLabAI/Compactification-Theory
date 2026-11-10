"""RRC-21: fresh visible-series rosters and known-only split calibration.

The source generator and graph compiler are public. Training and calibration
cannot reach the held-out source constructors or future-scoring data.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import random

try:
    from . import family19 as f19, family20 as f20, graph11 as g, topology16 as top
except ImportError:
    import family19 as f19
    import family20 as f20
    import graph11 as g
    import topology16 as top

SEED = 2026111021
KNOWN = ('signed', 'product', 'sparse')
NOVEL = ('rational', 'squared', 'twisted_product', 'cubic_sparse', 'lacunary_q3', 'amplified_signed')
PREFIX = top.PREFIX
HORIZON = top.HORIZON
BUDGET = 12
REFERENCE_PER_FAMILY = 140
CALIBRATION_PER_FAMILY = 60
TEST_KNOWN_PER_FAMILY = 64
TEST_NOVEL_PER_FAMILY = 48
NEGATIVE_N = 16
COLLISION_PAIRS = 6


def source(rng: random.Random, family: str):
    """Exact, integral, normalized formal series; no graph labels inside source."""
    if family in KNOWN or family in ('rational', 'squared'):
        return f19.source(rng, family)
    if family in ('twisted_product','cubic_sparse'):
        return f20.source(rng,family)
    if family == 'lacunary_q3':
        a = f19.source(rng,'signed')
        return tuple(a[n//3] if n%3==0 else 0 for n in range(HORIZON+1))
    if family == 'amplified_signed':
        return (1,) + tuple(rng.choice((-18,-12,-7,7,12,18)) for _ in range(HORIZON))
    raise ValueError('unknown series source family')


def make(rng, graph, topology, family, index, *, mutation=False, pair=None):
    demos = tuple((x := source(rng, family), top.execute(x,graph,topology)) for _ in range(2))
    if mutation:
        f,y=demos[1]; y=list(y); y[7]+=rng.choice((-3,-2,-1,1,2,3))
        demos = (demos[0], (f,tuple(y)))
    x=source(rng,family)
    return top.Case(f'rrc21-{family}-{index:05d}',family,topology,demos,x,
                    top.execute(x,graph,topology),graph,pair)


def dataset(*, include_test=True, reference_per=REFERENCE_PER_FAMILY,
            calibration_per=CALIBRATION_PER_FAMILY,
            known_per=TEST_KNOWN_PER_FAMILY, novel_per=TEST_NOVEL_PER_FAMILY,
            negatives=NEGATIVE_N, pairs=COLLISION_PAIRS):
    if min(reference_per,calibration_per,known_per,novel_per,negatives,pairs)<1:
        raise ValueError('strictly positive roster sizes required')
    result={'reference':[],'calibration':[]}
    if include_test:
        result.update({s:[] for s in KNOWN+NOVEL})
        result['negative']=[];result['collision']=[]
    seen=set()
    def append(key,c,duplicate=False):
        if c.visible in seen and not duplicate:return False
        seen.add(c.visible); result[key].append(c);return True
    def draw(split,family,count,offset):
        rng=random.Random(SEED+offset)
        tries=0
        while len([c for c in result[split] if c.family==family])<count:
            tries+=1
            if tries>count*700:raise RuntimeError('failed to generate distinct cases: '+split+' '+family)
            graph=rng.choice(g.GRAPHS);topology=rng.choice(top.TOPOLOGIES)
            c=make(rng,graph,topology,family,tries)
            append(split,c)
    for j,family in enumerate(KNOWN):
        draw('reference',family,reference_per,1000+101*j)
        draw('calibration',family,calibration_per,2000+101*j)
    if not include_test:return result
    for j,family in enumerate(KNOWN):draw(family,family,known_per,3000+111*j)
    for j,family in enumerate(NOVEL):draw(family,family,novel_per,4000+123*j)
    rng=random.Random(SEED+9500)
    tries=0
    while len(result['negative'])<negatives:
        tries+=1
        if tries>30000:raise RuntimeError('negative roster exhausted')
        family=rng.choice(KNOWN+NOVEL)
        c=make(rng,rng.choice(g.GRAPHS),rng.choice(top.TOPOLOGIES),family,tries,mutation=True)
        if c.visible in seen or top.matching_graphs(c):continue
        append('negative',c)
    rng=random.Random(SEED+9600)
    for i in range(pairs):
        for tries in range(2000):
            family=NOVEL[i%len(NOVEL)]
            c=make(rng,rng.choice(g.GRAPHS),rng.choice(top.TOPOLOGIES),family,2*i,pair=f'collision-{i}')
            if append('collision',c):break
        else:raise RuntimeError('collision roster exhausted')
        future=list(c.future);future[PREFIX+3]+=(-1 if i%2 else 1)
        other=replace(c,identifier=f'rrc21-collision-{2*i+1:05d}',future=tuple(future),hidden_graph=None)
        assert append('collision',other,duplicate=True)
    return result


def digest(cases):
    payload=[(c.identifier,c.family,c.topology,c.demos,c.query,c.future,c.hidden_graph,c.pair) for c in cases]
    return sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
