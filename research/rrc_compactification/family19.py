"""RRC-19: independent source families for balanced quotient-class learning.

Training families: iid signed units, arithmetic-part partition products,
sparse signed units. Held out: reciprocal sparse polynomials and convolution
squares. Only observed two-demo prefixes and topology go to students.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import json
import random

try:
    from . import graph11 as g, topology16 as t, learn18 as l
except ImportError:
    import graph11 as g
    import topology16 as t
    import learn18 as l

SEED=2026111019
TRAIN_FAMILIES=('signed','product','sparse')
OOD_FAMILIES=('rational','squared')
ALL_FAMILIES=TRAIN_FAMILIES+OOD_FAMILIES
TRAIN_PER_CELL=300
DEV_PER_CELL=35
TEST_PER_CELL=35
NEGATIVE_N=20
COLLISION_PAIRS=8
PREFIX=t.PREFIX
HORIZON=t.HORIZON
BUDGET=12


def source(rng:random.Random, family:str):
    if family=='signed':return g.source(rng)
    if family=='product':return t._source(rng, product=True)
    if family=='sparse':
        for _ in range(100):
            f=(1,)+tuple(rng.choice((-2,-1,1,2)) if rng.random()<.42 else 0 for _ in range(HORIZON))
            if sum(bool(z) for z in f[1:9])>=2:return f
        raise RuntimeError('sparse source exhausted')
    if family=='rational':
        p=[1]+[0]*HORIZON
        for k in range(1,9):
            if k<=2 or rng.random()<.42:p[k]=rng.choice((-2,-1,1,2))
        return g.inv(tuple(p))
    if family=='squared':
        return g.mul(source(rng,'sparse'),source(rng,'sparse'))
    raise ValueError('unknown source family')


def case(rng,graph,topology,split,family,index,*,mutation=False,pair=None):
    demo=[]
    for _ in range(2):
        f=source(rng,family)
        demo.append((f,t.execute(f,graph,topology)))
    if mutation:
        f,y=demo[1];z=list(y);z[7]+=rng.choice((-3,-2,-1,1,2,3))
        demo[1]=(f,tuple(z))
    x=source(rng,family)
    return t.Case(f'{split}-{topology}-{family}-{index:05d}',split,topology,
                  tuple(demo),x,t.execute(x,graph,topology),graph,pair)


def roster(*,include_test=True,train_per=TRAIN_PER_CELL,dev_per=DEV_PER_CELL,test_per=TEST_PER_CELL):
    """One generated roster, no OOD/negative/collision function in train-only path.

    Each train cell has the same count. Test cells use independent seeds and
    visible-prefix deduplication. Paired collisions are explicit exceptions.
    """
    if min(train_per,dev_per,test_per)<1:raise ValueError('positive cell sizes')
    output={'train':[],'dev':[]}
    if include_test:
        output.update({family+'_test':[] for family in ALL_FAMILIES})
        output['negative']=[];output['collision']=[]
    observed=set()
    def append(split,item,duplicate=False):
        if item.visible in observed and not duplicate:return False
        observed.add(item.visible);output[split].append(item);return True
    def fill(split,family,topology,count,seed):
        rng=random.Random(seed)
        current=0
        attempts=0
        while current<count:
            attempts+=1
            if attempts>1000*count:raise RuntimeError('source/case uniqueness exhausted '+split+' '+family+' '+topology)
            graph=rng.choice(g.GRAPHS)
            item=case(rng,graph,topology,split,family,attempts)
            if append(split,item):current+=1
    for stage,cell_count,offset in (('train',train_per,101),('dev',dev_per,202)):
        for j,family in enumerate(TRAIN_FAMILIES):
            for k,topology in enumerate(t.TOPOLOGIES):
                fill(stage,family,topology,cell_count,SEED+offset+j*1109+k*37)
    if not include_test:return output
    for j,family in enumerate(ALL_FAMILIES):
        for k,topology in enumerate(t.TOPOLOGIES):
            fill(family+'_test',family,topology,test_per,SEED+303+j*1907+k*71)
    rng=random.Random(SEED+8080)
    attempts=0
    while len(output['negative'])<NEGATIVE_N:
        attempts+=1
        if attempts>40000:raise RuntimeError('negative roster exhausted')
        topology=rng.choice(t.TOPOLOGIES);family=rng.choice(ALL_FAMILIES)
        sample=case(rng,rng.choice(g.GRAPHS),topology,'negative',family,attempts,mutation=True)
        if sample.visible in observed or t.matching_graphs(sample):continue
        append('negative',sample)
    rng=random.Random(SEED+9090)
    for i in range(COLLISION_PAIRS):
        for attempt in range(5000):
            family=ALL_FAMILIES[i%len(ALL_FAMILIES)]
            topology=t.TOPOLOGIES[i%len(t.TOPOLOGIES)]
            sample=case(rng,rng.choice(g.GRAPHS),topology,'collision',family,2*i,pair=f'pair-{i}')
            if append('collision',sample):break
        else:raise RuntimeError('collision source exhausted')
        y=list(sample.future);y[PREFIX+3]+=(-1 if i%2 else 1)
        twin=replace(sample,identifier=sample.identifier+'-mutant',future=tuple(y),hidden_graph=None)
        assert append('collision',twin,duplicate=True)
    return output


def source_family(case):
    """For metrics, never a student feature. Training uses cell IDs only."""
    return case.identifier.split('-')[2]


def digest(cases):
    payload=[(x.identifier,x.family,x.topology,x.demos,x.query,x.future,x.hidden_graph,x.pair) for x in cases]
    return sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
