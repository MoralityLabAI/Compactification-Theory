"""RRC-18: class-supervised direct prediction from observed formal-series pairs.

Only the public two 0..12 input/output demonstrations and a topology identity
enter the learner. Exact candidate execution and group labels are accessible
only to the offline teacher and the independent post-proposal verifier.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import math
import random

try:
    from . import topology16 as t, graph11 as g, certify18 as cert, graph12 as b12
except ImportError:
    import topology16 as t
    import graph11 as g
    import certify18 as cert
    import graph12 as b12

SEED = 2026111018
PREFIX = t.PREFIX
HORIZON = t.HORIZON
TOPOLOGIES = t.TOPOLOGIES
BUDGET = 12
TRAIN_PER = 900
DEV_PER = 100
COUNTS = {'signed_holdout': 240, 'product_shift': 160,
          'out_of_grammar': 20, 'future_collision': 16}


def class_map():
    """Offline certified equivalence for all 64 syntax programs, per topology."""
    result={}
    for topology in TOPOLOGIES:
        classes,_=cert.build_classes(topology)
        rep=lambda cl:min(cl,key=lambda p:(t.graph_cost(p,topology),g.GRAPHS.index(p)))
        mapping={p:rep(cl) for cl in classes for p in cl}
        representatives=tuple(sorted(set(mapping.values()),key=g.GRAPHS.index))
        assert set(mapping)==set(g.GRAPHS)
        result[topology]={'classes':classes,'representatives':representatives,'mapping':mapping}
    return result


def roster(*, include_test=True, train_per=TRAIN_PER, dev_per=DEV_PER):
    if train_per<=0 or dev_per<=0:raise ValueError('positive train/dev counts required')
    output={'train':[],'dev':[]}
    if include_test:output.update({k:[] for k in COUNTS})
    seen=set()
    def append(split,case,duplicate=False):
        if case.visible in seen and not duplicate:return False
        seen.add(case.visible)
        output[split].append(case)
        return True
    for split,n,offset in (('train',train_per,101),('dev',dev_per,202)):
        for i,topology in enumerate(TOPOLOGIES):
            rng=random.Random(SEED+offset+i*1013)
            attempts=0
            while len([x for x in output[split] if x.topology==topology])<n:
                attempts+=1
                if attempts>n*300:raise RuntimeError('roster sampling exhausted '+split)
                gr=rng.choice(g.GRAPHS)
                case=t._case(rng,gr,topology,split,f'{split}-{topology}-{attempts:05d}')
                append(split,case)
    if not include_test:return output
    for split,per_topo,offset,product in (('signed_holdout',60,303,False),('product_shift',40,404,True)):
        for i,topology in enumerate(TOPOLOGIES):
            rng=random.Random(SEED+offset+i*1013)
            successes=0;attempts=0
            while successes<per_topo:
                attempts+=1
                if attempts>50000:raise RuntimeError('test sampling exhausted '+split)
                graph=rng.choice(g.GRAPHS)
                c=t._case(rng,graph,topology,split,f'{split}-{topology}-{attempts:05d}',product=product)
                if append(split,c):successes+=1
    rng=random.Random(SEED+505)
    attempts=0
    while len(output['out_of_grammar'])<COUNTS['out_of_grammar']:
        attempts+=1
        if attempts>50000:raise RuntimeError('negative class exhaustion')
        topo=rng.choice(TOPOLOGIES)
        c=t._case(rng,rng.choice(g.GRAPHS),topo,'out_of_grammar',f'negative-{attempts:05d}',mutation=True)
        if c.visible in seen or t.matching_graphs(c):continue
        append('out_of_grammar',c)
    rng=random.Random(SEED+606)
    for i in range(COUNTS['future_collision']//2):
        for k in range(50000):
            topology=TOPOLOGIES[i%len(TOPOLOGIES)]
            c=t._case(rng,rng.choice(g.GRAPHS),topology,'future_collision',f'collision-{2*i:03d}',pair=f'pair-{i:02d}')
            if append('future_collision',c):break
        else:raise RuntimeError('collision sampling exhausted')
        altered=list(c.future);altered[PREFIX+3]+=1 if i%2 else -1
        duplicate=replace(c,identifier=f'collision-{2*i+1:03d}',future=tuple(altered),hidden_graph=None)
        assert append('future_collision',duplicate,duplicate=True)
    return output


def digest(items):
    return sha256(json.dumps([(c.identifier,c.topology,c.family,c.demos,c.query,c.future,c.hidden_graph,c.pair) for c in items],separators=(',',':')).encode()).hexdigest()


def topo_onehot(topology):
    if topology not in TOPOLOGIES:raise ValueError('unknown topology')
    return [float(topology==k) for k in TOPOLOGIES]


def normalize(value):
    return math.copysign(min(30.0,math.log1p(abs(value)))/10.,value) if value else 0.


def sequence(case):
    """13 time steps with 4 raw coeffs, normalized degree, topology one-hot."""
    a,b=case.demos
    return [[normalize(a[0][n]),normalize(a[1][n]),normalize(b[0][n]),normalize(b[1][n]),
             n/PREFIX]+topo_onehot(case.topology) for n in range(PREFIX+1)]


def flat(case):
    raw=[normalize(v) for f,y in case.demos for seq in (f,y) for v in seq[:PREFIX+1]]
    assert len(raw)==4*(PREFIX+1)
    return raw+topo_onehot(case.topology)


def labels(cases,mapping):
    return [g.GRAPHS.index(mapping[c.topology]['mapping'][c.hidden_graph]) for c in cases]


def candidates(logits, topology, mapping, *, count=12):
    allowed=mapping[topology]['representatives']
    return tuple(sorted(allowed,key=lambda p:(-float(logits[g.GRAPHS.index(p)]),g.GRAPHS.index(p)))[:count])


def mass_order(topology,mapping):
    classes=mapping[topology]['classes']
    members=mapping[topology]['mapping']
    return tuple(sorted(mapping[topology]['representatives'],key=lambda p:(
        -sum(members[x]==p for x in g.GRAPHS), t.graph_cost(p,topology),g.GRAPHS.index(p))))


def infer_cost(kind,hidden=32):
    if kind in ('recurrent','shuffled'):
        return (PREFIX+1)*3*hidden*(9+hidden)+hidden*64
    if kind=='feedforward':return 56*51+51*64
    return 0


def search(case, ordered, *, budget=BUDGET):
    """Each candidate is one paid two-demo execution; finite verification."""
    if budget<1:raise ValueError('budget')
    seen=set();trace=[];work=0
    for p in ordered:
        if len(trace)>=budget:break
        if p in seen:raise AssertionError('duplicate class query')
        seen.add(p);trace.append(p)
        work+=2*t.graph_cost(p,case.topology)
        # Execute both demonstrations independently, even if the first
        # mismatches. One query therefore always costs two executions;
        # the nominal work ledger charges exactly the performed operations.
        checks=[t.execute(f[:PREFIX+1],p,case.topology)==y[:PREFIX+1]
                for f,y in case.demos]
        if all(checks):
            return {'graph':p,'trace':tuple(trace),'integer_loop_proxy':work}
    return {'graph':None,'trace':tuple(trace),'integer_loop_proxy':work}
