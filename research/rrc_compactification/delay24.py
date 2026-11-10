"""RRC-24: late q-adic factor search, exact coefficient certificates and rosters.

A program is (O,K,c): G(F)=O(F)*(1+c*q**K), where O is I/N/R/S.
The model ranks O only. A public exact inference procedure determines (K,c)
after each paid O candidate. This is an *interpreter-assisted* ranker.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import random

try:
    from . import graph11 as g
except ImportError:
    import graph11 as g

SEED=2026111024
OPS=('I','N','R','S')
OBSERVED=192
FUTURE=256
MODEL_PREFIX=16
DELAYS=(64,80,96,112,128,144,160,176)
AMPLITUDES=(-2,-1,1,2)
TRAIN_FAMILIES=('signed','product','sparse')
TEST_FAMILIES=('signed','product','sparse','rational','squared','late_source')
TRAIN_PER=320
DEV_PER=64
TEST_PER=40
NEGATIVE_COUNT=12
COLLISION_PAIRS=8

@dataclass(frozen=True)
class Case:
    identifier:str
    family:str
    demos:tuple
    query:tuple
    future:tuple
    program:tuple|None
    pair:str|None=None
    @property
    def visible(self):
        return tuple((a[:OBSERVED+1],b[:OBSERVED+1]) for a,b in self.demos)


def generate_source(rng, family, degree=FUTURE):
    if family=='signed':
        return (1,rng.choice((-2,-1,1,2)))+tuple(rng.randint(-2,2) for _ in range(degree-1))
    if family=='sparse':
        return (1,rng.choice((-2,-1,1,2)))+tuple(rng.choice((-2,-1,1,2)) if rng.random()<.13 else 0 for _ in range(degree-1))
    if family=='product':
        mod=rng.choice((5,7,9,11)); allowed={1}
        for k in range(mod):
            if rng.random()<.45:allowed.add(k)
        coeff=[1]+[0]*degree
        for p in range(1,degree+1):
            if p%mod in allowed:
                for n in range(p,degree+1):coeff[n]+=coeff[n-p]
        return tuple(coeff)
    if family in ('rational','squared'):
        pol=[1,rng.choice((-2,-1,1,2))]+[0]*(degree-1)
        for _ in range(7):
            k=rng.randrange(2,min(degree,36)+1)
            pol[k]+=rng.choice((-2,-1,1,2))
        if family=='rational':return g.inv(tuple(pol))
        return g.mul(tuple(pol),tuple(pol))
    if family=='late_source':
        onset=rng.randrange(57,80)
        poly=[1]+[0]*degree
        poly[onset]=rng.choice((-2,-1,1,2))
        for n in range(onset+1,degree+1):
            if rng.random()<.17:poly[n]=rng.choice((-2,-1,1,2))
        return tuple(poly)
    raise ValueError('unknown source family')


def operate(f,operation):
    if not f or f[0]!=1:raise ValueError('input must be an integral unit series')
    if operation not in OPS:raise ValueError('unknown operation')
    return g.unary(tuple(f),operation)


def apply_factor(base,delay=None,amplitude=0):
    if delay is None:
        if amplitude!=0:raise ValueError('null delay needs zero amplitude')
        return tuple(base)
    if not 1<=delay<len(base) or amplitude not in AMPLITUDES:
        raise ValueError('invalid finite delayed factor')
    return tuple(v+(amplitude*base[n-delay] if n>=delay else 0) for n,v in enumerate(base))


def execute(f,program):
    op,k,c=program
    return apply_factor(operate(f,op),k,c)


def _sample_case(rng,family,identifier,*,delay=None,amplitude=None,op=None,pair=None):
    op=rng.choice(OPS) if op is None else op
    delay=rng.choice(DELAYS) if delay is None else delay
    amplitude=rng.choice(AMPLITUDES) if amplitude is None else amplitude
    prog=(op,delay,amplitude)
    demos=tuple((f:=generate_source(rng,family),execute(f,prog)) for _ in range(2))
    query=generate_source(rng,family)
    future=execute(query,prog)
    return Case(identifier,family,demos,query,future,prog,pair)


def canonical_signals(case,degree=MODEL_PREFIX):
    """Public 2-demo input/output coefficients, without hidden program or future."""
    if degree>OBSERVED:raise ValueError('overspecified public degree')
    return tuple((f[:degree+1],y[:degree+1]) for f,y in case.demos)


def dataset(*,include_test=True,train_per=TRAIN_PER,dev_per=DEV_PER,test_per=TEST_PER):
    if min(train_per,dev_per,test_per)<1:raise ValueError('positive counts required')
    groups={'train':[],'dev':[]}
    if include_test:groups.update({name:[] for name in TEST_FAMILIES})
    if include_test:groups.update({'negative':[],'collision':[]})
    seen=set()
    def append(key,item,allow_duplicate=False):
        visible=item.visible
        if visible in seen and not allow_duplicate:return False
        seen.add(visible);groups[key].append(item);return True
    for key,count,offset in [('train',train_per,1000),('dev',dev_per,2000)]:
        for idx,family in enumerate(TRAIN_FAMILIES):
            rng=random.Random(SEED+offset+idx*991)
            tries=0
            while sum(z.family==family for z in groups[key])<count:
                tries+=1
                if tries>count*500:raise RuntimeError('training exhaustion')
                c=_sample_case(rng,family,f'{key}-{family}-{tries:05d}')
                append(key,c)
    if not include_test:return groups
    for idx,family in enumerate(TEST_FAMILIES):
        rng=random.Random(SEED+4000+idx*1133)
        tries=0
        while len(groups[family])<test_per:
            tries+=1
            if tries>test_per*500:raise RuntimeError('heldout exhaustion')
            c=_sample_case(rng,family,f'{family}-{tries:05d}')
            append(family,c)
    rng=random.Random(SEED+8500)
    tries=0
    while len(groups['negative'])<NEGATIVE_COUNT:
        tries+=1
        if tries>12000:raise RuntimeError('negatives exhausted')
        c=_sample_case(rng,'signed',f'negative-{tries:04d}')
        f,y=c.demos[1];altered=list(y)
        # Deliberately inconsistent with the exact same delayed factor on both demos.
        altered[OBSERVED-3]+=1
        c=replace(c,demos=(c.demos[0],(f,tuple(altered))),program=None)
        if c.visible in seen:continue
        if any(verify_candidate(c,op)['program'] is not None for op in OPS):continue
        append('negative',c)
    rng=random.Random(SEED+9700)
    for i in range(COLLISION_PAIRS):
        while True:
            op=rng.choice(OPS)
            family=TEST_FAMILIES[i%len(TEST_FAMILIES)]
            c=_sample_case(rng,family,f'collision-{2*i:03d}',delay=224,amplitude=1,op=op,pair=f'pair-{i}')
            if c.visible in seen:continue
            # The alternative has no factor. Identical visible outputs up to q^192.
            base=replace(c,demos=tuple((f,operate(f,op)) for f,_ in c.demos),
                         future=operate(c.query,op),program=(op,None,0))
            assert append('collision',base)
            assert append('collision',c,allow_duplicate=True)
            break
    return groups


def learn_features(case):
    """Model receives only the first 17 coefficients of both demo input/output pairs."""
    import math
    def value(n):return math.copysign(min(24.,math.log1p(abs(n)))/8.,n) if n else 0.
    return tuple((value(f[n]),value(y[n]),value(h[n]),value(z[n]))
                 for n in range(MODEL_PREFIX+1)
                 for (f,y),(h,z) in [case.demos])


def verify_candidate(case,op,*,degree=OBSERVED):
    """One paid base-skeleton candidate on both public demos.

    Infer the unique first residual q-adic shift (delay, amplitude), then
    demand equality of ALL coefficients through the visible degree.
    This is a privileged exact algebraic suffix solver, shared by all arms.
    """
    if op not in OPS:raise ValueError('invalid base')
    if not 1<=degree<=OBSERVED:raise ValueError('invalid observation degree')
    common=None
    for f,y in case.demos:
        base=operate(f[:degree+1],op)
        dif=[a-b for a,b in zip(y[:degree+1],base)]
        first=next((n for n,x in enumerate(dif) if x),None)
        if first is None:term=(None,0)
        elif first in DELAYS and dif[first] in AMPLITUDES:term=(first,dif[first])
        else:return {'program':None,'base_evaluations':2,'source':'rejected'}
        if common is None:common=term
        elif term!=common:return {'program':None,'base_evaluations':2,'source':'rejected'}
        if tuple(y[:degree+1])!=apply_factor(base,*term):
            return {'program':None,'base_evaluations':2,'source':'rejected'}
    return {'program':(op,*common),'base_evaluations':2,'source':'verified'}


def first_difference_degree(f,g):
    if len(f)!=len(g):raise ValueError('length mismatch')
    return next((i for i,(x,y) in enumerate(zip(f,g)) if x!=y),None)


def base_order_exact_prefix(case,degree=MODEL_PREFIX):
    """A transparent short-degree public interpreter comparator."""
    signals=canonical_signals(case,degree)
    matches=[]
    for op in OPS:
        count=sum(operate(f,op)==y for f,y in signals)
        matches.append((2-count,OPS.index(op),op))
    return tuple(z[2] for z in sorted(matches))


def search(case,order,budget=4):
    if not 1<=budget<=4:raise ValueError('base-query budget 1..4')
    trace=[];program=None
    for op in order:
        if len(trace)==budget:break
        if op in trace:raise AssertionError('duplicate interpreter base call')
        trace.append(op)
        res=verify_candidate(case,op)
        if res['program'] is not None:
            program=res['program'];break
    return {'program':program,'trace':tuple(trace),'queries':len(trace),
            'base_executions':2*len(trace),'approx_ops':sum(2*operation_work(x,OBSERVED) for x in trace)}


def operation_work(op,degree):
    """Algebraic-proxy operations for the base map, plus O(T) shift compare."""
    return (degree*(degree+1)//2 if op=='R' else degree+1)+2*(degree+1)


def grade(case,program):
    if program is None:return {'accepted':0,'future_correct':0,'false_accept':0}
    # The interpreter-checked finite candidate is evaluated on the HELD-OUT query
    # only after the proposal is frozen.
    accepted=int(all(execute(f[:OBSERVED+1],program)==y[:OBSERVED+1] for f,y in case.demos))
    correct=int(accepted and execute(case.query,program)==case.future)
    return {'accepted':accepted,'future_correct':correct,'false_accept':accepted-correct}


def fingerprint(cases):
    return sha256(json.dumps([(c.identifier,c.family,c.demos,c.query,c.future,c.program,c.pair)
                for c in cases],separators=(',',':')).encode()).hexdigest()
