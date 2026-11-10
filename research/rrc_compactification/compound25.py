"""RRC-25: two-branch delayed suffixes with shared first-fault signatures.

Finite formal series, no learned model in the public exact interpreter.
Every accepted proposal is checked on two observed degree-192 demonstrations;
future degree-256 coefficients are used only in independent grading.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import random
from typing import Sequence

try:
    from . import delay24 as d
except ImportError:
    import delay24 as d

SEED = 2026111025
OBSERVED = 192
FUTURE = 256
OPS = d.OPS
DELAYS = (64, 80, 96, 112)
SPACINGS = (24, 32)
AMPLITUDES = (-2, -1, 1, 2)
TRAIN_FAMILIES = ('signed', 'sparse', 'product')
TEST_FAMILIES = TRAIN_FAMILIES + ('rational', 'squared', 'late_source')
TRAIN_PER = 240
DEV_PER = 48
TEST_PER = 32
WINDOW = 8
NEGATIVE_N = 12
COLLISION_PAIRS = 6

@dataclass(frozen=True)
class Case:
    identifier: str
    family: str
    demos: tuple
    query: tuple
    future: tuple
    program: tuple | None
    pair: str | None = None
    @property
    def visible(self):
        return tuple((f[:OBSERVED+1], y[:OBSERVED+1]) for f,y in self.demos)

@dataclass(frozen=True)
class Header:
    base: str
    delay: int
    spacing: int
    amplitude: int


def apply_suffix(base: Sequence[int], source: Sequence[int], program: tuple) -> tuple[int,...]:
    """B + c*q^K + c*q^(K+D)*U(F) + c*q^(K+2D)*V(F)."""
    baseop,k,delta,c,u,v = program
    if baseop not in OPS or u not in OPS or v not in OPS or k not in DELAYS or delta not in SPACINGS or c not in AMPLITUDES:
        raise ValueError('invalid composite program')
    if len(base)!=len(source) or not base or source[0]!=1 or base[0]!=1:
        raise ValueError('mismatched normalized series')
    degree=len(base)-1
    left=d.operate(source,u)
    right=d.operate(source,v)
    result=list(base)
    result[k]+=c
    for n in range(k+delta,degree+1):result[n]+=c*left[n-(k+delta)]
    for n in range(k+2*delta,degree+1):result[n]+=c*right[n-(k+2*delta)]
    return tuple(result)


def execute(source: Sequence[int], program: tuple) -> tuple[int,...]:
    return apply_suffix(d.operate(source,program[0]),source,program)


def source(rng:random.Random,family:str):
    return d.generate_source(rng,family,degree=FUTURE)


def make_case(rng:random.Random,family:str,identifier:str,program=None,pair=None):
    if program is None:
        program=(rng.choice(OPS),rng.choice(DELAYS),rng.choice(SPACINGS),
                 rng.choice(AMPLITUDES),rng.choice(OPS),rng.choice(OPS))
    demos=tuple((f:=source(rng,family),execute(f,program)) for _ in range(2))
    query=source(rng,family)
    return Case(identifier,family,demos,query,execute(query,program),program,pair)


def roster(*,include_test=True,train_per=TRAIN_PER,dev_per=DEV_PER,test_per=TEST_PER):
    if min(train_per,dev_per,test_per)<1:raise ValueError('positive sample counts required')
    groups={'train':[],'dev':[]}
    if include_test:groups.update({k:[] for k in TEST_FAMILIES});groups.update({'negative':[],'collision':[]})
    observed=set()
    def add(split,case,duplicate=False):
        if case.visible in observed and not duplicate:return False
        observed.add(case.visible);groups[split].append(case);return True
    def fill(split,family,n,seed):
        rng=random.Random(seed)
        for tries in range(n*600):
            if sum(z.family==family for z in groups[split])>=n:return
            item=make_case(rng,family,f'{split}-{family}-{tries:05d}')
            add(split,item)
        raise RuntimeError('roster exhausted: '+split+'/'+family)
    for ix,fam in enumerate(TRAIN_FAMILIES):
        fill('train',fam,train_per,SEED+1000+1009*ix)
        fill('dev',fam,dev_per,SEED+2000+1013*ix)
    if not include_test:return groups
    for ix,fam in enumerate(TEST_FAMILIES):fill(fam,fam,test_per,SEED+3000+1019*ix)
    rng=random.Random(SEED+7700)
    for attempt in range(NEGATIVE_N*1000):
        if len(groups['negative'])==NEGATIVE_N:break
        c=make_case(rng,'signed',f'negative-{attempt:04d}')
        f,y=c.demos[1];bad=list(y);bad[OBSERVED-2]+=1
        alt=replace(c,demos=(c.demos[0],(f,tuple(bad))),program=None)
        if alt.visible not in observed and complete_search(alt)['program'] is None:add('negative',alt)
    if len(groups['negative'])!=NEGATIVE_N:raise RuntimeError('negative roster exhausted')
    rng=random.Random(SEED+8800)
    for i in range(COLLISION_PAIRS):
        fam=TEST_FAMILIES[i%5]
        while True:
            base=make_case(rng,fam,f'collision-{2*i:04d}',pair=f'pair-{i}')
            if base.visible in observed:continue
            candidate=complete_search(base)['program']
            if grade(base,candidate)['future_correct']!=1:continue
            assert add('collision',base)
            break
        future=list(base.future);future[224]+=(1 if i%2==0 else -1)
        modified=replace(base,identifier=f'collision-{2*i+1:04d}',future=tuple(future),program=None)
        assert add('collision',modified,duplicate=True)
    return groups


def fingerprint(items):
    return sha256(json.dumps([(z.identifier,z.family,z.demos,z.query,z.future,z.program,z.pair) for z in items],separators=(',',':')).encode()).hexdigest()


def possible_headers(case:Case, *, audited=False):
    """Public *four-base* algebraic scan. Infers K,c,Delta, no U/V.

    Header candidates share the first two nonzero delayed residuals. The
    secondary U,V semantics are intentionally left unresolved.
    """
    headers=[];executions=0;work=0
    for op in OPS:
        inferred=[]
        for f,y in case.demos:
            executions+=1;work+=base_work(op)
            base=d.operate(f[:OBSERVED+1],op)
            residual=[a-b for a,b in zip(y[:OBSERVED+1],base)]
            k=next((i for i,v in enumerate(residual) if v),None)
            if k not in DELAYS:break
            c=residual[k]
            if c not in AMPLITUDES:break
            residual[k]-=c
            next_degree=next((i for i,v in enumerate(residual) if v),None)
            if next_degree is None or next_degree-k not in SPACINGS or residual[next_degree]!=c:break
            inferred.append((k,next_degree-k,c))
        if len(inferred)==2 and inferred[0]==inferred[1]:
            k,delta,c=inferred[0]
            headers.append(Header(op,k,delta,c))
    return (tuple(headers),executions,work) if audited else tuple(headers)


def verify_template(case:Case,header:Header,u:str,v:str) -> bool:
    """One *paid* full candidate check on two public demonstrations."""
    program=(header.base,header.delay,header.spacing,header.amplitude,u,v)
    return all([execute(f[:OBSERVED+1],program)==y[:OBSERVED+1] for f,y in case.demos])


def preview_errors(case:Case,header:Header,u:str,v:str) -> tuple[int,int]:
    """A charged short-template preview through the two delayed windows.

    Only the U branch to degree Delta+8 and V branch to degree 8 are run;
    no withheld future labels or query coefficients are consulted.
    """
    k,delta,c=header.delay,header.spacing,header.amplitude
    differences=[]
    for f,y in case.demos:
        b=d.operate(f[:OBSERVED+1],header.base)
        left=d.operate(f[:delta+WINDOW+1],u)
        right=d.operate(f[:WINDOW+1],v)
        for n in range(WINDOW+1):
            ix=k+delta+n
            actual=y[ix]-b[ix]-(c if ix==k else 0)
            expected=c*left[n]
            differences.append(abs(actual-expected))
        for n in range(WINDOW+1):
            ix=k+2*delta+n
            actual=y[ix]-b[ix]
            expected=c*(left[delta+n]+right[n])
            differences.append(abs(actual-expected))
    return (next((i for i,vv in enumerate(differences) if vv),len(differences)),
            sum(min(vv,1000000) for vv in differences))


def features(case:Case,header:Header):
    """Observed data only, 9 time steps x 8 values; no candidate execution.

    Header discovery has already executed O(F), a separately charged oracle.
    No source family, hidden graph, U/V identity or future enters the model.
    """
    k,delta,c=header.delay,header.spacing,header.amplitude
    output=[]
    def norm(v):return math.copysign(min(20.0,math.log1p(abs(v)))/8.0,v) if v else 0.0
    base=[d.operate(f[:OBSERVED+1],header.base) for f,_ in case.demos]
    for n in range(WINDOW+1):
        row=[]
        for (f,y),b in zip(case.demos,base):
            row.extend((norm(f[n]),norm(f[delta+n]),norm((y[k+delta+n]-b[k+delta+n])/c),
                        norm((y[k+2*delta+n]-b[k+2*delta+n])/c)))
        output.append(row)
    return tuple(tuple(r) for r in output)


def labels(program):return OPS.index(program[4])*4+OPS.index(program[5])


def template_order(case,header,arm,model=None):
    if arm=='fixed':return tuple((u,v) for u in OPS for v in OPS),0,0
    if arm=='preview':
        pairs=tuple((u,v) for u in OPS for v in OPS)
        results=[preview_errors(case,header,u,v) for u,v in pairs]
        return tuple(pairs[j] for j in sorted(range(16),key=lambda j:(-results[j][0],results[j][1],j))),16,0
    if model is None:raise ValueError('missing checkpoint')
    try:from . import model25
    except ImportError:import model25
    scores=model25.logits(case,header,model)
    return tuple((OPS[j//4],OPS[j%4]) for j in sorted(range(16),key=lambda j:(-scores[j],j))),0,model25.macs(model['kind'])


def search(case,arm,model=None,budget=4):
    if not 1<=budget<=len(OPS)**3:raise ValueError('invalid full verification budget')
    headers,base_calls,header_work=possible_headers(case,audited=True)
    seen=[];found=None;preview=0;macs=0;processed=[]
    for header in headers:
        if len(seen)>=budget:break
        processed.append((header.base,header.spacing))
        order,nprev,nmac=template_order(case,header,arm,model)
        preview+=nprev;macs+=nmac
        for u,v in order:
            if len(seen)>=budget:break
            candidate=(header.base,header.delay,header.spacing,header.amplitude,u,v)
            if candidate in seen:raise AssertionError('duplicate candidate')
            seen.append(candidate)
            if verify_template(case,header,u,v):found=candidate;break
        if found is not None or len(seen)>=budget:break
    return {'program':found,'trace':tuple(seen),'headers':len(headers),
            'header_base_executions':base_calls,'header_work_proxy':header_work,
            'full_checks':len(seen),'preview_pairs':preview,
            'processed_headers':tuple(processed),
            'used_preview':arm=='preview', 'used_model':model is not None,
            'model_macs':macs}


def complete_search(case):return search(case,'fixed',budget=64)



def visible_template_count(case):
    """Exactly count observationally equivalent programs via cached operators.

    Independent from trained candidate ranking and future grades. All 16
    ordered U/V branches are checked through q^192 on both public demos.
    """
    count=0
    for header in possible_headers(case):
        matrices=[]
        for f,y in case.demos:
            f=f[:OBSERVED+1];y=y[:OBSERVED+1]
            matrices.append((d.operate(f,header.base),
                             {op:d.operate(f,op) for op in OPS},y))
        k,delta,c=header.delay,header.spacing,header.amplitude
        for u in OPS:
            for v in OPS:
                valid=True
                for base,op_values,target in matrices:
                    left=op_values[u];right=op_values[v]
                    for n in range(OBSERVED+1):
                        expected=base[n]+(c if n==k else 0)
                        if n>=k+delta:expected+=c*left[n-k-delta]
                        if n>=k+2*delta:expected+=c*right[n-k-2*delta]
                        if expected!=target[n]:valid=False;break
                    if not valid:break
                count+=int(valid)
    return count


def grade(case,program):
    if program is None:return {'accepted':0,'future_correct':0,'false_accept':0}
    accepted=int(all(execute(f[:OBSERVED+1],program)==y[:OBSERVED+1] for f,y in case.demos))
    future=int(accepted and execute(case.query,program)==case.future)
    return {'accepted':accepted,'future_correct':future,'false_accept':accepted-future}


def base_work(op,degree=OBSERVED):return d.operation_work(op,degree)


def total_work(out):
    """Nominal, operation-dependent coefficient-loop proxy including previews.

    Uses delay24's declared O(T**2) reciprocal and O(T) other costs;
    Python dispatch, integer bit complexity, training and grading are excluded.
    """
    header=out['header_work_proxy']
    candidate=sum(2*(base_work(z[0])+base_work(z[4])+base_work(z[5])+3*(OBSERVED+1)) for z in out['trace'])
    preview=0
    for op,delta in out['processed_headers']:
        if out['used_preview']:
            for u in OPS:
                for v in OPS:
                    preview+=2*(base_work(op)+base_work(u,delta+WINDOW)+base_work(v,WINDOW))
    features=2*sum(base_work(op) for op,_ in out['processed_headers']) if out['used_model'] else 0
    return header+candidate+preview+features
