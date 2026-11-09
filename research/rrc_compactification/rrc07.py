"""RRC-07: fresh periodic products and finite exact first-fault residual feedback.

Exact feedback is an algorithmic comparator, kept outside neural inference.
The new test cases are withheld from the RRC-07 training invocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isqrt
import json
import random
from typing import Sequence

try:
    from .rrc06 import PREFIX, HORIZON, raw_features, rounded_predictions
    from .rrc05 import (Program05, product_coefficients, coefficient_residual,
                        proposed_programs_from_euler)
    from .rrc import gap_coefficients, euler_exponents
    from .program04 import arbitrary_part_coefficients, late_defect
except ImportError:
    from rrc06 import PREFIX, HORIZON, raw_features, rounded_predictions
    from rrc05 import (Program05, product_coefficients, coefficient_residual,
                       proposed_programs_from_euler)
    from rrc import gap_coefficients, euler_exponents
    from program04 import arbitrary_part_coefficients, late_defect

SEED=2026110917
TRAIN_PERIODS=(2,3,4,6,8,9,10,12)
TEST_PERIODS=(5,7,11,13,17,19)
SIZES={'train':600,'dev':96,'unseen_moduli':72,'unseen_cap':30,
       'local_gap':30,'nonperiodic':12,'collision':10}

@dataclass(frozen=True)
class Case07:
    id: str
    split: str
    family: str
    prefix: tuple[int,...]
    full: tuple[int,...]
    origin: str
    pair: str|None=None

    def prefix_hash(self)->str:
        return sha256(json.dumps(self.prefix,separators=(',',':')).encode()).hexdigest()


def new_roster(*,train_size:int=SIZES['train'],dev_size:int=SIZES['dev'],
               unseen_per_period:int=12,include_test:bool=True)->dict[str,list[Case07]]:
    """Domain registration; train07 calls include_test=False, without test generation."""
    if min(train_size,dev_size,unseen_per_period)<1:raise ValueError('empty roster')
    data={k:[] for k in SIZES}
    observed=set()
    def sample(split,desired,seed,periods,caps,degree):
        rng=random.Random(seed)
        tries=0
        while len(data[split])<desired:
            tries+=1
            if tries>desired*500:raise RuntimeError('unique source exhaustion: '+split)
            period=rng.choice(periods); cap=rng.choice(caps)
            density=rng.choice((.2,.35,.5,.65,.8))
            residues=tuple(r for r in range(period) if rng.random()<density)
            if not residues:residues=(rng.randrange(period),)
            program=Program05(cap,period,residues)
            full=tuple(product_coefficients(degree,program))
            prefix=full[:PREFIX+1]
            if prefix in observed:continue
            observed.add(prefix)
            data[split].append(Case07(f'{split}-{len(data[split]):03d}',split,
                            'periodic',prefix,full,repr(program)))
    sample('train',train_size,SEED+10,TRAIN_PERIODS,(0,1),PREFIX)
    sample('dev',dev_size,SEED+20,TRAIN_PERIODS,(0,1),HORIZON)
    if not include_test:return data
    sample('unseen_moduli',len(TEST_PERIODS)*unseen_per_period,SEED+30,
           TEST_PERIODS,(0,1),HORIZON)
    sample('unseen_cap',SIZES['unseen_cap'],SEED+40,TRAIN_PERIODS,(2,),HORIZON)
    for gap in range(1,6):
        for minimum in range(1,7):
            full=tuple(gap_coefficients(HORIZON,gap=gap,minimum_part=minimum))
            data['local_gap'].append(Case07(f'gap{gap}-min{minimum}','local_gap',
                  'gap',full[:PREFIX+1],full,f'gap={gap} minimum={minimum}'))
    def isprime(n):return n>=2 and all(n%k for k in range(2,isqrt(n)+1))
    families={
       'pentagonal':{n*(3*n-1)//2 for n in range(1,12)},
       'cube':{n**3 for n in range(1,6)},
       'twinprime':{n for n in range(2,HORIZON+1) if isprime(n) and isprime(n+2)},
       'semiprime':{n for n in range(4,HORIZON+1) if sum(isprime(d) and isprime(n//d) for d in range(2,isqrt(n)+1) if n%d==0)>0}
    }
    for name,parts in families.items():
        for cap in (0,1,2):
            full=tuple(arbitrary_part_coefficients(HORIZON,sorted(parts),cap))
            data['nonperiodic'].append(Case07(f'{name}-cap{cap}','nonperiodic',
                  'nonperiodic',full[:PREFIX+1],full,name+f' cap={cap}'))
    rng=random.Random(SEED+50)
    for k in (67,79,91,107,123):
        period=rng.choice((7,11,13,17));cap=rng.choice((0,1))
        res=tuple(i for i in range(period) if rng.random()<.42)
        if not res:res=(1,)
        program=Program05(cap,period,res)
        base=tuple(product_coefficients(HORIZON,program))
        changed=tuple(late_defect(base,k))
        g=f'late-{k}'
        data['collision'].extend((
            Case07(g+'-base','collision','collision',base[:PREFIX+1],base,repr(program),g),
            Case07(g+'-mutant','collision','collision',changed[:PREFIX+1],changed,
                   repr(program)+f'/(1-q^{k})',g)))
    return data


def roster_hash(cases:Sequence[Case07])->str:
    items=[{'id':c.id,'split':c.split,'family':c.family,'prefix':c.prefix_hash(),
            'full':sha256(json.dumps(c.full,separators=(',',':')).encode()).hexdigest(),
            'origin':c.origin,'pair':c.pair} for c in cases]
    return sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def first_fault(prefix:Sequence[int],estimate:Sequence[int])->int|None:
    """Full exact residual, one *charged* verifier query. No teacher inversion."""
    if len(prefix)!=PREFIX+1 or len(estimate)!=PREFIX+1:raise ValueError('length mismatch')
    residual=coefficient_residual(prefix,estimate)
    return next((n for n in range(1,PREFIX+1) if residual[n]),None)


def fault_feedback(prefix:Sequence[int],estimate:Sequence[int],budget:int)->dict:
    """Repair one *first* faulty coefficient with a disclosed exact oracle per pass.

    At the first faulty index m, R[m] == m*(a[m]-estimate[m]), since all
    smaller exponents already agree. Charge a complete residual verification
    after every edit; the last verification is recorded separately.
    """
    if budget<0 or budget>PREFIX:raise ValueError('budget outside 0..64')
    if estimate[0]!=0 or any(not isinstance(x,int) for x in estimate):
        raise ValueError('integer candidate required')
    a=list(estimate);edits=[]; queries=0; last_residual=None
    for _ in range(budget):
        r=coefficient_residual(prefix,a);queries+=1
        m=next((n for n in range(1,PREFIX+1) if r[n]),None)
        if m is None:
            last_residual=r
            break
        if r[m]%m:raise AssertionError('triangular first-fault relation failed')
        delta=r[m]//m
        a[m]+=delta
        edits.append({'degree':m,'delta':delta})
    if last_residual is None:
        result=coefficient_residual(prefix,a);queries+=1
    else:
        result=last_residual
    fault=next((n for n in range(1,PREFIX+1) if result[n]),None)
    programs=proposed_programs_from_euler(a,prefix) if fault is None else ()
    return {'proposal':a,'first_fault':fault,'corrected':len(edits),
            'queried_residual_vectors':queries,'edits':edits,
            'verified_prefix':fault is None,'program':programs[0] if programs else None}


def verified_program(prefix:Sequence[int],proposal:Sequence[int])->Program05|None:
    if first_fault(prefix,proposal) is not None:return None
    return next(iter(proposed_programs_from_euler(proposal,prefix)),None)


def score_proposal(case:Case07,proposal:Sequence[int])->dict:
    if len(proposal)!=PREFIX+1 or proposal[0]!=0 or any(not isinstance(x,int) for x in proposal):
        raise ValueError('malformed Euler proposal')
    expected=euler_exponents(case.prefix)  # evaluation only
    fault=first_fault(case.prefix,proposal)
    p=verified_program(case.prefix,proposal) if fault is None else None
    future=int(p is not None and tuple(product_coefficients(HORIZON,p))==case.full)
    return {'a_correct':sum(a==b for a,b in zip(proposal[1:],expected[1:])),
            'first_error':next((n for n in range(1,PREFIX+1) if proposal[n]!=expected[n]),None),
            'first_residual':fault,'verified_prefix':int(fault is None),
            'accepted_program':int(p is not None),'correct_future':future,
            'false_accept':int(p is not None)-future,
            'program':p.as_dict() if p else None}
