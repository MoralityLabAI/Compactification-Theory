"""RRC-06: raw coefficient observations and independent exact trace certificates.

Neural inference uses features of f_0..f_64 only. Exact transformations in
this module are training labels, evaluation baselines, or post-hoc verifier
calls. They are never called by the inference routines in raw_student06.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Sequence

try:
    from .rrc05 import (Program05, CAPS, PREFIX, HORIZON, PERIOD_CAP,
                        product_coefficients, coefficient_residual,
                        log_derivative_coefficients, proposed_programs_from_euler)
    from .rrc import euler_exponents, gap_coefficients
    from .program04 import arbitrary_part_coefficients, late_defect
except ImportError:
    from rrc05 import (Program05, CAPS, PREFIX, HORIZON, PERIOD_CAP,
                       product_coefficients, coefficient_residual,
                       log_derivative_coefficients, proposed_programs_from_euler)
    from rrc import euler_exponents, gap_coefficients
    from program04 import arbitrary_part_coefficients, late_defect

SEED=2026110916
TRAIN_PERIODS=(2,3,4,6,8,9,10,12)
UNSEEN_PERIODS=(5,7,11,13,17,19)
SPLIT_SIZES={'train':720,'dev':120,'unseen_period':72,'unseen_cap':36,
             'local_gap':40,'nonperiodic':15,'out_of_grammar':12,'collision':10}

@dataclass(frozen=True)
class Case:
    name:str
    split:str
    family:str
    prefix:tuple[int,...]
    full:tuple[int,...]
    source:str
    collision_group:str|None=None

    def prefix_hash(self)->str:
        return sha256(json.dumps(self.prefix,separators=(',',':')).encode()).hexdigest()


def raw_features(f:Sequence[int])->list[list[float]]:
    """Only public raw counts, neighboring log difference and position.

    The teacher's b or a, multiplicity caps, residues, divisors, and source
    family are excluded. log1p is a generic stabilization transformation.
    """
    if len(f)!=PREFIX+1 or f[0]!=1 or any(not isinstance(x,int) or x<0 for x in f):
        raise ValueError('expected nonnegative integer f_0..f_64 with f_0=1')
    logs=[math.log1p(v)/16.0 for v in f]
    return [[logs[i], (logs[i]-logs[i-1])*8.0 if i else 0.0, i/PREFIX]
            for i in range(PREFIX+1)]


def random_program(rng:random.Random, periods:Sequence[int],caps:Sequence[int])->Program05:
    period=rng.choice(periods)
    cap=rng.choice(caps)
    density=rng.choice((.2,.35,.5,.65,.8))
    mask=tuple(i for i in range(period) if rng.random()<density)
    if not mask:mask=(rng.randrange(period),)
    return Program05(cap,period,mask)


def roster(*,train_size:int=720,dev_size:int=120,
           unseen_per_period:int=12,unseen_cap_count:int=36,
           include_test:bool=True)->dict[str,list[Case]]:
    if min(train_size,dev_size,unseen_per_period,unseen_cap_count)<1:
        raise ValueError('all roster sizes must be positive')
    splits={key:[] for key in SPLIT_SIZES}
    seen=set()
    def draw(label:str,target:int,seed:int,periods:Sequence[int],caps:Sequence[int],degree:int):
        rng=random.Random(seed)
        trials=0
        while len(splits[label])<target:
            trials+=1
            if trials>target*500:raise RuntimeError(f'no unique prefixes for {label}')
            prog=random_program(rng,periods,caps)
            full=tuple(product_coefficients(degree,prog))
            prefix=full[:PREFIX+1]
            if prefix in seen:continue
            seen.add(prefix)
            splits[label].append(Case(f'{label}-{len(splits[label]):03d}',label,
                           'periodic_product',prefix,full,repr(prog)))
    draw('train',train_size,SEED+10,TRAIN_PERIODS,(0,1),PREFIX)
    draw('dev',dev_size,SEED+20,TRAIN_PERIODS,(0,1),HORIZON)
    if not include_test:return splits
    draw('unseen_period',6*unseen_per_period,SEED+30,UNSEEN_PERIODS,(0,1),HORIZON)
    draw('unseen_cap',unseen_cap_count,SEED+40,TRAIN_PERIODS,(2,),HORIZON)
    for gap in range(1,6):
        for minimum in range(1,9):
            full=tuple(gap_coefficients(HORIZON,gap=gap,minimum_part=minimum))
            splits['local_gap'].append(Case(f'gap{gap}-min{minimum}','local_gap',
                'local_gap',full[:PREFIX+1],full,f'gap({gap},{minimum})'))
    def squarefree(n:int)->bool:
        return n>=1 and all(n%(d*d) for d in range(2,math.isqrt(n)+1))
    sets={
        'squarefree':[n for n in range(1,HORIZON+1) if squarefree(n)],
        'powers_three':[3**k for k in range(5)],
        'factorials':[math.factorial(k) for k in range(1,7)],
        'oblong':[k*(k+1) for k in range(1,12)],
        'odd_prime_powers':[n for n in range(3,HORIZON+1,2)
                           if any(n==p**k for p in (3,5,7,11,13) for k in range(1,7))]
    }
    for name,parts in sets.items():
        for cap in CAPS:
            full=tuple(arbitrary_part_coefficients(HORIZON,parts,cap))
            splits['nonperiodic'].append(Case(f'{name}-cap{cap}',
                'nonperiodic','nonperiodic',full[:PREFIX+1],full,name+f' cap={cap}'))
    rng=random.Random(SEED+50)
    for i in range(12):
        period=(23,29,31)[i%3]
        cap=i%3
        mask=tuple(k for k in range(period) if rng.random()<(.3,.5,.7)[i%3])
        if not mask:mask=(1,)
        allowed=[n for n in range(1,HORIZON+1) if n%period in mask]
        full=tuple(arbitrary_part_coefficients(HORIZON,allowed,cap))
        splits['out_of_grammar'].append(Case(f'out-{i:02d}',
            'out_of_grammar','out_of_grammar',full[:PREFIX+1],full,
            f'cap={cap} period={period} mask={mask}'))
    for i,k in enumerate((67,79,101,113,127)):
        # Preserves all observed coefficients while changing a future value.
        prog=random_program(rng,(11,13,17),(0,1))
        base=tuple(product_coefficients(HORIZON,prog))
        mutant=tuple(late_defect(base,k))
        group=f'late-{k}'
        splits['collision'].extend((
            Case(f'{group}-base','collision','collision',base[:PREFIX+1],base,repr(prog),group),
            Case(f'{group}-mutant','collision','collision',mutant[:PREFIX+1],mutant,
                 repr(prog)+f'/(1-q^{k})',group)))
    return splits


def roster_digest(cases:Sequence[Case])->str:
    entries=[{'name':c.name,'split':c.split,'family':c.family,
              'prefix':c.prefix_hash(),'source':c.source,
              'full_sha256':sha256(json.dumps(c.full,separators=(',',':')).encode()).hexdigest(),
              'collision_group':c.collision_group} for c in cases]
    return sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def teacher(f:Sequence[int])->tuple[list[int],list[int]]:
    """Only training supervision and independent reference scoring."""
    return euler_exponents(f),log_derivative_coefficients(f)


def b_series_residual(f:Sequence[int],b_hat:Sequence[int])->list[int]:
    if len(f)!=len(b_hat) or f[0]!=1 or b_hat[0]!=0:
        raise ValueError('invalid series')
    return [0]+[n*f[n]-sum(b_hat[k]*f[n-k] for k in range(1,n+1))
                for n in range(1,len(f))]


def first_nonzero(residual:Sequence[int])->int|None:
    return next((n for n in range(1,len(residual)) if residual[n]),None)


def certify(prefix:Sequence[int],a_hat:Sequence[int],b_hat:Sequence[int])->dict:
    """Standalone exact verifier. Verifies the entire offered trace and program.

    Important: this does not *repair* the neural model's output using Euler
    inversion. The unique exact teacher is separately called in scoring.
    """
    if len(prefix)!=PREFIX+1 or len(a_hat)!=len(prefix) or len(b_hat)!=len(prefix):
        raise ValueError('length mismatch')
    if a_hat[0]!=0 or b_hat[0]!=0 or any(not isinstance(x,int) for x in a_hat+b_hat):
        raise ValueError('expected integer proposed traces')
    f_res=coefficient_residual(prefix,a_hat)
    b_res=b_series_residual(prefix,b_hat)
    first_a=first_nonzero(f_res)
    first_b=first_nonzero(b_res)
    candidates=proposed_programs_from_euler(a_hat,prefix) if first_a is None else ()
    program=candidates[0] if candidates else None
    return {'first_exponent_series_residual':first_a,
            'first_log_derivative_residual':first_b,
            'verified_exponent_trace':first_a is None,
            'verified_b_trace':first_b is None,
            'verified_joint_trace':first_a is None and first_b is None,
            'program':program}


def rounded_predictions(logits:Sequence[Sequence[float]])->tuple[list[int],list[int]]:
    """Decode *only* neural outputs. Channels are 3 Euler class scores + y_b.

    y_b=asinh((b_n/n))/3 is a learned numeric target; inversion makes
    integer candidate b via rounding, without exact algebraic correction.
    """
    if len(logits)!=PREFIX+1 or any(len(row)!=4 for row in logits):
        raise ValueError('expected 65 by 4 logits')
    a=[0]; b=[0]
    for n in range(1,PREFIX+1):
        row=logits[n]
        if any(not math.isfinite(float(z)) for z in row):
            raise ValueError('nonfinite output')
        a.append(max(range(3),key=lambda c:row[c])-1)
        x=max(-15.,min(15.,float(row[3])))
        b.append(round(n*math.sinh(3*x)))
    return a,b


def score_case(prefix:Sequence[int],full:Sequence[int],logits:Sequence[Sequence[float]])->dict:
    a,b=rounded_predictions(logits)
    cert=certify(prefix,a,b)
    exact_a,exact_b=teacher(prefix) # evaluation only, never model input
    p=cert['program']
    future=(product_coefficients(HORIZON,p)==list(full)) if p else False
    return {'exact_a_coordinates':sum(x==y for x,y in zip(a[1:],exact_a[1:])),
            'exact_b_coordinates':sum(x==y for x,y in zip(b[1:],exact_b[1:])),
            'first_a_error':next((n for n in range(1,PREFIX+1) if a[n]!=exact_a[n]),None),
            'first_b_error':next((n for n in range(1,PREFIX+1) if b[n]!=exact_b[n]),None),
            'first_a_residual':cert['first_exponent_series_residual'],
            'first_b_residual':cert['first_log_derivative_residual'],
            'prefix_verified':int(p is not None),
            'joint_trace_verified':int(cert['verified_joint_trace']),
            'correct_future':int(future),
            'false_accept':int(p is not None and not future),
            'program':p.as_dict() if p else None}
