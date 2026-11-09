"""RRC-08: finite symbolic circuit induction from raw generating-series coefficients.

A learner fits eight operator coefficients from training traces. Inference gets
f_0..f_T and the *learned* integer circuit only; it never calls an Euler oracle.
The public basis/loops are a substantial architectural prior, charged as code.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import isqrt
import json
import random
from typing import Sequence

try:
    from .rrc05 import (Program05, product_coefficients, coefficient_residual,
                        proper_divisors, proposed_programs_from_euler,
                        log_derivative_coefficients)
    from .rrc import euler_exponents, gap_coefficients
    from .program04 import arbitrary_part_coefficients, late_defect
except ImportError:
    from rrc05 import (Program05, product_coefficients, coefficient_residual,
                       proper_divisors, proposed_programs_from_euler,
                       log_derivative_coefficients)
    from rrc import euler_exponents, gap_coefficients
    from program04 import arbitrary_part_coefficients, late_defect

PREFIX = 64
HORIZON = 128
SEED = 2026110908
TRAIN_PERIODS = (2, 3, 4, 6, 8, 9, 10, 12)
TEST_PERIODS = (5, 7, 11, 13, 17, 19)
SPLIT_SIZES = {'train':192,'dev':40,'unseen_period':60,'power':12,
               'inverse':12,'signed_twist':12,'gap':20,'irregular':15,
               'out_of_grammar':12,'collision':12}
COEFF_LIMIT = 2


@dataclass(frozen=True)
class Operator08:
    """Four integer coefficients per update, over supplied public term basis.

    b[n] = b0*n*f[n] + b1*sum_{k<n} b[k]f[n-k] + b2*b[n-1] + b3
    a[n] = (a0*b[n] + a1*sum_{d|n,d<n}d*a[d] + n*a2*a[n-1] + n*a3)/n
    """
    b_terms: tuple[int, int, int, int]
    a_terms: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.b_terms) != 4 or len(self.a_terms) != 4:
            raise ValueError('exactly eight coefficients')
        if any(type(v) is not int or abs(v)>COEFF_LIMIT for v in self.b_terms+self.a_terms):
            raise ValueError('integer coefficients must lie in [-2,2]')

    @property
    def syntax_bits(self) -> int:
        # 3 bits per signed coefficient in a fixed bounded vocabulary; decoder omitted.
        return 8*3

    def as_dict(self) -> dict:
        return {'b_terms':list(self.b_terms),'a_terms':list(self.a_terms),
                'syntax_bits_excluding_basis_interpreter':self.syntax_bits}


@dataclass(frozen=True)
class Case08:
    id: str
    split: str
    family: str
    prefix: tuple[int,...]
    full: tuple[int,...]
    origin: str
    pair: str|None = None


def _validate(f:Sequence[int]) -> None:
    if not f or len(f)>HORIZON+1 or type(f[0]) is not int or f[0]!=1 or any(
            type(v) is not int for v in f):
        raise ValueError('integer formal series with f0=1 required')


def serial_circuit(f:Sequence[int],operator:Operator08) -> tuple[list[int],list[int]]|None:
    """Execute learned operator in increasing degree. No source labels/oracles.

    Any nonintegral proposed step fails closed. This is a candidate interpreter,
    not an exact-correction call: incorrect weights stay incorrect/rejected.
    """
    _validate(f)
    degree=len(f)-1
    b=[0]*(degree+1);a=[0]*(degree+1)
    divisor=proper_divisors(degree)
    x,y,z,c=operator.b_terms
    p,q,r,s=operator.a_terms
    for n in range(1,degree+1):
        cross=sum(b[k]*f[n-k] for k in range(1,n))
        b[n]=x*n*f[n]+y*cross+z*b[n-1]+c
        divisor_sum=sum(d*a[d] for d in divisor[n])
        numerator=p*b[n]+q*divisor_sum+n*(r*a[n-1]+s)
        if numerator % n:
            return None
        a[n]=numerator//n
    return b,a


def parallel_circuit(f:Sequence[int],operator:Operator08,passes:int)->tuple[list[int],list[Fraction]]:
    """Exact simultaneous Jacobi-style depth ablation; no sequential shortcut."""
    _validate(f)
    if passes<0 or passes>HORIZON+1:raise ValueError('unregistered passes')
    degree=len(f)-1;divisor=proper_divisors(degree)
    b=[0]*(degree+1);a=[Fraction(0)]*(degree+1)
    x,y,z,c=operator.b_terms;p,q,r,s=operator.a_terms
    for _ in range(passes):
        new_b=[0]+[x*n*f[n]+y*sum(b[k]*f[n-k] for k in range(1,n))+
                   z*b[n-1]+c for n in range(1,degree+1)]
        new_a=[Fraction(0)]+[Fraction(p*b[n]+q*sum(d*a[d] for d in divisor[n]),n)+
                 r*a[n-1]+s for n in range(1,degree+1)]
        b,a=new_b,new_a
    return b,a


def parallel_snapshots(f:Sequence[int],operator:Operator08,
                       depths:Sequence[int])->dict[int,tuple[list[int],list[Fraction]]]:
    """Cache one actual simultaneous trajectory for repeated depth ablations."""
    _validate(f)
    if not depths or any(type(k) is not int or k<0 or k>HORIZON+1 for k in depths):
        raise ValueError('invalid registered depth roster')
    limit=max(depths);needed=set(depths)
    degree=len(f)-1;divisor=proper_divisors(degree)
    b=[0]*(degree+1);a=[Fraction(0)]*(degree+1)
    snapshots={}
    if 0 in needed:snapshots[0]=(b.copy(),a.copy())
    x,y,z,c=operator.b_terms;p,q,r,s=operator.a_terms
    for depth in range(1,limit+1):
        next_b=[0]+[x*n*f[n]+y*sum(b[k]*f[n-k] for k in range(1,n))+
                    z*b[n-1]+c for n in range(1,degree+1)]
        next_a=[Fraction(0)]+[Fraction(p*b[n]+q*sum(d*a[d] for d in divisor[n]),n)+
                   r*a[n-1]+s for n in range(1,degree+1)]
        b,a=next_b,next_a
        if depth in needed:snapshots[depth]=(b.copy(),a.copy())
    return snapshots


def coefficient_first_fault(f:Sequence[int],a:Sequence[int]) -> int|None:
    if len(f)!=len(a):raise ValueError('series length mismatch')
    if any(isinstance(v,Fraction) and v.denominator!=1 for v in a):
        # Earliest nonintegral degree is itself a failed program step.
        return next(n for n in range(1,len(a)) if isinstance(a[n],Fraction) and a[n].denominator!=1)
    r=coefficient_residual(f,[int(v) for v in a])
    return next((n for n in range(1,len(r)) if r[n]),None)


def propose_program(f:Sequence[int],a:Sequence[int]) -> Program05|None:
    _validate(f)
    if len(a)!=len(f) or a[0]!=0 or any(type(v) is not int for v in a):return None
    if coefficient_first_fault(f,a) is not None:return None
    return next(iter(proposed_programs_from_euler(a,f)),None)


def score_trace(case:Case08,trace:tuple[list[int],list[int]]|None) -> dict:
    exact_a=euler_exponents(case.prefix) # only evaluation, not used to construct trace
    if trace is None:
        return {'integral':0,'exact_coordinates':0,'first_error':1,
                'verified_trace':0,'accepted_program':0,'correct_future':0,
                'false_accept':0,'certificate':None}
    b,a=trace
    matched=sum(x==y for x,y in zip(a[1:],exact_a[1:]))
    fault=coefficient_first_fault(case.prefix,a)
    bad_b=next((n for n in range(1,PREFIX+1) if
         n*case.prefix[n]!=sum(b[k]*case.prefix[n-k] for k in range(1,n+1))),None)
    p=propose_program(case.prefix,a) if fault is None and bad_b is None else None
    future=int(p is not None and tuple(product_coefficients(HORIZON,p))==case.full)
    return {'integral':1,'exact_coordinates':matched,
            'first_error':next((n for n in range(1,PREFIX+1) if a[n]!=exact_a[n]),None),
            'verified_trace':int(fault is None and bad_b is None),
            'first_b_residual':bad_b,'accepted_program':int(p is not None),
            'correct_future':future,'false_accept':int(p is not None)-future,
            'certificate':p.as_dict() if p else None}


def score(case:Case08,operator:Operator08) -> dict:
    return score_trace(case,serial_circuit(case.prefix,operator))


def _mul_series(a:Sequence[int],b:Sequence[int],degree:int=HORIZON)->list[int]:
    out=[0]*(degree+1)
    for i,x in enumerate(a[:degree+1]):
        if x:
            for j,y in enumerate(b[:degree+1-i]):
                if y:out[i+j]+=x*y
    return out


def _reciprocal_series(f:Sequence[int])->list[int]:
    _validate(f)
    g=[1]+[0]*(len(f)-1)
    for n in range(1,len(f)):
        g[n]=-sum(f[k]*g[n-k] for k in range(1,n+1))
    return g


def _fresh_program(rng:random.Random,periods:Sequence[int])->Program05:
    mod=rng.choice(periods);cap=rng.choice((0,1,2));density=rng.choice((.3,.45,.6,.75))
    res=tuple(k for k in range(mod) if rng.random()<density)
    if not res:res=(rng.randrange(mod),)
    return Program05(cap,mod,res)


def roster(*,include_test:bool=True,train_size:int=SPLIT_SIZES['train'],
           dev_size:int=SPLIT_SIZES['dev'],unseen_per_period:int=10)->dict[str,list[Case08]]:
    if min(train_size,dev_size,unseen_per_period)<1:raise ValueError('positive roster sizes')
    out={name:[] for name in SPLIT_SIZES}
    seen=set()
    def insert(split,full,origin,family=None,pair=None,unique=True):
        full=tuple(full)
        if len(full)!=HORIZON+1 and split!='train':raise AssertionError('missing horizon')
        p=full[:PREFIX+1]
        if unique and p in seen:return False
        seen.add(p)
        out[split].append(Case08(f'{split}-{len(out[split]):03d}',split,
                                   family or split,tuple(p),full,origin,pair))
        return True
    def draw(split,count,seed,periods):
        rng=random.Random(seed);attempts=0
        while len(out[split])<count:
            attempts+=1
            if attempts>count*1000:raise RuntimeError('unique prefix exhaustion '+split)
            prog=_fresh_program(rng,periods)
            full=product_coefficients(HORIZON,prog)
            insert(split,full,repr(prog),family='periodic_product')
    draw('train',train_size,SEED+11,TRAIN_PERIODS)
    draw('dev',dev_size,SEED+22,TRAIN_PERIODS)
    if not include_test:return out
    draw('unseen_period',len(TEST_PERIODS)*unseen_per_period,SEED+33,TEST_PERIODS)
    rng=random.Random(SEED+44)
    for family,transform,target in (
            ('power',lambda x:_mul_series(x,x),SPLIT_SIZES['power']),
            ('inverse',_reciprocal_series,SPLIT_SIZES['inverse']),
            ('signed_twist',lambda x:[v*(-1)**n for n,v in enumerate(x)],SPLIT_SIZES['signed_twist'])):
        attempts=0
        while len(out[family])<target:
            attempts+=1
            if attempts>target*400:raise RuntimeError('transformation exhaustion')
            src=_fresh_program(rng,(5,7,11,13,17))
            full=transform(product_coefficients(HORIZON,src))
            insert(family,full,f'{family}({src!r})',family=family)
    for gap in range(1,5):
        for minimum in range(1,6):
            insert('gap',gap_coefficients(HORIZON,gap=gap,minimum_part=minimum),
                   f'gap({gap},{minimum})',family='local_gap',unique=False)
    def is_prime(n):return n>=2 and all(n%k for k in range(2,isqrt(n)+1))
    allowed={
       'squares':[n*n for n in range(1,12)],
       'primes':[n for n in range(1,HORIZON+1) if is_prime(n)],
       'triangular':[n*(n+1)//2 for n in range(1,17)],
       'powers2':[2**n for n in range(8)],
       'fibonacci':[1,2,3,5,8,13,21,34,55,89],
    }
    for family,part_set in allowed.items():
        for cap in (0,1,2):
            insert('irregular',arbitrary_part_coefficients(HORIZON,part_set,cap),
                   f'{family}:cap{cap}',family='nonperiodic',unique=False)
    out_rng=random.Random(SEED+55)
    attempts=0
    while len(out['out_of_grammar'])<SPLIT_SIZES['out_of_grammar']:
        attempts+=1
        if attempts>10000:raise RuntimeError('out-of-grammar exhaustion')
        mod=out_rng.choice((23,29,31));cap=out_rng.choice((0,1,2))
        mask=tuple(r for r in range(mod) if out_rng.random()<.5)
        if not mask:mask=(out_rng.randrange(mod),)
        parts=[n for n in range(1,HORIZON+1) if n%mod in mask]
        insert('out_of_grammar',arbitrary_part_coefficients(HORIZON,parts,cap),
               f'period={mod}:cap={cap}:mask={mask}',family='large_period')
    for k in (65,73,87,99,111,127):
        src=_fresh_program(rng,(5,7,11,13))
        base=product_coefficients(HORIZON,src)
        mutant=late_defect(base,k)
        group=f'late-{k}'
        insert('collision',base,f'{src!r} base',family='collision',pair=group,unique=False)
        insert('collision',mutant,f'{src!r} mutant@{k}',family='collision',pair=group,unique=False)
    return out


def roster_hash(cases:Sequence[Case08])->str:
    rows=[{'id':c.id,'split':c.split,'family':c.family,'prefix':c.prefix,
           'full_sha256':sha256(json.dumps(c.full,separators=(',',':')).encode()).hexdigest(),
           'origin':c.origin,'pair':c.pair} for c in cases]
    return sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sample_features(train:Sequence[Case08],train_degree:int=16) -> tuple[list,list,list,list]:
    """Training-only exact intermediate supervision, unit-normalized by row.

    This is a strong teaching signal. At inference the trace is constructed
    solely from raw coefficients and the fitted integer operator.
    """
    if not 1<=train_degree<=PREFIX:raise ValueError('invalid train degree')
    xb=[];yb=[];xa=[];ya=[]
    divisors=proper_divisors(PREFIX)
    for c in train:
        f=c.prefix
        b=log_derivative_coefficients(f)
        a=euler_exponents(f)
        for n in range(1,train_degree+1):
            cross=sum(b[k]*f[n-k] for k in range(1,n))
            dsum=sum(d*a[d] for d in divisors[n])
            sb=(n*f[n],cross,b[n-1],1)
            sa=(b[n]/n,dsum/n,a[n-1],1)
            norm_b=max(1,*map(abs,sb));norm_a=max(1,*map(abs,sa))
            xb.append([float(z/norm_b) for z in sb]);yb.append(float(b[n]/norm_b))
            xa.append([float(z/norm_a) for z in sa]);ya.append(float(a[n]/norm_a))
    return xb,yb,xa,ya
