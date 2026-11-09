"""RRC-04: exact bounded-multiplicity product program synthesis from coefficients.

Only the finite coefficients of a formal power series are available to synthesis.
The identity of the source generator, future coefficients and neural weights are
absent from the solver. Uses stdlib integer arithmetic and the RRC-01 Euler inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Sequence

try:
    from .rrc import euler_exponents, gap_coefficients
except ImportError:
    from rrc import euler_exponents, gap_coefficients

HERE = Path(__file__).resolve().parent
PREFIX = 64
HORIZON = 128
MAX_PERIOD = 12
CAPS = (0, 1, 2)  # 0 means unlimited multiplicity
TRAIN_PERIODS = (2, 3, 4, 6, 8, 9, 10, 12)
TEST_PERIODS = (5, 7, 11)
SEED = 20261109


@dataclass(frozen=True)
class Program:
    cap: int
    period: int
    residues: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.cap not in CAPS or not 1 <= self.period <= MAX_PERIOD:
            raise ValueError('cap or period outside declared grammar')
        if tuple(sorted(set(self.residues))) != self.residues:
            raise ValueError('residues must be sorted, unique')
        if any(not 0 <= r < self.period for r in self.residues):
            raise ValueError('residue outside modulus')

    @property
    def bits(self) -> int:
        """2-bit cap tag, Elias gamma period, and a period-bit mask.

        Syntax-only code length; all common interpreter/search/parameter costs excluded.
        """
        return 2 + (2 * (self.period.bit_length()-1) + 1) + self.period

    @property
    def key(self) -> tuple:
        return (self.bits, self.cap, self.period, self.residues)

    def as_dict(self) -> dict:
        return {'cap':self.cap, 'period':self.period,
                'residues':list(self.residues), 'syntax_bits':self.bits}


def coefficients(degree: int, program: Program) -> list[int]:
    """Independent finite-polynomial multiplication for bounded/unbounded parts."""
    if degree < 0: raise ValueError('degree must be nonnegative')
    result=[1]+[0]*degree
    residues=set(program.residues)
    for part in range(1,degree+1):
        if part%program.period not in residues: continue
        if program.cap == 0:
            for n in range(part,degree+1):
                result[n] += result[n-part]
        else:
            old=result.copy()
            for n in range(degree+1):
                if old[n]:
                    for count in range(1,program.cap+1):
                        nxt=n+part*count
                        if nxt>degree: break
                        result[nxt] += old[n]
    return result


def arbitrary_part_coefficients(degree:int, allowed:Sequence[int], cap:int=0)->list[int]:
    """Separate generator for non-periodic controls, with explicit allowed-part list."""
    if cap not in CAPS or degree<0 or any(p<1 for p in allowed):
        raise ValueError('invalid arbitrary-part grammar')
    result=[1]+[0]*degree
    for part in sorted(set(allowed)):
        if part>degree: continue
        old=result.copy()
        if cap == 0:
            for n in range(part,degree+1): result[n] += result[n-part]
        else:
            for n,value in enumerate(old):
                if value:
                    for k in range(1,cap+1):
                        if n+k*part<=degree: result[n+k*part] += value
    return result


def late_defect(base:Sequence[int], degree:int)->list[int]:
    """Mutate F(q) to F(q)/(1-q^degree), preserving coefficients below degree."""
    if not 1<=degree<len(base): raise ValueError('defect out of range')
    return [sum(base[n-k*degree] for k in range(n//degree+1)) for n in range(len(base))]


def recover_allowed(exponents:Sequence[int], cap:int)->list[int]|None:
    """Euler inversion: e_n = A_n - [c+1 divides n] A_(n/(c+1))."""
    if cap not in CAPS: raise ValueError('invalid cap')
    membership=[0]*len(exponents)
    for n in range(1,len(exponents)):
        v=exponents[n]
        if cap and n%(cap+1)==0:
            v+=membership[n//(cap+1)]
        if v not in (0,1): return None
        membership[n]=v
    return membership


def fits_period(membership:Sequence[int],period:int)->tuple[int,...]|None:
    if not 1<=period<=MAX_PERIOD: raise ValueError('period out of range')
    classes=[[membership[n] for n in range(1,len(membership)) if n%period==r]
             for r in range(period)]
    if any(len(v)<2 or len(set(v))!=1 for v in classes): return None
    return tuple(r for r,row in enumerate(classes) if row[0])


def synthesize_all(prefix:Sequence[int],*, max_period:int=MAX_PERIOD)->tuple[Program,...]:
    """Exact finite-prefix program search; inspect no coefficients after T.

    The inverse Euler transform is unique to degree T. For any fixed cap,
    membership is then uniquely forced through T, so there are only 3*12
    cap/period skeleton checks. Every emitted candidate is also independently
    executed through T. This is an exact grammar-class solver, not a theorem
    about all possible mathematical generating functions.
    """
    if len(prefix)<2*max_period+1 or prefix[0]!=1 or max_period<1 or max_period>MAX_PERIOD:
        raise ValueError('need f0=1 and >=2 observed part values per residue')
    exponents=euler_exponents(prefix)
    candidates=[]
    for cap in CAPS:
        allowed=recover_allowed(exponents,cap)
        if allowed is None: continue
        for period in range(1,max_period+1):
            mask=fits_period(allowed,period)
            if mask is None: continue
            program=Program(cap,period,mask)
            if coefficients(len(prefix)-1,program)!=list(prefix):
                raise AssertionError('Euler inverse and direct execution disagree')
            candidates.append(program)
    return tuple(sorted(set(candidates),key=lambda p:p.key))


def synthesize(prefix:Sequence[int])->Program|None:
    return next(iter(synthesize_all(prefix)),None)


def verify_prefix(prefix:Sequence[int], proposed:Program)->bool:
    return coefficients(len(prefix)-1,proposed)==list(prefix)


def complete_skeleton(prefix:Sequence[int], cap:int, period:int)->Program|None:
    """The exact verifier can fill a neural cap/period *proposal* with a mask.

    This separate hybrid benchmark charges an Euler inverse and reconstructs
    the mask; it is not direct neural discovery. No held-out data are used.
    """
    if cap not in CAPS or not 1<=period<=MAX_PERIOD or prefix[0]!=1:
        raise ValueError('invalid proposed skeleton')
    membership=recover_allowed(euler_exponents(prefix),cap)
    if membership is None: return None
    mask=fits_period(membership,period)
    if mask is None: return None
    p=Program(cap,period,mask)
    return p if verify_prefix(prefix,p) else None


def simple_features(prefix:Sequence[int])->list[list[float]]:
    """Public, fixed-order feature scaling of raw counts, no Euler exponents."""
    if len(prefix)!=PREFIX+1 or prefix[0]!=1 or any(c<0 for c in prefix):
        raise ValueError('wrong raw prefix')
    logs=[math.log1p(v)/16.0 for v in prefix]
    return [[logs[i],(logs[i]-logs[i-1] if i else 0.0)*8.0,i/PREFIX]
            for i in range(PREFIX+1)]


@dataclass(frozen=True)
class Case:
    name:str
    split:str
    family:str
    prefix:tuple[int,...]
    full:tuple[int,...]
    source:str
    collision_group:str|None=None

    def signature(self):
        return sha256(json.dumps(list(self.prefix),separators=(',',':')).encode()).hexdigest()


def random_program(rng:random.Random, periods:Sequence[int])->Program:
    cap=rng.choice(CAPS)
    period=rng.choice(periods)
    # Vary density and require at least one allowed residue to avoid zero-programs.
    density=rng.choice((.20,.35,.50,.65,.80))
    mask=tuple(r for r in range(period) if rng.random()<density)
    if not mask: mask=(rng.randrange(period),)
    return Program(cap,period,mask)


def roster(*, train_size:int=1500, dev_size:int=180,
           unseen_size:int=108, include_test:bool=True)->dict[str,list[Case]]:
    """Frozen source-family split; only product families appear during training.

    Test gap, nonperiodic and defect generators are entirely excluded from
    optimization and model selection. Unseen moduli are held out, even though
    the product syntax is shared. All incidental prefix duplicates are rejected
    across the normal split; collision pairs are deliberate exceptions.
    """
    if min(train_size,dev_size,unseen_size)<1: raise ValueError('positive split sizes')
    out={k:[] for k in ('train','dev','unseen_modulus','local_gap','nonperiodic','collision')}
    seen=set()
    for split,target,seed,periods,degree in (
            ('train',train_size,SEED+101,TRAIN_PERIODS,PREFIX),
            ('dev',dev_size,SEED+202,TRAIN_PERIODS,HORIZON),
            ('unseen_modulus',unseen_size,SEED+303,TEST_PERIODS,HORIZON)):
        rng=random.Random(seed)
        attempts=0
        while len(out[split])<target:
            attempts+=1
            if attempts>target*350: raise AssertionError('could not draw unique prefixes')
            p=random_program(rng,periods)
            value=tuple(coefficients(degree,p))
            prefix=value[:PREFIX+1]
            sig=tuple(prefix)
            if sig in seen: continue
            seen.add(sig)
            idx=len(out[split])
            out[split].append(Case(f'{split}-{idx:04d}',split,
                f'product_cap{p.cap}',tuple(prefix),value,repr(p)))
    if not include_test:
        return out
    for gap in range(1,5):
        for minimum in range(1,9):
            full=tuple(gap_coefficients(HORIZON,gap=gap,minimum_part=minimum))
            out['local_gap'].append(Case(f'gap{gap}-min{minimum}','test','local_gap',
                full[:PREFIX+1],full,f'gap({gap},{minimum})'))

    def is_prime(n:int)->bool:
        return n>=2 and all(n%d for d in range(2,math.isqrt(n)+1))
    seqs={
        'squares': [j*j for j in range(1,12)],
        'triangular':[j*(j+1)//2 for j in range(1,17)],
        'primes':[n for n in range(1,HORIZON+1) if is_prime(n)],
        'powers2':[2**j for j in range(8)],
        'fibonacci':[1,2,3,5,8,13,21,34,55,89],
    }
    for name,seq in seqs.items():
        for cap in CAPS:
            full=tuple(arbitrary_part_coefficients(HORIZON,seq,cap))
            out['nonperiodic'].append(Case(f'{name}-cap{cap}','test','nonperiodic',
                full[:PREFIX+1],full,f'{name}(cap{cap})'))
    # Evaluate both members of an identical observed-prefix pair. Their heldout
    # f_K differ by f_0=1, making universal prefix-only certification impossible.
    base_masks=((1,4),(1,2),(1,3),(1,2,4),(1,),(1,3,4),(1,2,3),(1,2,3,4))
    defects=(65,73,87,96,103,112,119,127)
    for i,(mask,k) in enumerate(zip(base_masks,defects)):
        p=Program(0,5,mask)
        base=tuple(coefficients(HORIZON,p))
        mutant=tuple(late_defect(base,k))
        assert base[:PREFIX+1]==mutant[:PREFIX+1]
        for label,full in (('base',base),('mutant',mutant)):
            out['collision'].append(Case(f'pair{i}-{label}','test','collision',
                full[:PREFIX+1],full,f'{p};{label} at {k}',f'pair{i}'))
    # No accidental cross split prefix duplicates, aside from the explicit
    # controlled collisions. Some mathematical test families naturally overlap;
    # those are recorded as observational aliases, never train data.
    train_signatures={c.signature() for c in out['train']}
    for split in ('dev','unseen_modulus'):
        if train_signatures.intersection(c.signature() for c in out[split]):
            raise AssertionError('train/evaluation prefix overlap')
    return out


def roster_digest(cases:Sequence[Case])->str:
    rows=[(c.name,c.split,c.family,c.signature(),c.source,c.collision_group) for c in cases]
    return sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def brute_single(n:int, parts:Sequence[int],cap:int)->int:
    """Small independent recursive enumeration for tests (no polynomial DP)."""
    allowed=tuple(sorted(set(p for p in parts if p<=n)))
    @lru_cache(None)
    def visit(remaining:int,index:int)->int:
        if remaining==0:return 1
        if index>=len(allowed): return 0
        p=allowed[index]
        upper=remaining//p if cap==0 else min(cap,remaining//p)
        return sum(visit(remaining-k*p,index+1) for k in range(upper+1))
    return visit(n,0)
