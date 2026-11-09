"""RRC-05: exact formal-series residuals and a depth-bounded divisor-graph solver.

All oracles use exact integers/Fractions. Learned gains are an explicitly
algebra-assisted comparator, not end-to-end neural discovery from raw coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
import math
import random
from typing import Sequence

try:
    from .program04 import arbitrary_part_coefficients, gap_coefficients, late_defect, recover_allowed, brute_single
    from .rrc import euler_exponents
except ImportError:
    from program04 import arbitrary_part_coefficients, gap_coefficients, late_defect, recover_allowed, brute_single
    from rrc import euler_exponents

PREFIX = 64
HORIZON = 128
PERIOD_CAP = 20
CAPS = (0, 1, 2)
TRAIN_PERIODS = (2, 3, 4, 6, 8, 9, 10, 12)
TEST_PERIODS = (5, 7, 11, 13, 17, 19)
SEED = 2026110905


@dataclass(frozen=True)
class Program05:
    cap: int
    period: int
    residues: tuple[int, ...]

    def __post_init__(self):
        if self.cap not in CAPS or not 1 <= self.period <= PERIOD_CAP:
            raise ValueError('cap/period outside registered grammar')
        if self.residues != tuple(sorted(set(self.residues))) or any(
                r < 0 or r >= self.period for r in self.residues):
            raise ValueError('invalid residue mask')

    @property
    def syntax_bits(self):
        return 2 + 2 * (self.period.bit_length() - 1) + 1 + self.period

    @property
    def key(self):
        return self.syntax_bits, self.cap, self.period, self.residues

    def as_dict(self):
        return {'cap': self.cap, 'period': self.period,
                'residues': list(self.residues), 'syntax_bits': self.syntax_bits}


def product_coefficients(degree: int, program: Program05) -> list[int]:
    """Coefficient oracle by finite polynomial coin multiplication."""
    if degree < 0:
        raise ValueError('negative degree')
    out = [1] + [0] * degree
    allowed = set(program.residues)
    for part in range(1, degree + 1):
        if part % program.period not in allowed:
            continue
        if program.cap == 0:
            for n in range(part, degree + 1):
                out[n] += out[n - part]
        else:
            old = out.copy()
            for n in range(degree + 1):
                for k in range(1, program.cap + 1):
                    if n + k * part <= degree:
                        out[n + k * part] += old[n]
    return out


def log_derivative_coefficients(f: Sequence[int]) -> list[int]:
    """Solve n f[n] = sum_{k=1}^n b[k] f[n-k] exactly, f[0]=1."""
    if not f or f[0] != 1 or any(not isinstance(x, int) for x in f):
        raise ValueError('integer series with f[0]=1 required')
    b = [0] * len(f)
    for n in range(1, len(f)):
        b[n] = n * f[n] - sum(b[k] * f[n-k] for k in range(1, n))
    return b


@lru_cache(None)
def proper_divisors(degree: int) -> tuple[tuple[int, ...], ...]:
    if degree < 0:
        raise ValueError('negative degree')
    return tuple(tuple(d for d in range(1, n) if n % d == 0)
                 for n in range(degree+1))


def divisor_residual(b: Sequence[int], a: Sequence[int]) -> list[int]:
    if len(b) != len(a) or len(b) < 2:
        raise ValueError('invalid coefficient lengths')
    degree = len(b) - 1
    proper = proper_divisors(degree)
    return [0] + [b[n] - n*a[n] - sum(d*a[d] for d in proper[n])
                  for n in range(1, degree+1)]


def coefficient_residual(f: Sequence[int], a: Sequence[int]) -> list[int]:
    """Exact qF'-F*B residual; zero iff proposed Euler coefficients reproduce f."""
    if len(f) != len(a) or f[0] != 1:
        raise ValueError('invalid series')
    degree = len(f) - 1
    bprime = [0]*(degree+1)
    for n in range(1, degree+1):
        for m in range(n, degree+1, n):
            bprime[m] += n*a[n]
    return [0]+[n*f[n]-sum(bprime[k]*f[n-k] for k in range(1,n+1))
                for n in range(1,degree+1)]


def jacobi_exact(b: Sequence[int], passes: int) -> list[Fraction]:
    """Simultaneous c-Pa updates, initially zero; exact after floor(log2 N)+1."""
    if passes < 0 or len(b) < 2:
        raise ValueError('invalid iteration budget')
    N = len(b)-1
    proper = proper_divisors(N)
    x = [Fraction(0)] * (N+1)
    for _ in range(passes):
        x = [Fraction(0)] + [Fraction(b[n],n) - sum(
            (Fraction(d,n)*x[d] for d in proper[n]), Fraction(0))
            for n in range(1,N+1)]
    return x


def learned_iteration(b: Sequence[int], passes: int, alpha: float,
                      beta: float = 1., *, erase_proper: bool = False) -> list[float]:
    """Two shared learned gains, fed exact log-derivative input b.

    x[t+1] = x[t] + alpha * (c - x[t] - beta P x[t]); x[0]=0.
    Assumed public divisor graph is supplied free of model weights, counted
    separately as a deterministic algebraic preprocessor.
    """
    if passes < 0 or len(b) < 2 or not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError('invalid model input')
    N = len(b)-1
    proper = proper_divisors(N)
    x = [0.]*(N+1)
    for _ in range(passes):
        y = [0.]*(N+1)
        for n in range(1,N+1):
            px = 0. if erase_proper else sum(d*x[d] for d in proper[n])/n
            y[n] = x[n] + alpha*(b[n]/n - x[n] - beta*px)
        x = y
    return x


def feedforward_estimate(b: Sequence[int], gain: float, bend: float) -> list[float]:
    """Two-parameter, one-pass comparator with the same b/n input."""
    return [0.] + [gain*(b[n]/n) + bend*math.tanh(b[n]/n)
                   for n in range(1,len(b))]


def quantize(x: Sequence[float], *, tolerance: float | None = None) -> list[int] | None:
    """Nearest-integer candidates. Optional trust-region threshold learned on dev only."""
    if not x or any(not math.isfinite(float(v)) for v in x):
        return None
    out = [0]+[int(round(float(v))) for v in x[1:]]
    if tolerance is not None and any(abs(float(v)-a)>tolerance
                                     for v,a in zip(x[1:],out[1:])):
        return None
    return out


def proposed_programs_from_euler(a: Sequence[int], prefix: Sequence[int],
                                  *, period_cap: int = PERIOD_CAP) -> tuple[Program05,...]:
    """Use *proposed* integer Euler exponents, with mandatory independent execution.

    Neither raw-series Euler inversion nor hidden source program is accessed here.
    Algebraic residual is audited separately and can be used as an abstention gate.
    """
    if len(a)!=len(prefix) or a[0] != 0 or prefix[0] != 1:
        raise ValueError('invalid inputs')
    if len(a)-1 < 2*period_cap:
        raise ValueError('insufficient prefix for two residue observations')
    accepted=[]
    for cap in CAPS:
        membership = recover_allowed(a,cap)
        if membership is None:
            continue
        for period in range(1,period_cap+1):
            groups=[[membership[n] for n in range(1,len(membership))
                     if n%period == r] for r in range(period)]
            if any(len(v)<2 or len(set(v))!=1 for v in groups):
                continue
            p=Program05(cap,period,tuple(r for r,group in enumerate(groups) if group[0]))
            if product_coefficients(len(prefix)-1,p)==list(prefix):
                accepted.append(p)
    return tuple(sorted(set(accepted),key=lambda p:p.key))


def exact_program(prefix: Sequence[int]) -> Program05|None:
    a = euler_exponents(prefix)
    return next(iter(proposed_programs_from_euler(a,prefix)),None)


def candidate_from_estimate(prefix: Sequence[int], b: Sequence[int],
                            estimation: Sequence[float],
                            *, tolerance: float|None = None) -> dict:
    """Provide exact algebraic residual certificate and executable grammar check."""
    a=quantize(estimation,tolerance=tolerance)
    if a is None:
        return {'program':None,'first_divisor_residual':None,
                'first_series_residual':None,'max_integer_rounding_error':None,
                'exact_intermediate':False}
    div=divisor_residual(b,a)
    series=coefficient_residual(prefix,a)
    first_d=next((i for i in range(1,len(div)) if div[i]),None)
    first_s=next((i for i in range(1,len(series)) if series[i]),None)
    programs=proposed_programs_from_euler(a,prefix) if first_s is None else ()
    return {'program':programs[0] if programs else None,
            'first_divisor_residual':first_d,'first_series_residual':first_s,
            'max_integer_rounding_error':round(max(abs(float(v)-z)
                        for v,z in zip(estimation,a)),9),
            'exact_intermediate':first_d is None and first_s is None}


@dataclass(frozen=True)
class Case05:
    name: str
    family: str
    split: str
    prefix: tuple[int,...]
    full: tuple[int,...]
    source: str
    collision_group: str|None = None

    def prefix_hash(self):
        return sha256(json.dumps(list(self.prefix),separators=(',',':')).encode()).hexdigest()


def random_program(rng: random.Random, periods: Sequence[int]) -> Program05:
    cap=rng.choice(CAPS)
    period=rng.choice(periods)
    density=rng.choice((.20,.35,.50,.65,.80))
    residues=tuple(i for i in range(period) if rng.random()<density)
    if not residues: residues=(rng.randrange(period),)
    return Program05(cap,period,tuple(sorted(residues)))


def roster(*, train_size: int = 384, dev_size: int = 96,
           unseen_per_period: int = 20, include_test: bool = True) -> dict[str,list[Case05]]:
    """Data split and fixed seeds. Test families are never generated in train05."""
    if min(train_size,dev_size,unseen_per_period) < 1:
        raise ValueError('empty split')
    out={label:[] for label in ('train','dev','unseen_modulus','local_gap',
                                'nonperiodic','collision','out_of_grammar')}
    seen=set()
    source_splits=[
        ('train',train_size,SEED+101,TRAIN_PERIODS,PREFIX),
        ('dev',dev_size,SEED+202,TRAIN_PERIODS,HORIZON),
    ]
    if include_test:
        source_splits.append(('unseen_modulus',unseen_per_period*len(TEST_PERIODS),
                              SEED+303,TEST_PERIODS,HORIZON))
    for label,size,seed,periods,degree in source_splits:
        rng=random.Random(seed)
        attempts=0
        while len(out[label])<size:
            attempts+=1
            if attempts>500*size:
                raise AssertionError('unique-prefix generation exhausted')
            p=random_program(rng,periods)
            full=tuple(product_coefficients(degree,p))
            prefix=full[:PREFIX+1]
            if prefix in seen:continue
            seen.add(prefix)
            name=f'{label}-{len(out[label]):04d}'
            out[label].append(Case05(name,'periodic_product',label,prefix,full,repr(p)))
    if not include_test:
        return out
    # Held-out local-gap family. Gap=2/min=1,2 are the RR positive controls.
    for gap in range(1,6):
        for minimum in range(1,9):
            full=tuple(gap_coefficients(HORIZON,gap=gap,minimum_part=minimum))
            out['local_gap'].append(Case05(f'gap{gap}-min{minimum}',
                'local_gap','local_gap',full[:PREFIX+1],full,f'gap({gap},{minimum})'))
    def omega(n:int)->int:
        cnt=0
        d=2
        while d*d<=n:
            while n%d==0:
                cnt+=1; n//=d
            d+=1
        return cnt+int(n>1)
    sets={
        'pentagonal': [k*(3*k-1)//2 for k in range(1,18)],
        'centered_hexagonal': [3*k*(k-1)+1 for k in range(1,9)],
        'cubes': [k**3 for k in range(1,6)],
        'semiprimes': [n for n in range(2,HORIZON+1) if omega(n)==2],
        'primes_1_mod4': [n for n in range(2,HORIZON+1)
                          if n%4==1 and all(n%d for d in range(2,math.isqrt(n)+1))]
    }
    for name,allowed in sets.items():
        for cap in CAPS:
            full=tuple(arbitrary_part_coefficients(HORIZON,allowed,cap))
            out['nonperiodic'].append(Case05(f'{name}-cap{cap}','nonperiodic',
                'nonperiodic',full[:PREFIX+1],full,name+f' cap={cap}'))
    rng=random.Random(SEED+505)
    # Grammar outside the declared period cap; report any accidental finite aliases.
    for i in range(18):
        cap=rng.choice(CAPS)
        # Construct a period >20 without granting it to the bounded interpreter.
        period=(23,29,31)[i%3]
        density=(.2,.4,.6)[i%3]
        mask=tuple(k for k in range(period) if rng.random()<density)
        if not mask:mask=(1,)
        allowed=[j for j in range(1,HORIZON+1) if j%period in mask]
        full=tuple(arbitrary_part_coefficients(HORIZON,allowed,cap))
        out['out_of_grammar'].append(Case05(f'out-{i:03d}',
            'out_of_grammar','out_of_grammar',full[:PREFIX+1],full,
            f'cap={cap};period={period};mask={mask}'))
    base_masks=((1,4),(1,2,4),(1,6,9),(1,4,7),(2,3,8),(3,5,11))
    ks=(65,77,89,105,116,127)
    for i,(mask,k) in enumerate(zip(base_masks,ks)):
        p=Program05(0,13,tuple(sorted(mask)))
        base=tuple(product_coefficients(HORIZON,p))
        mutated=tuple(late_defect(base,k))
        assert base[:PREFIX+1]==mutated[:PREFIX+1]
        for label,vec in (('base',base),('mutant',mutated)):
            out['collision'].append(Case05(f'pair{i}-{label}',
                'collision','collision',vec[:PREFIX+1],vec,
                f'{p}, {label}, new factor at {k}',f'pair{i}'))
    train_sigs={x.prefix_hash() for x in out['train']}
    for label in ('dev','unseen_modulus'):
        if train_sigs.intersection(x.prefix_hash() for x in out[label]):
            raise AssertionError('cross-split prefix leakage')
    return out


def roster_digest(cases: Sequence[Case05]) -> str:
    return sha256(json.dumps([(x.name,x.source,x.prefix_hash(),x.collision_group)
                for x in cases],separators=(',',':'),sort_keys=True).encode()).hexdigest()
