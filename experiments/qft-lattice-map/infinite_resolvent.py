"""Finite witnesses and exact-integer certificates for an infinite Dirac resolvent.

For the Shao--Sopenko free Dirac h(k), choose k=pi/2, z=i*y (y>0).
Every matrix element is a rational linear combination of pi and 1/pi:
  H_nm = pi*(n+1/2) delta_nm - 4i*(-1)^n/(pi*(m-n)^2)
for odd m-n, with zero off-diagonal elements otherwise.

All certificate VALIDATION uses integer directed rounding, an exact rational
Machin-series enclosure for pi, and an analytic infinite tail estimate. The
initial candidate solutions can come from non-verified numpy computations;
validation trusts only their *quantized integer bytes*, never the solver.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from math import isqrt, pi
from pathlib import Path

import numpy as np


PI_SCALE = 10**20
RESIDUAL_SCALE = 10**18
VECTOR_SCALE = 10**12
SCHEMA = 'qft-lattice-infinite-resolvent-v1'


def ceildiv(a: int, b: int) -> int:
    assert b > 0
    return -((-a)//b)


def sqrt_ceil(n: int) -> int:
    assert n >= 0
    x = isqrt(n)
    return x if x*x == n else x+1


def arctan_partial(inv_x: int, count: int) -> Fraction:
    assert inv_x > 1 and count > 0
    return sum((Fraction(1, inv_x**(2*j+1)*(2*j+1)) if j%2==0 else
                Fraction(-1, inv_x**(2*j+1)*(2*j+1))) for j in range(count))


def machin_pi_interval(scale: int = PI_SCALE, even_terms: int = 14) -> tuple[int,int]:
    """Return exact enclosure pi in [lo/scale, hi/scale].

    Machin identity: pi = 16 atan(1/5) - 4 atan(1/239).
    Alternating Taylor sums with even count bound atan from below; odd
    count from above. Outward-directed integer rounding finishes the proof.
    """
    assert even_terms >= 2 and even_terms % 2 == 0
    lo = 16*arctan_partial(5, even_terms) - 4*arctan_partial(239, even_terms+1)
    hi = 16*arctan_partial(5, even_terms+1) - 4*arctan_partial(239, even_terms)
    assert 3 < lo < hi < Fraction(22,7)
    return (lo.numerator*scale//lo.denominator,
            ceildiv(hi.numerator*scale, hi.denominator))


def inverse_pi_interval(scale: int = PI_SCALE) -> tuple[int,int]:
    lower, upper = machin_pi_interval(scale)
    return scale**2//upper, ceildiv(scale**2, lower)


def h_pi_half(m: int) -> np.ndarray:
    """Finite section of the *exactly simplified* k=pi/2 matrix."""
    if m < 2 or m % 2:
        raise ValueError('m must be positive and even')
    n = np.arange(-m//2, m//2, dtype=int)
    d = n[None,:]-n[:,None]
    off = np.zeros((m,m),dtype=complex)
    mask = (d % 2 != 0)
    off[mask] = (-4j/pi)*(np.where(n%2==0,1,-1)[:,None]
                      * np.divide(1.,d*d,out=np.zeros((m,m)),where=d!=0))[mask]
    return np.diag(pi*(n+.5)).astype(complex)+off


def resolvent_witness(m: int, y: int = 2, core: tuple[int,...] = (-1,0)) -> list[list[list[int]]]:
    """Generate an *untrusted* approximate inverse solution, then quantize."""
    if not 0 < y or not core or len(set(core)) != len(core):
        raise ValueError('invalid spectral point or repeated core')
    n = list(range(-m//2,m//2))
    if not set(core).issubset(set(n)):
        raise ValueError('core falls outside retained modes')
    F = np.zeros((m,len(core)), complex)
    for c,f in enumerate(core): F[n.index(f),c]=1
    U = np.linalg.solve(h_pi_half(m)-1j*y*np.eye(m), F)
    return [[[int(round(float(z.real)*VECTOR_SCALE)),
              int(round(float(z.imag)*VECTOR_SCALE))]
             for z in row] for row in U]


def product_interval(coefficient: int, denominator: int, positive_interval: tuple[int,int],
                     input_scale: int = VECTOR_SCALE) -> tuple[int,int]:
    """Outward-rounded interval for (coefficient/input_scale/denom)*x.

    x is in [lo/PI_SCALE,hi/PI_SCALE], result is in units 1/RESIDUAL_SCALE.
    """
    assert denominator > 0 and input_scale > 0
    xlo,xhi = positive_interval
    p,q = (coefficient*xlo,coefficient*xhi)
    if p > q: p,q=q,p
    den = input_scale*denominator*PI_SCALE
    return p*RESIDUAL_SCALE//den, ceildiv(q*RESIDUAL_SCALE,den)


def exact_interval(coefficient: int, denominator: int) -> tuple[int,int]:
    assert denominator>0
    p=coefficient*RESIDUAL_SCALE
    return p//denominator,ceildiv(p,denominator)


def interval_add(a: tuple[int,int],b:tuple[int,int])->tuple[int,int]:
    return a[0]+b[0],a[1]+b[1]


def component_bound(a:tuple[int,int])->int:
    return max(abs(a[0]),abs(a[1]))


def check_witness(m:int, U:list[list[list[int]]], y:int=2,
                  core:tuple[int,...]=(-1,0), cutoff:int=2048) -> dict:
    """Validate a core-resolvent witness entirely by exact integer bounds.

    If H is the infinite self-adjoint operator specified in THEORY.md,
    (H-i*y)R=I, so ||R F-J U|| <= ||(H-i*y)J U-F||/y.

    Residual rows -cutoff..cutoff-1 are bounded with outward fixed-point
    interval arithmetic. The *infinite* |n|>=cutoff remainder is bounded
    using |V_nm|<=4/(pi*(|n|-m/2)^2) and the integral bound for sum n^-4.
    """
    assert m>=2 and m%2==0 and isinstance(y,int) and y>0
    p=m//2
    if cutoff<=p+1: raise ValueError('cutoff must exceed m/2+1')
    if not core or len(set(core))!=len(core): raise ValueError('invalid core')
    if len(U)!=m or any(len(row)!=len(core) for row in U):
        raise ValueError('witness array shape')
    if any(not isinstance(v,int) for row in U for pair in row for v in pair):
        raise ValueError('witness entries must be signed integers')
    if any(len(pair)!=2 for row in U for pair in row):
        raise ValueError('witness entries must have real, imaginary')
    if not set(core).issubset(set(range(-p,p))):
        raise ValueError('core falls outside P')
    pi_interval = machin_pi_interval()
    inv_interval=inverse_pi_interval()
    near_squared=0
    for n in range(-cutoff,cutoff):
        inside=-p<=n<p
        nsign=1 if n%2==0 else -1
        for c,f in enumerate(core):
            real=(0,0);imag=(0,0)
            if inside:
                ur,ui=U[n+p][c]
                # diagonal pi(n+1/2)*u_n
                real=interval_add(real,product_interval((2*n+1)*ur,2,pi_interval))
                imag=interval_add(imag,product_interval((2*n+1)*ui,2,pi_interval))
                # -i*y*u_n, and -f_n
                real=interval_add(real,exact_interval(y*ui, VECTOR_SCALE))
                imag=interval_add(imag,exact_interval(-y*ur,VECTOR_SCALE))
                if n==f: real=interval_add(real,(-RESIDUAL_SCALE,-RESIDUAL_SCALE))
            # -4i*(-1)^n / (pi*(m-n)^2) * u_m for odd m-n
            for j,mode in enumerate(range(-p,p)):
                d=mode-n
                if d%2==0:continue
                ur,ui=U[j][c]
                if ur==0 and ui==0: continue
                real=interval_add(real,product_interval(4*nsign*ui,d*d,inv_interval))
                imag=interval_add(imag,product_interval(-4*nsign*ur,d*d,inv_interval))
            near_squared+=component_bound(real)**2+component_bound(imag)**2
    near_scaled=sqrt_ceil(near_squared)

    # For |n|>=cutoff, |n-mode|>=|n|-p, and
    # sum_{n=cutoff}^infty (n-p)^-4 <= 1/[3(cutoff-p-1)^3].
    l1_counts=[sum(abs(U[j][c][0])+abs(U[j][c][1]) for j in range(m))
               for c in range(len(core))]
    sum_l1_sq=sum(x*x for x in l1_counts)
    pi_lower=pi_interval[0]
    gap=cutoff-p-1
    tail_numerator=32*RESIDUAL_SCALE**2*PI_SCALE**2*sum_l1_sq
    tail_denominator=3*pi_lower**2*VECTOR_SCALE**2*gap**3
    tail_scaled=sqrt_ceil(ceildiv(tail_numerator,tail_denominator))
    # Both upper bounds are in units 1/RESIDUAL_SCALE.
    total_scaled=ceildiv(near_scaled+tail_scaled,y)
    return dict(
        M=m, y=y, core=list(core), cutoff=cutoff,
        pi_interval=[str(v) for v in pi_interval],
        inverse_pi_interval=[str(v) for v in inv_interval],
        residual_near_squared_upper_scaled=str(near_squared),
        residual_near_upper_scaled=str(near_scaled),
        residual_tail_upper_scaled=str(tail_scaled),
        resolvent_error_upper_scaled=str(total_scaled),
        resolution_scale=str(RESIDUAL_SCALE),
        error_upper=float(Fraction(total_scaled,RESIDUAL_SCALE)),
        near_residual_upper=float(Fraction(near_scaled,RESIDUAL_SCALE)),
        tail_residual_upper=float(Fraction(tail_scaled,RESIDUAL_SCALE)),
    )


def analytic_full_projection_bound(m:int,k:float,z:complex) -> dict:
    """Schur-complement bound for the entire M-mode compressed resolvent.

    ||V|| <= pi, shown by even/odd flavor Fourier symbols (THEORY.md).
    d_Q >= pi*(M+1)/2 - |Re z|, so if d_Q>pi:
    ||P(H-z)^(-1)P -(PHP-z)^(-1)|| <= pi^2/[|Im z|^2(d_Q-pi)].
    This *real* inequality is rigorous for every real k.
    """
    if m<=0 or m%2: raise ValueError('m must be positive and even')
    y=abs(z.imag)
    if not y: raise ValueError('z must have nonzero imaginary part')
    delta=pi*(m+1)/2-abs(z.real)
    value=pi*pi/(y*y*(delta-pi)) if delta>pi else float('inf')
    return {'M':m,'k':k,'z':[z.real,z.imag],
            'tail_diagonal_gap_lower':delta,'uniform_perturbation_bound':pi,
            'schur_upper':value,'trivial_upper':2/y,
            'certified_upper':min(value,2/y)}


def build_payload(m_values=(2,6,10,14,22,30),cutoff=2048,y=2)->dict:
    data={'schema':SCHEMA,'source':'https://arxiv.org/abs/2611.05329v1',
          'mode':'k=pi/2, z=iy, core n=-1,0',
          'rules':'integer interval residual + infinite-series tail',
          'cases':[]}
    for m in m_values:
        U=resolvent_witness(m,y)
        cert=check_witness(m,U,y,cutoff=cutoff)
        data['cases'].append({'M':m,'witness_vector_integer_pairs':U,
                             'certificate':cert,
                             'full_projection_bound':analytic_full_projection_bound(m,pi/2,1j*y)})
    return data


def verify_payload(payload:dict)->bool:
    if payload['schema']!=SCHEMA or payload['mode']!='k=pi/2, z=iy, core n=-1,0':
        raise ValueError('unknown schema')
    for case in payload['cases']:
        ref=check_witness(case['M'],case['witness_vector_integer_pairs'],
                          case['certificate']['y'],tuple(case['certificate']['core']),
                          case['certificate']['cutoff'])
        if case['certificate']!=ref: raise AssertionError('certificate mismatch')
    return True


def write_payload(path:Path,payload:dict)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(payload,sort_keys=True,indent=2)+'\n'
    path.write_text(body)


def file_digest(path:Path)->str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
