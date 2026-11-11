"""Exhaustive p=31 shear Bayes-risk curves at matched B-bit code budgets.

Run: apy research/dynamical_closure/shear_controls.py --output shear_results.json
Numerical phases define integer code partitions; risks on those partitions
are computed with exact rational arithmetic over every microscopic state.
"""
import argparse
import json
from math import cos, pi, sin
from pathlib import Path

import numpy as np

from cyclic_addition import states
from shear_exact import balanced_curve, oracle_curve, present, risks


def feature_banks(p=31):
    xs = states(p)
    def pair(z):
        angle = 2*pi*z/p
        return cos(angle), sin(angle)
    # Raw input phases do not explicitly contain the sum feature.
    raw = np.array([(*pair(a), *pair(b)) for a,b in xs], dtype=np.float64)
    # Task-engineered character basis, including the hidden a required by shear.
    engineered = np.array([(*pair(present(a,b,p)), *pair(a)) for a,b in xs], dtype=np.float64)
    return raw, engineered


def allocate(budget, dimensions=4):
    if not 0 <= budget <= 10:
        raise ValueError('budget must be in 0..10')
    return tuple(sum(step % dimensions == d for step in range(budget)) for d in range(dimensions))


def quantize(features, budget, basis=None):
    if basis is None:
        basis = np.eye(4)
    rotated = features @ basis
    # A fixed bound from the rotation, independent of state labels or budget.
    bound = np.abs(basis).sum(axis=0)
    bits = allocate(budget)
    code = np.zeros_like(rotated, dtype=np.int16)
    for d, count in enumerate(bits):
        if count:
            n = 2**count
            scaled = n * (rotated[:, d] + bound[d]) / (2*bound[d])
            code[:, d] = np.floor(scaled).clip(0, n-1).astype(np.int16)
    return tuple(tuple(map(int, row)) for row in code)


def curve(features, basis=None, p=31):
    output = []
    for budget in range(11):
        code = quantize(features, budget, basis)
        r0, r1 = risks(code, p)
        output.append({'bits': budget, 'allocation': allocate(budget),
                       'codes': len(set(code)), 'present_risk': str(r0),
                       'after_shear_risk': str(r1)})
    return output


def random_basis(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(4,4)))
    return q @ np.diag(np.where(np.diag(r) < 0, -1., 1.))


def run(p=31, random_seeds=(0,1,2)):
    if p != 31:
        raise ValueError('this controlled 0..10 bit protocol is fixed at p=31')
    raw, engineered = feature_banks(p)
    cov = np.cov(raw.T, bias=True)
    if not np.allclose(cov, np.eye(4)*.5, atol=1e-12, rtol=0):
        raise AssertionError('unexpected raw-feature covariance')
    # Covariance has four equal eigenvalues. Coordinate axes are an explicit,
    # deterministic PCA tie-break, not discovered privileged components.
    controls = {'fourier_sum_a': curve(engineered, p=p),
                'pca_input_tie_axes': curve(raw, p=p)}
    for seed in random_seeds:
        controls[f'random_fourier_{seed}'] = curve(engineered, random_basis(seed), p)
        controls[f'random_input_{seed}'] = curve(raw, random_basis(seed), p)
    return {'p':p, 'states':p*p, 'task':'h0=a+b, h1=2a+b mod p',
            'measure':'uniform exhaustive states', 'loss':'one-hot squared Bayes error',
            'code_budget':'B bits TOTAL, round-robin dyadic allocation to 4 fixed coordinates',
            'decoder':'optimal conditional mean per attained codeword for every method',
            'phase_dtype':'float64; exact risk conditional on realized integer partitions',
            'pca_tie':'uniform input-phase covariance = 0.5 I; fixed canonical axes',
            'random_seeds':list(random_seeds),
            'present_first_reference':oracle_curve(p),
            'balanced_reference':balanced_curve(p), 'controls':controls}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=str)
    args=parser.parse_args()
    result=run()
    text=json.dumps(result,indent=2)
    if args.output:
        Path(args.output).write_text(text+'\n')
    print(text)
