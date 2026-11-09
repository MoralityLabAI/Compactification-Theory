"""Finite-flavor selection using validated infinite-resolvent error bounds.

The witness solver proposes candidates. A separate integer-arithmetic verifier
checks each candidate against the actual infinite Dirac operator at k=pi/2.
The search reports only the first passing M in a fixed explicit grid; it
assumes neither monotonicity nor global optimality in M.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from infinite_resolvent import (
    SCHEMA, RESIDUAL_SCALE, resolvent_witness, check_witness,
    verify_payload, write_payload,
)

DEFAULT_GRID = (2, 6, 10, 14, 22, 30, 38, 54, 70, 94, 126)


def select(target: Fraction, grid=DEFAULT_GRID, cutoff: int = 1024,
           y: int = 2) -> dict:
    """Enumerate supplied even truncations; validate each exact-integer receipt."""
    if target <= 0 or y <= 0:
        raise ValueError('target and y must be positive')
    if not grid or any(m < 2 or m % 2 for m in grid):
        raise ValueError('grid must contain positive even truncations')
    if len(set(grid)) != len(grid) or tuple(sorted(grid)) != tuple(grid):
        raise ValueError('grid must be strictly increasing')
    cases = []
    passed = None
    for m in grid:
        witness = resolvent_witness(m, y)
        cert = check_witness(m, witness, y=y, cutoff=cutoff)
        exact_upper = Fraction(int(cert['resolvent_error_upper_scaled']), RESIDUAL_SCALE)
        case = {'M': m, 'witness_vector_integer_pairs': witness,
                'certificate': cert,
                'target_passed': exact_upper <= target}
        cases.append(case)
        if exact_upper <= target:
            passed = m
            break
    payload = {
        'schema': SCHEMA,
        'source': 'https://arxiv.org/abs/2611.05329v1',
        'mode': 'k=pi/2, z=iy, core n=-1,0',
        'rules': 'integer interval residual + infinite-series tail',
        'selection': {'target_numerator': str(target.numerator),
                      'target_denominator': str(target.denominator),
                      'candidate_grid': list(grid), 'first_passing_M': passed},
        'cases': cases,
    }
    verify_selection(payload)
    return payload


def verify_selection(payload: dict) -> bool:
    """Verify witness bytes and minimality within evaluated candidates.

    Does not call a floating-point solver. Counts untested grid levels as
    unknown, and insists that the first pass terminates the recorded scan.
    """
    verify_payload(payload)
    record = payload['selection']
    target = Fraction(int(record['target_numerator']),
                      int(record['target_denominator']))
    grid = record['candidate_grid']
    cases = payload['cases']
    if target <= 0 or not grid or any(m < 2 or m % 2 for m in grid):
        raise ValueError('invalid selection request')
    if tuple(sorted(set(grid))) != tuple(grid):
        raise ValueError('grid must be strictly increasing')
    if len(cases) > len(grid) or not cases:
        raise ValueError('missing or excess evaluated candidates')
    first = None
    for i, case in enumerate(cases):
        if case['M'] != grid[i]:
            raise AssertionError('candidate grid / receipt mismatch')
        cert = case['certificate']
        exact_upper = Fraction(int(cert['resolvent_error_upper_scaled']),
                               int(cert['resolution_scale']))
        passed = exact_upper <= target
        if case['target_passed'] != passed:
            raise AssertionError('incorrect reported threshold decision')
        if passed:
            if i != len(cases)-1:
                raise AssertionError('scan continued past first pass')
            first = case['M']
    if first is None and len(cases) != len(grid):
        raise AssertionError('scan omitted candidates before a pass')
    if record['first_passing_M'] != first:
        raise AssertionError('incorrect first-passing index')
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='0.001', help='exact decimal or fraction, e.g. 1/1000')
    p.add_argument('--grid', default=','.join(map(str, DEFAULT_GRID)))
    p.add_argument('--cutoff', type=int, default=1024)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify', type=Path)
    a = p.parse_args()
    if a.verify:
        payload = json.loads(a.verify.read_text())
        verify_selection(payload)
    else:
        payload = select(Fraction(a.target), tuple(map(int, a.grid.split(','))), a.cutoff)
        if a.output:
            write_payload(a.output, payload)
    for c in payload['cases']:
        cert = c['certificate']
        print(f"M={c['M']:>3}: bound {cert['error_upper']:.9f}, pass={c['target_passed']}")
    print('first passing M:', payload['selection']['first_passing_M'])


if __name__ == '__main__':
    main()
