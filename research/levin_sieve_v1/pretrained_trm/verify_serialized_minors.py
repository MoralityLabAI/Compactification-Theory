"""LSV-13 independent proof checker for serialized Arb-ball witness tables.

Uses only Python standard-library exact Fraction arithmetic and permutation
expansion. Does not trust reported determinant values or success booleans.
A trusted Arb evaluation step is required to establish that the printed balls
contain the true derivatives of the exact-real pretrained network.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
_BALL = re.compile(r'^\[('+_NUMBER+r') \+/- ('+_NUMBER+r')\]$')
_RADIUS_ONLY = re.compile(r'^\[\+/- ('+_NUMBER+r')\]$')


def _q(s: str) -> Fraction:
    return Fraction(Decimal(s))


def parse_ball(s: str) -> tuple[Fraction, Fraction]:
    if not isinstance(s,str): raise ValueError('expected decimal interval string')
    s=s.strip()
    if m:=_BALL.fullmatch(s):
        c=_q(m[1]); rad=_q(m[2])
        if rad<0:raise ValueError('negative radius')
        return c-rad,c+rad
    if m:=_RADIUS_ONLY.fullmatch(s):
        rad=_q(m[1]);
        if rad<0:raise ValueError('negative radius')
        return -rad,rad
    if re.fullmatch(_NUMBER,s):
        z=_q(s);return z,z
    raise ValueError('unsupported or malformed ball encoding: '+s)


def add(a,b): return a[0]+b[0],a[1]+b[1]


def mul(a,b):
    p=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
    return min(p),max(p)


def neg(a):return -a[1],-a[0]


def independent_det(matrix):
    """Leibniz formula using 24 disjoint 4x4 interval terms, no Arb."""
    n=len(matrix)
    if n!=4 or any(len(row)!=4 for row in matrix):
        raise ValueError('expected 4x4 interval witness')
    result=(Fraction(0),Fraction(0))
    for p in itertools.permutations(range(4)):
        term=(Fraction(1),Fraction(1))
        inversions=sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))
        for i in range(4):term=mul(term,matrix[i][p[i]])
        if inversions%2:term=neg(term)
        result=add(result,term)
    return result


def upper_abs_dist(observed:Fraction, interval):
    """Upper bound on |observed-true| given true in interval."""
    return max(abs(observed-interval[0]),abs(observed-interval[1]))


def verify_record(record,require_autograd=True):
    rows=record['verified_jacobian_entry_enclosures']
    if len(rows)!=4 or any(len(row)!=4 for row in rows):
        raise ValueError('missing or malformed 4x4 row witness')
    intervals=[[parse_ball(x) for x in row] for row in rows]
    det=independent_det(intervals)
    if det[0]<=0<=det[1]:
        raise AssertionError(f"case {record['case_index']}: exact rational determinant interval contains zero")
    old_threshold=_q(str(record['previous_required_entry_error_lt']))
    if old_threshold<=0:raise ValueError('invalid saved error budget')
    max_error=None
    observed=record.get('observed_autograd_binary64_hex')
    if observed is None:
        if require_autograd:raise AssertionError('missing independent autograd witness')
    else:
        if len(observed)!=4 or any(len(row)!=4 for row in observed):
            raise ValueError('bad observed derivative matrix')
        bounds=[]
        for row in range(4):
            for col in range(4):
                q=Fraction.from_float(float.fromhex(observed[row][col]))
                bounds.append(upper_abs_dist(q,intervals[row][col]))
        max_error=max(bounds)
        if max_error>=old_threshold:
            raise AssertionError(f"case {record['case_index']}: LSV12 entrywise budget not proved")
    return {'case_index':record['case_index'],
            'rank_four_proved':True,
            'previous_lsv12_autograd_error_budget_proved':observed is not None,
            'lower_bound_abs_det_rational':max(det[0],-det[1]) if det[0]>0 else -det[1],
            'max_autograd_entry_error_rational':max_error}


def verify_receipt(receipt,require_autograd=True):
    if receipt['schema']!='LSV-13/arb-validated-goal-minors-v1':
        raise ValueError('unexpected receipt schema')
    if receipt['cases_checked']!=len(receipt['cases']):
        raise ValueError('case count mismatch')
    if [c['case_index'] for c in receipt['cases']]!=list(range(len(receipt['cases']))):
        raise ValueError('roster is reordered or nonconsecutive')
    checked=[verify_record(case,require_autograd) for case in receipt['cases']]
    return {'cases_checked':len(checked),
            'independent_exact_rational_rank_four_count':len(checked),
            'independent_autograd_budget_count':sum(v['previous_lsv12_autograd_error_budget_proved'] for v in checked),
            'smallest_abs_det_lower_bound':str(min(v['lower_bound_abs_det_rational'] for v in checked))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('receipt',type=Path)
    p.add_argument('--allow-legacy-without-observed-hex',action='store_true')
    args=p.parse_args()
    result=verify_receipt(json.loads(args.receipt.read_text()),
                          require_autograd=not args.allow_legacy_without_observed_hex)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
