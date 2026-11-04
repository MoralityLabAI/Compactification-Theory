"""LSV-14: full-ambient-ball Jacobian-rank certificate for frozen recurrent states.

All 3888 coordinates at the stage-one state vary independently in a closed
L-infinity ball. 4 selected tangent directions are differentiated across
that WHOLE box using Arb interval jets. Nonzero determinant proves rank four
at EVERY real initial state in that region, conditionally on trusted Arb.

This certifies a finite-radius local differential obstruction, not trajectory
recovery, hidden-to-hidden contraction, task correctness, or hardware floats.
"""
from __future__ import annotations

from fractions import Fraction
from flint import arb, arb_mat, ctx
from validated_minor import (State, CertifiedBlock, head_pair_margin,
                              all_head_logits, serialized_ball_contains_original)
from verify_serialized_minors import independent_det, parse_ball


def _fraction(s: str) -> Fraction:
    if not isinstance(s, str) or not s:
        raise ValueError('radius must be a positive rational string')
    x=Fraction(s)
    if x < 0 or x > 1:
        raise ValueError('radius must be in [0,1]')
    return x


def interval_state(stage1, columns, radius: str) -> State:
    """Build the 3888-dimensional axis-aligned input box with four unit jets.

    Exact float32/float64 coordinate midpoints are interpreted as exact reals.
    arb(mid, rad) rounds the nonnegative radius UP as necessary, so the
    initial box contains all points at L-infinity distance <= exact radius.
    """
    if tuple(stage1.shape) != (81, 48):
        raise ValueError('expected one 81x48 stage-one state')
    if len(columns)!=4 or len(set(columns))!=4 or any(type(k)!=int or not 0<=k<3888 for k in columns):
        raise ValueError('require 4 distinct valid coordinate indices')
    radius_exact=_fraction(radius)
    entries=[]
    # Use a canonical *rational* string as radius input, not a Python float.
    rad = f'{radius_exact.numerator}/{radius_exact.denominator}' if radius_exact else '0'
    for row in stage1.detach().cpu().tolist():
        entries.append([arb(float(v),rad) for v in row])
    data=arb_mat(entries)
    dirs=[]
    for col in columns:
        tangents=[arb(0)]*3888
        tangents[col]=arb(1)
        dirs.append(arb_mat(81,48,tangents))
    return State(data,tuple(dirs))


def _nonzero_rational(interval):
    return interval[0]>0 or interval[1]<0


def robust_minor(model, stage1, columns, cell, winner, rival,
                 radius: str, precision: int=160) -> dict:
    """Produce a checkable *uniform* 4x4 determinant enclosure over B_inf.

    For mathematical differentiability, the fixed digit pair is a specified
    smooth real-valued margin, independent of whether the digit ordering
    persists everywhere in the ball. Full 10-way class-order stability is
    a separately reported property, evaluated via final-state logit balls.
    """
    if type(precision)!=int or precision<96:
        raise ValueError('use at least 96 Arb precision bits')
    if not (type(cell)==int and 0<=cell<81 and type(winner)==int and
            type(rival)==int and 0<=winner<10 and 0<=rival<10 and winner!=rival):
        raise ValueError('bad output cell or class pair')
    ctx.prec=precision
    st=interval_state(stage1,columns,radius)
    net=CertifiedBlock(model.block)
    rows=[]
    for k in range(4):
        _,gradient=head_pair_margin(st,model.head,cell,winner,rival)
        rows.append(gradient)
        if k<3:
            st=net(st)
    for row in rows:
        for v in row:
            if not v.is_finite():
                raise ArithmeticError('nonfinite Jacobian ball')
    mat=arb_mat(rows)
    det=mat.det()
    if not det.is_finite():
        raise ArithmeticError('nonfinite determinant ball')
    # Printed decimal enclosures must be independently checked to contain
    # their source Arb objects before passing to a pure-Python verifier.
    printed=[[v.str(30) for v in row] for row in rows]
    if not all(serialized_ball_contains_original(rows[i][j],printed[i][j])
               for i in range(4) for j in range(4)):
        raise AssertionError('truncated Jacobian decimal interval')
    printed_det=det.str(30)
    if not serialized_ball_contains_original(det,printed_det):
        raise AssertionError('truncated Arb determinant')
    rational_det=independent_det([[parse_ball(v) for v in row] for row in printed])
    logits=all_head_logits(st,model.head,cell)
    ordered=all(logits[winner].lower()>logits[j].upper()
                for j in range(10) if j!=winner)
    runner=all(logits[rival].lower()>logits[j].upper()
               for j in range(10) if j not in (winner,rival))
    return {
        'radius':radius,
        'initial_box_dimensions':3888,
        'selected_columns':list(columns),
        'goal_cell':cell,
        'winner_digit':winner,'rival_digit':rival,
        'entry_intervals':printed,
        'determinant_arb_enclosure':printed_det,
        'arb_det_excludes_zero':not det.contains(0),
        'independent_rational_det_excludes_zero':_nonzero_rational(rational_det),
        'independent_rational_abs_det_lower_bound':str(rational_det[0] if rational_det[0]>0 else -rational_det[1] if rational_det[1]<0 else 0),
        'top1_stable_across_box':ordered,
        'top2_stable_across_box':ordered and runner,
    }


def check_single_receipt(row: dict) -> tuple[bool, Fraction]:
    """Independent rational check of the SERIALIZED finite-radius witness."""
    if _fraction(row['radius'])<=0:
        raise ValueError('finite-radius certificate requires strictly positive radius')
    if row['initial_box_dimensions']!=3888 or len(row['selected_columns'])!=4:
        raise ValueError('bad box or projection geometry')
    if len(set(row['selected_columns']))!=4 or any(type(i)!=int or i<0 or i>=3888 for i in row['selected_columns']):
        raise ValueError('bad selected columns')
    entries=row['entry_intervals']
    if len(entries)!=4 or any(len(r)!=4 for r in entries):
        raise ValueError('need 16 true-derivative interval entries')
    bounds=[[parse_ball(s) for s in rr] for rr in entries]
    lo,hi=independent_det(bounds)
    if not _nonzero_rational((lo,hi)):
        return False, Fraction(0)
    return True, lo if lo>0 else -hi


def check_full_receipt(receipt: dict):
    if receipt.get('schema')!='LSV-14/positive-radius-goal-rank-v1':
        raise ValueError('unexpected receipt schema')
    if receipt.get('radius') not in ('1e-22','1e-20','1e-18','1e-16','1e-14','1e-12','1e-10','1e-8','1e-6'):
        raise ValueError('unregistered radius')
    checked=[]
    for index,row in enumerate(receipt['cases']):
        if row['case_index']!=index or row['radius']!=receipt['radius']:
            raise ValueError('case order or radius mismatch')
        certified,lb=check_single_receipt(row)
        checked.append((certified,lb))
    return {
        'cases_checked':len(checked),
        'positive_radius_rank_four_cases':sum(int(v[0]) for v in checked),
        'strict_lower_bound_of_abs_det': str(min([v[1] for v in checked if v[0]],default=Fraction(0))),
    }
