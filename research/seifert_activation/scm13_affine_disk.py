"""SCM-13: exact affine-disk braid certificate with bounded nonlinear remainder.

Given observed chart strands z_i(theta) = a_i + radius * B_i(cos theta,sin theta)
+ e_i(theta), with ||e_i(theta)|| <= eta_i for EVERY theta, compute
conditional pair winding and full-braid null certificates. All geometry uses
exact Fractions and outward rational sqrt bounds. The eta_i claims require a
separate source-level proof; sampling/autodiff alone does not supply them.

For full-rank differences B_i-B_j, the affine collision preimage is
u_ij = -(B_i-B_j)^(-1)(a_i-a_j). Its norm is the critical intervention radius.
If every pair remains collision-free during radial contraction, the full
labeled braid is null, including its nonabelian Artin component.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Sequence, Mapping

from continuous import rat, _sqrt_bounds


def _p2(value) -> tuple[Fraction, Fraction]:
    if len(value) != 2:
        raise ValueError('Points must have two coordinates')
    return (rat(value[0]), rat(value[1]))


def _matrix(value) -> tuple[tuple[Fraction,Fraction],tuple[Fraction,Fraction]]:
    if len(value) != 2:
        raise ValueError('Each response matrix must be 2 by 2')
    rows=tuple(_p2(row) for row in value)
    return (rows[0],rows[1])


def _difference(a,b):
    return ((a[0][0]-b[0][0],a[0][1]-b[0][1]),
            (a[1][0]-b[1][0],a[1][1]-b[1][1]))


def _det(a):
    return a[0][0]*a[1][1]-a[0][1]*a[1][0]


def _inverse(a):
    d=_det(a)
    if d==0:
        return None
    return ((a[1][1]/d,-a[0][1]/d),(-a[1][0]/d,a[0][0]/d))


def _mul(a,v):
    return (a[0][0]*v[0]+a[0][1]*v[1],a[1][0]*v[0]+a[1][1]*v[1])


def _n2(v):
    return v[0]*v[0]+v[1]*v[1]


def _frob2(matrix):
    return sum(x*x for row in matrix for x in row)


def _encode(x):
    return str(x)


def taylor_errors_from_hessian(hessian_bounds: Sequence, radius) -> list[Fraction]:
    """Remainder eta_i <= H_i radius^2/2, if H_i is certified on full disk.

    H_i bounds the operator norm of the 2nd derivative of the token's chart
    response with respect to the two intervention coefficients. This is a
    source-derived premise, never estimated by finite differences here.
    """
    r=rat(radius)
    if r<0:
        raise ValueError('Radius must be nonnegative')
    out=[rat(h)*r*r/2 for h in hessian_bounds]
    if any(h<0 for h in out):
        raise ValueError('Hessian bounds must be nonnegative')
    return out


def affine_disk_certificate(
    base: Sequence[Sequence],
    jacobians: Sequence[Sequence[Sequence]],
    radius,
    errors: Sequence,
    *,
    margin='0',
    sqrt_digits=14,
) -> dict:
    """Certify a labeled cyclic braid's topology relative to its affine surrogate.

    A pair is certified when the exact lower bound for the smallest distance
    from its affine ellipse to the collision origin exceeds eta_i+eta_j+margin.
    With a rank-two B_ij, the affine winding is sign det B_ij when its collision
    preimage lies strictly inside radius, otherwise 0. This sign convention is
    counterclockwise *pairwise winding*, not SCM-01's Artin linking sign.

    If every pair is certified and every affine winding is zero, the entire
    labeled braid is trivial, because radial contraction avoids all diagonal
    collision loci. Zero pair linking in general does NOT imply a trivial braid;
    this stronger conclusion relies on the full affine-disk premise.
    """
    n=len(base)
    if n<2 or len(jacobians)!=n or len(errors)!=n:
        raise ValueError('Require at least two matching strands, Jacobians, and bounds')
    r=rat(radius);extra=rat(margin)
    if r<=0 or extra<0:
        raise ValueError('Radius must be positive and margin nonnegative')
    a=[_p2(p) for p in base]
    b=[_matrix(m) for m in jacobians]
    eta=[rat(v) for v in errors]
    if any(v<0 for v in eta):
        raise ValueError('Remainder bounds must be nonnegative')
    # Validate square-root precision even if no pair is invertible.
    if not isinstance(sqrt_digits,int) or not 0<=sqrt_digits<=18:
        raise ValueError('Sqrt precision must be an integer in 0..18')
    reports={}; all_certified=True; all_zero=True
    for i,j in combinations(range(n),2):
        d=(a[i][0]-a[j][0],a[i][1]-a[j][1])
        B=_difference(b[i],b[j])
        det=_det(B)
        inv=_inverse(B)
        k=f'{i},{j}'
        required=eta[i]+eta[j]+extra
        if inv is None:
            reports[k]={
                'status':'rank_deficient_unresolved',
                'determinant':'0',
                'required_clearance':str(required),
                'pair_winding':None,
                'certified':False,
            }
            all_certified=False;all_zero=False
            continue
        collision=_mul(inv,(-d[0],-d[1]))
        c2=_n2(collision)
        lo,hi=_sqrt_bounds(c2,sqrt_digits)
        # Closed interval containing the exact critical radius.
        if r<lo:
            radial_gap=lo-r
            side='outside'
            winding=0
        elif r>hi:
            radial_gap=r-hi
            side='inside'
            winding=1 if det>0 else -1
        else:
            radial_gap=Fraction(0)
            side='threshold_unresolved'
            winding=None
        # For invertible B, ||B v|| >= ||v|| / ||B^-1||_F.
        # Use an exact squared comparison without an unsafe sqrt approximation.
        inv_frob2=_frob2(inv)
        meets=(radial_gap>0 and radial_gap*radial_gap > required*required*inv_frob2)
        certified=bool(meets)
        all_certified=all_certified and certified
        all_zero=all_zero and winding==0
        reports[k]={
            'status':'certified' if certified else 'insufficient_margin',
            'determinant':str(det),
            'critical_radius_interval':[str(lo),str(hi)],
            'critical_radius_squared':str(c2),
            'side':side,
            'pair_winding':winding,
            'relative_error_plus_margin':str(required),
            'radial_clearance_gap_lower':str(radial_gap),
            'inverse_frobenius_norm_squared':str(inv_frob2),
            'clearance_inequality_squared':{
                'lhs':str(radial_gap*radial_gap),
                'rhs':str(required*required*inv_frob2),
            },
            'certified':certified,
        }
    if not all_certified:
        result='inconclusive'
    elif all_zero:
        result='conditional_full_null_braid'
    else:
        result='conditional_affine_braid_preserved'
    return {
        'schema':'scm13/affine-disk-certificate/v1',
        'result':result,
        'per_pair':reports,
        'radius':str(r),
        'full_affine_disk_homotopy_certified':bool(all_certified and all_zero),
        'all_pair_windings_certified':bool(all_certified),
        'winding_convention':'counterclockwise labeled pair z_i-z_j',
        'requires': 'proven full-circle per-strand nonlinear remainder bounds and exact-real chart Jacobians',
        'source_hessian_verified':False,
        'trained_neural_observation_certified':False,
    }
