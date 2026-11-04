"""LSV-17 exact arithmetic for goal-directed Taylor certificates.

This checker verifies the final inequalities from supplied sound bounds.
Producing those bounds requires the separately audited Arb computation.
"""
from fractions import Fraction as Q


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, Q)):
        raise TypeError('exact int, decimal/rational string, or Fraction required')
    return Q(x)


def margin_lower(margin, radius, gradient_l1, defects):
    m, r, g = map(rational, (margin, radius, gradient_l1))
    ds = [rational(d) for d in defects]
    if r < 0 or g < 0 or any(d < 0 for d in ds):
        raise ValueError('negative radius, gradient bound, or defect')
    return m - r*g - sum(ds, Q(0))


def chain_budget(radius_l2, layers):
    """Local Taylor defects with optional adversarial post-layer error.

    Each layer declares uniform derivative bounds a,b on its input ball,
    a verified clean downstream adjoint norm q, and optional error eta.
    The caller must establish these bounds on the resulting input radii.
    """
    R = rational(radius_l2)
    if R < 0:
        raise ValueError('radius must be nonnegative')
    result=[]
    for row in layers:
        a,b,q = (rational(row[k]) for k in ('a','b','q'))
        eta = rational(row.get('eta', 0))
        if min(a,b,q,eta) < 0:
            raise ValueError('all bounds must be nonnegative')
        local = b*R*R/2 + eta
        result.append({'input_radius':R, 'local_defect':local,
                       'output_defect':q*local})
        R = a*R+eta
    return R,result


def ln_curvature_slack(q, y):
    """Exact identity proving the sharp radial LayerNorm Hessian constant."""
    q,y = rational(q),rational(y)
    if not 0 <= q <= 1 or not 0 <= y <= 1:
        raise ValueError('q and y must be in [0,1]')
    squared=q*(1+(8-6*q)*y+(-12*q+9*q*q)*y*y)
    slack=(4-3*q)*(3*q*y-1)**2/3
    if Q(4,3)-squared != slack:
        raise AssertionError('polynomial identity failure')
    return slack


def quadratic_threshold_passes(margin, gradient, curvature, radius):
    m,g,h,r = map(rational,(margin,gradient,curvature,radius))
    if min(g,h,r) < 0:
        raise ValueError('nonnegative bounds required')
    return m-g*r-h*r*r/2 > 0
