"""NOR-07S: stationary-center, exact rational nonlinear GRU tube.

A fast rigorous specialization of NOR-07T's general Volterra theorem for a
high-precision approximate fixed point c of the trained zero-input GRU.

Let J be the rational midpoint of DF(c), eta its validated operator error,
e >= ||F(c)-c||, and M a global Hessian bound on the invariant hidden cube.
The same single J is used at every step, retaining powers ||J^k|| rather
than their far looser one-step norm products.

For all initial x in B(c,R), validated radii obey
delta_n = ||J^n||*R
  + sum_{k<n} ||J^(n-k-1)||*(e+eta*delta_k+M*delta_k^2/2).
For the derivative product, a separate bound is
E_n = sum_{k<n} ||J^(n-k-1)||*(eta+M*delta_k)
                           *(||J^k||+E_k).
Thus ||D(F^L)(x)||<=||J^L||+E_L for *every* x in the ball.

All certificate inequalities use Fraction, rigorous exponential intervals,
and outward rational square roots. This method remains valid for nonnormal
J; no spectral-radius shortcut or commutation of perturbed Jacobians.
"""
from __future__ import annotations

from fractions import Fraction as Q

from trained_gru_cert import (
    Interval,N,ZERO,ONE,SCALE,gates,jacobian_interval,hessian_bound,
    round_rational,radius_upper,sqrt_upper,
)


def midpoint_matrix_and_error(weights,center):
    ji=jacobian_interval(weights,center)
    B=[[round_rational((x.lo+x.hi)/2) for x in row] for row in ji]
    eta=sqrt_upper(sum((
       max(abs(x.lo-b),abs(x.hi-b))**2
       for row, mid in zip(ji,B,strict=True)
       for x,b in zip(row,mid,strict=True)),ZERO))
    return B,eta


def matmul(A,B):
    return [[sum((A[i][k]*B[k][j] for k in range(N)),ZERO)
             for j in range(N)] for i in range(N)]


def frob_upper(matrix):
    return sqrt_upper(sum((v*v for row in matrix for v in row),ZERO))


def stationary_tube(weights,center,radius,length):
    if length<1 or radius<=0 or len(center)!=N:
        raise ValueError("length, radius and state width invalid")
    if any(abs(x)+radius>=ONE for x in center):
        raise ValueError("initial ball must be inside invariant hidden cube")

    J,eta=midpoint_matrix_and_error(weights,center)
    M=hessian_bound(weights)
    one_step=gates(weights,[Interval.point(x) for x in center])[0]
    e=radius_upper(one_step,center)

    # Exact rational powers (J^0,...,J^L).
    power=[[Q(int(i==j)) for j in range(N)] for i in range(N)]
    norms=[ONE]
    for k in range(1,length+1):
        power=matmul(J,power)
        norms.append(frob_upper(power))

    delta=[radius]
    derivative_error=[ZERO]
    point_error=[ZERO]
    for n in range(1,length+1):
        next_delta=norms[n]*radius+sum((
            norms[n-k-1]*(e+eta*delta[k]+M*delta[k]*delta[k]/2)
            for k in range(n)),ZERO)
        delta.append(next_delta)
        deriv=sum((
            norms[n-k-1]*(eta+M*delta[k])*(norms[k]+derivative_error[k])
            for k in range(n)),ZERO)
        derivative_error.append(deriv)
        next_point=sum((
            norms[n-k-1]*(e+eta*point_error[k]+
                M*point_error[k]*point_error[k]/2)
            for k in range(n)),ZERO)
        point_error.append(next_point)

    K=norms[length]+derivative_error[length]
    s=point_error[length]
    return {
        "certification_method":"constant rational center, noncommutative Volterra powers and global Hessian",
        "block_length":length,
        "initial_radius_fraction":str(radius),
        "point_jacobian_error_fraction":str(eta),
        "one_step_center_residual_upper_fraction":str(e),
        "global_hessian_upper_fraction":str(M),
        "nominal_product_frobenius_upper_fraction":str(norms[length]),
        "block_derivative_variation_upper_fraction":str(derivative_error[length]),
        "block_operator_norm_upper_fraction":str(K),
        "block_operator_norm_upper_float":float(K),
        "center_residual_upper_fraction":str(s),
        "center_residual_upper_float":float(s),
        "max_tube_radius_upper_fraction":str(max(delta)),
        "max_point_shadowing_error_fraction":str(max(point_error)),
        "invariance_margin_fraction":str(radius-K*radius-s),
        "contracting_invariant_ball":K<ONE and s+K*radius<=radius,
    }
