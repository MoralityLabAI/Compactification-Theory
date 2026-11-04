"""QO-06: exact finite perturbation certificates for learned hit-rate quotients.

A fully specified finite hit/hold system has rates p_i^a. Given estimates
phat_i^a with sup error eta, the fixed-partition TV radius and its
minimized value are 1-Lipschitz in eta, and a fitted partition/midpoint
macro kernel has true one-step error at most empirical optimum + eta.

A separate Hoeffding + union bound applies ONLY to pre-fixed sample counts
with iid Bernoulli outcomes within each microscopic state/action pair.
"""

from fractions import Fraction as Q
from math import ceil, comb, exp, log, sqrt

from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k


def _validate_pair(true, estimate):
    actual = tuple(tuple(Q(x) for x in p) for p in true)
    predicted = tuple(tuple(Q(x) for x in p) for p in estimate)
    if not actual or not actual[0] or len(actual) != len(predicted):
        raise ValueError("nonempty paired rate data required")
    d = len(actual[0])
    if any(len(p) != d for p in actual+predicted):
        raise ValueError("paired rate dimension mismatch")
    if any(x < 0 or x > 1 for p in actual+predicted for x in p):
        raise ValueError("hit/hold rates must lie in [0,1]")
    return actual, predicted


def max_estimation_error(true, estimate):
    actual, predicted = _validate_pair(true, estimate)
    return max(abs(x-y) for a,b in zip(actual,predicted)
               for x,y in zip(a,b))


def partition_radius(points, labels):
    """Exact optimal TV radius for the given partition; excludes absorbing goal."""
    pts = tuple(tuple(Q(x) for x in p) for p in points)
    if not pts or len(labels) != len(pts) or not pts[0]:
        raise ValueError("matching nonempty rate data and class labels required")
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(pts[i])
    return max(((max(p[j] for p in group)-min(p[j] for p in group))/2
                for group in groups.values() for j in range(len(pts[0]))),
               default=Q(0))


def empirical_midpoints(points, labels):
    """A stochastic macro kernel is specified by its per-class/action hit rates."""
    pts = tuple(tuple(Q(x) for x in p) for p in points)
    if not pts or len(labels)!=len(pts):
        raise ValueError("missing states or labels")
    groups = {}
    for i, z in enumerate(labels):
        groups.setdefault(z, []).append(pts[i])
    return {z: tuple((min(p[j] for p in group)+max(p[j] for p in group))/2
                     for j in range(len(pts[0])))
            for z,group in groups.items()}


def frozen_macro_error(true, labels, fitted_midpoints):
    pts = tuple(tuple(Q(x) for x in p) for p in true)
    if not pts or len(labels)!=len(pts):
        raise ValueError("invalid true rates or macro state labels")
    return max(abs(x-fitted_midpoints[labels[i]][j])
               for i,p in enumerate(pts) for j,x in enumerate(p))


def empirical_certificate(true, estimate, max_transient_classes):
    """Exhaustively auditable finite synthetic certificate; true rates here are known.

    In real evaluation, only empirical_best + a valid externally certified eta
    is observable; true rates, true_best and ex-post regret are audit fields.
    """
    actual, predicted = _validate_pair(true, estimate)
    if type(max_transient_classes) is not int or max_transient_classes < 1:
        raise ValueError("positive class budget required")
    empirical_best, _, labels_goal = optimum_for_k(predicted, max_transient_classes)
    actual_best, _, _ = optimum_for_k(actual, max_transient_classes)
    labels = labels_goal[:-1]
    eta = max_estimation_error(actual, predicted)
    midpoints = empirical_midpoints(predicted, labels)
    empirical_eval = partition_radius(predicted, labels)
    true_fixed_partition = partition_radius(actual, labels)
    true_frozen = frozen_macro_error(actual, labels, midpoints)
    excess = true_fixed_partition - actual_best
    assert empirical_eval == empirical_best
    assert abs(actual_best-empirical_best) <= eta
    assert abs(true_fixed_partition-empirical_eval) <= eta
    assert true_frozen <= empirical_best+eta
    assert excess <= 2*eta
    return {
        "uniform_rate_error": eta,
        "empirical_optimum": empirical_best,
        "true_optimum": actual_best,
        "true_fixed_partition_defect": true_fixed_partition,
        "true_frozen_macro_defect": true_frozen,
        "true_partition_regret": excess,
        "frozen_macro_upper_bound": empirical_best+eta,
        "optimization_regret_upper_bound": 2*eta,
        "transient_labels": tuple(labels),
        "midpoint_kernels": midpoints,
    }


def hoeffding_union_radius(microstates, actions, draws_per_pair, alpha):
    """Uniform fixed-N Bernoulli sup-error radius at confidence >= 1-alpha."""
    n, d, N = microstates, actions, draws_per_pair
    a = float(alpha)
    if any(type(x) is not int or x < 1 for x in (n,d,N)) or not 0<a<1:
        raise ValueError("n,d,N positive integers; alpha in (0,1)")
    return min(1., sqrt(log(2*n*d/a)/(2*N)))


def hoeffding_union_failure_bound(microstates, actions, draws_per_pair, eta):
    if any(type(x) is not int or x < 1 for x in
           (microstates, actions, draws_per_pair)):
        raise ValueError("positive n,d,N required")
    e = float(eta)
    if not 0 <= e <= 1:
        raise ValueError("eta must be in [0,1]")
    return min(1., 2*microstates*actions*exp(-2*draws_per_pair*e*e))


def per_pair_samples_for_regret(microstates, actions, regret, alpha):
    """Sufficient fixed N per microstate/action to certify excess <= regret."""
    n,d = microstates,actions
    delta,a = float(regret),float(alpha)
    if any(type(x) is not int or x < 1 for x in (n,d)) or not 0<delta<1 or not 0<a<1:
        raise ValueError("n,d positive; target regret and alpha in (0,1)")
    return ceil(2*log(2*n*d/a)/(delta*delta))


def exact_binomial_violation(p, N, eta):
    """Independent tiny-experiment oracle: P(|sample mean-p|>eta)."""
    p, eta = Q(p), Q(eta)
    if type(N) is not int or N < 1 or not 0 <= p <= 1 or eta < 0:
        raise ValueError("N>=1, p in [0,1], eta>=0")
    return sum((Q(comb(N,k))*p**k*(1-p)**(N-k)
                for k in range(N+1) if abs(Q(k,N)-p)>eta),Q(0))
