"""QO-09: exact finite Bernoulli testing limits for quotient identifiability.

Binary worlds A/B differ only in ONE addressable state/action hit rate.
The nuisance probe stream has the same independent law under both worlds.
Exact Bayes risk of B informative trials is the likelihood-overlap formula.
An arbitrary adaptive mixture with nuisance probes is a garbling of B
informative draws plus world-independent randomness.

All finite calculations are exact fractions, except for an explicitly
labelled numerical Pinsker/KL comparison.
"""
from fractions import Fraction as Q
from itertools import product
from math import ceil, comb, log, sqrt


def _p(value):
    x = Q(value)
    if not 0 < x < 1:
        raise ValueError("hypothesis probabilities must lie strictly in (0,1)")
    return x


def binomial_sequence_tv(draws, p, q):
    """Exact TV of length-B iid Bernoulli bitstrings, grouped by success count."""
    if type(draws) is not int or draws < 0:
        raise ValueError("draws must be a nonnegative integer")
    p, q = _p(p), _p(q)
    return sum((
        Q(comb(draws, s)) *
        abs(p**s * (1-p)**(draws-s) -
            q**s * (1-q)**(draws-s))
        for s in range(draws+1)
    ),Q(0)) / 2


def binomial_bayes_risk(draws, p, q):
    """Optimal equal-prior *average* error: (1-TV)/2."""
    return (1-binomial_sequence_tv(draws,p,q))/2


def first_budget_for_bayes_error(p, q, target_error, max_budget):
    p, q = _p(p), _p(q)
    target = Q(target_error)
    if (not 0 <= target <= Q(1,2) or
            type(max_budget) is not int or max_budget < 0):
        raise ValueError("target error in [0,1/2], nonnegative budget")
    for n in range(max_budget+1):
        if binomial_bayes_risk(n,p,q) <= target:
            return n
    return None


def adaptive_policy_tv(draws, p, q, nuisance, mask):
    """Exact transcript TV for one deterministic binary-history query policy.

    At every binary observation prefix, action 1 probes informative cell,
    action 0 probes world-independent nuisance cell. The selection tree is
    encoded as a bit mask over breadth-first binary-history nodes.
    """
    if type(draws) is not int or not 0 <= draws <= 10:
        raise ValueError("finite policy oracle limited to B<=10")
    p,q,nuisance = _p(p),_p(q),_p(nuisance)
    nodes = (1 << draws) - 1
    if type(mask) is not int or mask < 0 or mask >= (1 << nodes):
        raise ValueError("invalid policy mask")
    total=Q(0)
    for observations in product((0,1), repeat=draws):
        prob_p=Q(1)
        prob_q=Q(1)
        prefix=0
        for depth,hit in enumerate(observations):
            node=(1 << depth)-1+prefix
            informative=bool(mask & (1 << node))
            a,b=(p,q) if informative else (nuisance,nuisance)
            prob_p *= a if hit else 1-a
            prob_q *= b if hit else 1-b
            prefix=2*prefix+hit
        total += abs(prob_p-prob_q)
    return total/2


def symmetric_kl(p_difference):
    """Numerical KL(Ber(1/2) || Ber(1/2+delta)); NOT a rational certificate."""
    delta=float(Q(p_difference))
    if not 0 < delta < 0.5:
        raise ValueError("delta in (0,1/2) required")
    return -0.5*log(1-4*delta*delta)


def pinsker_tv_upper(draws, delta):
    if type(draws) is not int or draws < 0:
        raise ValueError("draws must be a nonnegative integer")
    return min(1.0,sqrt(draws*symmetric_kl(delta)/2))


def pinsker_budget_lower(delta, desired_average_error):
    """Numerical necessary lower bound to attain target Bayes error.

    Uses Pinsker with KL chain rule. This is an approximate float; the
    exact finite Bayes risk is evaluated by binomial_bayes_risk.
    """
    error=float(Q(desired_average_error))
    if not 0 <= error < 0.5:
        raise ValueError("desired average error in [0,1/2) required")
    return (1-2*error)**2*2/symmetric_kl(delta)
