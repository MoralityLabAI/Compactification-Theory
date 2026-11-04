"""QO-07: anytime-valid active probes for goal-visible hit/hold quotients.

This module gives exact-rational, outer-rounded beta-binomial mixture
confidence sets on addressable microstate/action Bernoulli streams,
a finite robust quotient optimizer, and synthetic probe schedulers.

Coverage relies on each cell's infinite draw stream being i.i.d. Bernoulli
at fixed p; an adaptive policy must choose its next cell BEFORE its draw.
States must be separately addressable and resettable. Hidden-state POMDP
identification is outside this contract.
"""
from fractions import Fraction as Q
from math import comb, prod
from random import Random

from research.ingression_dynamics_v1.goal_aware_partition import partitions


def _prob(x):
    p = Q(x)
    if not Q(0) <= p <= Q(1):
        raise ValueError("probability must be in [0,1]")
    return p


def mixture_evalue(trials, successes, null_probability):
    """Uniform-Beta Bernoulli-mixture likelihood ratio at a fixed sample path.

    Return None if the asserted null gives this observed path probability 0.
    """
    if type(trials) is not int or type(successes) is not int or (
            trials < 0 or successes < 0 or successes > trials):
        raise ValueError("invalid Bernoulli counts")
    p = _prob(null_probability)
    denominator = p ** successes * (1 - p) ** (trials - successes)
    if denominator == 0:
        return None
    return Q(1, (trials + 1) * comb(trials, successes)) / denominator


def anytime_outer_interval(trials, successes, alpha=Q(1, 20),
                           simultaneous_cells=1, bisection_bits=12):
    """Conservative rational bracket for a beta-binomial e-process CS.

    The superlevel set {p: M_t(p) <= cells/alpha} is a closed interval.
    Bisection returns an OUTER bracket; never round a bound inward.
    Intersect across times externally to get a nested confidence sequence.
    """
    if type(trials) is not int or type(successes) is not int or (
            trials < 0 or successes < 0 or successes > trials):
        raise ValueError("invalid Bernoulli counts")
    alpha = Q(alpha)
    if not (0 < alpha < 1) or type(simultaneous_cells) is not int or (
            simultaneous_cells < 1) or type(bisection_bits) is not int or (
            bisection_bits < 1):
        raise ValueError("invalid confidence sequence parameters")
    if trials == 0:
        return Q(0), Q(1)
    critical = alpha / (simultaneous_cells * (trials + 1)
                        * comb(trials, successes))
    mode = Q(successes, trials)

    def inside(p):
        return p ** successes * (1 - p) ** (trials - successes) >= critical

    assert inside(mode)
    left = Q(0)
    if successes > 0:
        lo, hi = Q(0), mode
        for _ in range(bisection_bits):
            mid = (lo + hi) / 2
            if inside(mid):
                hi = mid
            else:
                lo = mid
        left = lo
    right = Q(1)
    if successes < trials:
        lo, hi = mode, Q(1)
        for _ in range(bisection_bits):
            mid = (lo + hi) / 2
            if inside(mid):
                lo = mid
            else:
                hi = mid
        right = hi
    return left, right


class ProbeLedger:
    """Only observed Bernoulli hits enter the estimator and planner."""
    def __init__(self, microstates, actions, alpha=Q(1, 20), bisection_bits=12):
        if (type(microstates) is not int or microstates < 1 or
                type(actions) is not int or actions < 1):
            raise ValueError("positive microstate and action counts required")
        self.n = microstates
        self.d = actions
        self.alpha = Q(alpha)
        self.bits = bisection_bits
        anytime_outer_interval(0, 0, self.alpha, self.n*self.d, self.bits)
        self.counts = [[0]*self.d for _ in range(self.n)]
        self.hits = [[0]*self.d for _ in range(self.n)]
        self.bounds = [[(Q(0), Q(1)) for _ in range(self.d)]
                       for _ in range(self.n)]
        self.receipts = []

    def observe(self, state, action, hit):
        if (type(state) is not int or not (0 <= state < self.n) or
                type(action) is not int or not (0 <= action < self.d) or
                type(hit) is not int or hit not in (0, 1)):
            raise ValueError("probe index and observed hit must be valid integers")
        trials = self.counts[state][action] + 1
        successes = self.hits[state][action] + hit
        lo, hi = anytime_outer_interval(trials, successes, self.alpha,
                                       self.n*self.d, self.bits)
        old_lo, old_hi = self.bounds[state][action]
        new = (max(lo, old_lo), min(hi, old_hi))
        if new[0] > new[1]:
            raise ValueError("empty confidence sequence after unexpected data")
        self.counts[state][action] = trials
        self.hits[state][action] = successes
        self.bounds[state][action] = new
        self.receipts.append((state, action, hit))
        return new

    def snapshot(self):
        return tuple(tuple(x for x in row) for row in self.bounds)

    def query_count(self):
        return len(self.receipts)


def validate_bounds(bounds):
    rows = tuple(tuple((Q(lo), Q(hi)) for lo, hi in row) for row in bounds)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("nonempty rectangular interval matrix required")
    if any(not (0 <= lo <= hi <= 1) for row in rows for lo, hi in row):
        raise ValueError("interval endpoints must be ordered probabilities")
    return rows


def partition_interval_bounds(bounds, labels):
    """For a fixed partition, return (best robust frozen error,
    optimistic bound on true optimal error, robust midpoint macro rates).

    The uncertainty set is rectangular across all cells, so each robust
    midpoint/radius can be optimized independently.
    """
    bounds = validate_bounds(bounds)
    if len(labels) != len(bounds):
        raise ValueError("one transient macrostate label per microstate required")
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    radius = Q(0)
    optimistic = Q(0)
    centers = {}
    for z, inds in groups.items():
        for action in range(len(bounds[0])):
            low = min(bounds[i][action][0] for i in inds)
            high = max(bounds[i][action][1] for i in inds)
            max_lower = max(bounds[i][action][0] for i in inds)
            min_upper = min(bounds[i][action][1] for i in inds)
            radius = max(radius, (high-low)/2)
            optimistic = max(optimistic, max(Q(0), max_lower-min_upper)/2)
            centers[(z, action)] = (low+high)/2
    return radius, optimistic, centers


def robust_optimum(bounds, max_transient_classes):
    """Exact finite minmax over all partitions, for small n only.

    Returns a posterior-data-only (upper, lower, labels, midpoint rates)
    certificate. The common goal occupies a separate macrostate.
    """
    bounds = validate_bounds(bounds)
    if type(max_transient_classes) is not int or max_transient_classes < 1:
        raise ValueError("positive class budget required")
    n = len(bounds)
    if n > 8:
        raise ValueError("exhaustive Bell-partition oracle limited to 8 states")
    best = None
    lower_star = None
    for p in partitions(tuple(range(n))):
        classes = len(set(p.values()))
        if classes > max_transient_classes:
            continue
        labels = tuple(p[i] for i in range(n))
        upper, lower, centers = partition_interval_bounds(bounds, labels)
        lower_star = lower if lower_star is None else min(lower_star, lower)
        key = (upper, classes, labels)
        if best is None or key < best[0]:
            best = key, centers
    assert best is not None
    upper, _, labels = best[0]
    assert lower_star <= upper
    return upper, lower_star, labels, best[1]


def certificate(bounds, max_transient_classes, tolerance):
    upper, lower, labels, centers = robust_optimum(bounds, max_transient_classes)
    tolerance = _prob(tolerance)
    if upper <= tolerance:
        conclusion = "certified-feasible"
    elif lower > tolerance:
        conclusion = "certified-infeasible"
    else:
        conclusion = "undetermined"
    widths = tuple((hi-lo for row in validate_bounds(bounds) for lo, hi in row))
    assert upper-lower <= max(widths)
    return {
        "decision": conclusion,
        "robust_macro_error_upper": upper,
        "true_optimum_lower": lower,
        "optimality_gap_upper": upper-lower,
        "transient_labels": labels,
        "goal_label": len(set(labels)),
        "mutable_label_bits": len(set(labels)).bit_length(),
        "frozen_midpoints": centers,
        "maximum_interval_width": max(widths),
    }


def choose_probe(ledger, max_transient_classes, policy="bottleneck"):
    """Heuristic probe selection using evidence only, before the next draw.

    Neither the bottleneck rule nor width sampling has a claimed statistical
    efficiency advantage. Anytime coverage holds for ANY predictable rule.
    """
    if policy not in ("bottleneck", "width", "round_robin"):
        raise ValueError("unknown sampling policy")
    bounds = ledger.snapshot()
    candidates = [(i, a) for i in range(ledger.n) for a in range(ledger.d)]
    if policy == "round_robin":
        return min(candidates, key=lambda ia: (ledger.counts[ia[0]][ia[1]], ia))
    def width(ia):
        lo, hi = bounds[ia[0]][ia[1]]
        return hi-lo
    if policy == "width":
        return min(candidates,
                   key=lambda ia: (-width(ia), ledger.counts[ia[0]][ia[1]], ia))
    upper, _, labels, _ = robust_optimum(bounds, max_transient_classes)
    bottlenecks = set()
    for z in set(labels):
        inds = [i for i in range(ledger.n) if labels[i] == z]
        for a in range(ledger.d):
            lo = min(bounds[i][a][0] for i in inds)
            hi = max(bounds[i][a][1] for i in inds)
            if (hi-lo)/2 == upper:
                bottlenecks.update((i,a) for i in inds if
                                   bounds[i][a][0] == lo or bounds[i][a][1] == hi)
    choices = bottlenecks or set(candidates)
    return min(choices, key=lambda ia: (-width(ia),
                                      ledger.counts[ia[0]][ia[1]], ia))


def frozen_error(true_rates, labels, fitted_midpoints):
    """Audit-only true uniform one-step TV defect of a frozen macro kernel."""
    return max(abs(Q(p)-fitted_midpoints[(labels[i],a)])
               for i,row in enumerate(true_rates) for a,p in enumerate(row))


def run_synthetic(rates, budget, max_transient_classes, policy="bottleneck",
                  seed=20261104, alpha=Q(1,20), tolerance=Q(1,4),
                  bisection_bits=12, stop_on_certificate=False):
    """Paired seeded *generative* digital experiment, no physical inference.

    Each cell has an independent pre-generated i.i.d. Bernoulli stream.
    Only queried observations enter the estimator; true rates are audit-only.
    """
    rates = tuple(tuple(_prob(x) for x in row) for row in rates)
    if not rates or not rates[0] or any(len(row) != len(rates[0]) for row in rates):
        raise ValueError("nonempty rectangular true rate table required")
    if type(budget) is not int or budget < 0:
        raise ValueError("nonnegative integer budget required")
    ledger = ProbeLedger(len(rates), len(rates[0]), alpha, bisection_bits)
    streams = {(i,a): Random(int(seed)*1000003 + i*9907 + a*7919)
               for i in range(ledger.n) for a in range(ledger.d)}
    for _ in range(budget):
        if stop_on_certificate and certificate(ledger.snapshot(),
                max_transient_classes, tolerance)["decision"] != "undetermined":
            break
        i,a = choose_probe(ledger, max_transient_classes, policy)
        outcome = int(streams[i,a].random() < float(rates[i][a]))
        ledger.observe(i,a,outcome)
    report = certificate(ledger.snapshot(), max_transient_classes, tolerance)
    report["observed_queries"] = ledger.query_count()
    report["counts"] = tuple(tuple(row) for row in ledger.counts)
    report["hits"] = tuple(tuple(row) for row in ledger.hits)
    report["receipts"] = tuple(ledger.receipts)
    report["covered_true_rates"] = all(
        lo <= rates[i][a] <= hi
        for i,row in enumerate(ledger.bounds) for a,(lo,hi) in enumerate(row))
    report["true_frozen_error"] = frozen_error(
        rates, report["transient_labels"], report["frozen_midpoints"])
    return report


def hidden_alias_word_probability(rates, outcomes):
    """Non-addressable hidden state is redrawn uniformly before EACH trial.

    Rates (0,1) and (1/2,1/2) then produce identical Bernoulli(1/2)
    observations for every sequence, despite distinct exact quotients.
    """
    rates = tuple(_prob(x) for x in rates)
    if not rates or any(type(o) is not int or o not in (0,1) for o in outcomes):
        raise ValueError("valid rates and observed hit bits required")
    marginal = sum(rates, Q(0))/len(rates)
    return prod((marginal if x else 1-marginal for x in outcomes), start=Q(1))
