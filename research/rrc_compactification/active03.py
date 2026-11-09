"""RRC-03 exact active-query identification of delayed Euler-product defects.

Every candidate is identical through degree 64. The query oracle returns one
binary comparison of a queried coefficient to the publicly declared baseline.
This is an active *extension* of RRC-02; it does not constitute an identity proof.
Pure Python, with no GPU or model dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

try:
    from .rrc import ResidueRule, product_coefficients, gap_coefficients
except ImportError:
    from rrc import ResidueRule, product_coefficients, gap_coefficients

PREFIX_DEGREE = 64
DEFAULT_HYPOTHESES = 33
MAX_HYPOTHESES = 65
MAX_QUERIES = 6
STUDENT_H_SIZES = (13, 25, 49)
HERE = Path(__file__).resolve().parent


def candidate_defect(idx: int, size: int) -> int | None:
    """Last hypothesis is the unchanged baseline (an infinite defect degree)."""
    if size < 2 or size > MAX_HYPOTHESES or not 0 <= idx < size:
        raise ValueError('invalid candidate or hypothesis count')
    return None if idx == size - 1 else PREFIX_DEGREE + idx + 1


@lru_cache(maxsize=4)
def baseline(degree: int) -> tuple[int, ...]:
    """Known period-5 generator; the allowed part 1 implies f_n>0 at all n."""
    if degree < 0 or degree > PREFIX_DEGREE + MAX_HYPOTHESES:
        raise ValueError('out-of-range degree')
    answer = tuple(product_coefficients(degree, ResidueRule(5, (1, 4))))
    assert all(v >= 1 for v in answer)
    return answer


def mutant_coefficient(idx: int, size: int, n: int) -> int:
    """Exact F(q)/(1-q^K) coefficient from independent convolution."""
    if not PREFIX_DEGREE + 1 <= n <= PREFIX_DEGREE + size - 1:
        raise ValueError('degree outside active boundary')
    k = candidate_defect(idx, size)
    f = baseline(n)
    return f[n] if k is None else sum(f[n - j*k] for j in range(n//k + 1))


def query_bit(idx: int, size: int, threshold: int) -> int:
    """One paid observation, bit = 1 iff candidate's coefficient exceeds base.

    Threshold is the zero-based mutant index; actual degree is 65+threshold.
    In this task, positivity of baseline coefficients proves a threshold oracle.
    """
    if not 0 <= threshold < size - 1:
        raise ValueError('illegal query index')
    n = PREFIX_DEGREE + 1 + threshold
    return int(mutant_coefficient(idx, size, n) > baseline(n)[n])


def posterior_interval(history: tuple[tuple[int, int], ...], size: int) -> tuple[int, int]:
    lo, hi = 0, size - 1
    for t, response in history:
        if not 0 <= t < size - 1 or response not in (0, 1):
            raise ValueError('malformed observation')
        if response:
            hi = min(hi, t)
        else:
            lo = max(lo, t + 1)
        if lo > hi:
            raise ValueError('inconsistent evidence')
    return lo, hi


def midpoint_policy(history: tuple[tuple[int, int], ...], size: int, budget: int) -> int:
    lo, hi = posterior_interval(history, size)
    if lo == hi:
        return max(0, min(size - 2, lo))
    return (lo + hi - 1) // 2


def fixed_thresholds(size: int, budget: int) -> tuple[int, ...]:
    """B distinct, evenly separated thresholds; best count of static leaves."""
    if budget < 0:
        raise ValueError('negative query budget')
    n = min(size - 1, budget)
    return tuple((i * size)//(n + 1) - 1 for i in range(1, n + 1))


def static_policy(history: tuple[tuple[int, int], ...], size: int, budget: int) -> int:
    thresholds = fixed_thresholds(size, budget)
    return thresholds[len(history)] if len(history) < len(thresholds) else size - 2


@dataclass(frozen=True)
class Trace:
    actual: int
    predicted: int
    observations: tuple[tuple[int, int], ...]
    posterior: tuple[int, int]
    queries_used: int
    wasted_queries: int

    @property
    def exact(self) -> bool:
        return self.actual == self.predicted


def run_policy(actual: int, size: int, budget: int,
               policy: Callable[[tuple[tuple[int, int], ...], int, int], int]) -> Trace:
    """Policy sees previous observations and public size/budget, not candidate."""
    candidate_defect(actual, size)
    if budget < 0 or budget > MAX_QUERIES:
        raise ValueError('budget outside registered range')
    history: tuple[tuple[int, int], ...] = ()
    wasted = 0
    for _ in range(budget):
        lo, hi = posterior_interval(history, size)
        if lo == hi:
            break
        t = policy(history, size, budget)
        if not isinstance(t, int) or not 0 <= t < size - 1:
            raise ValueError('policy emitted illegal action')
        if t < lo or t >= hi:
            wasted += 1
        history += ((t, query_bit(actual, size, t)),)
    interval = posterior_interval(history, size)
    return Trace(actual, interval[0], history, interval, len(history), wasted)


def all_policy_metrics(size: int, budget: int, policy) -> dict:
    traces = [run_policy(i, size, budget, policy) for i in range(size)]
    n_exact = sum(t.exact for t in traces)
    distinct = {tuple(t.observations) for t in traces}
    return {
        'hypotheses': size, 'budget': budget, 'correct': n_exact,
        'accuracy_num': n_exact, 'accuracy_den': size,
        'transcripts': len(distinct),
        'wasted_queries': sum(t.wasted_queries for t in traces),
        'total_queries': sum(t.queries_used for t in traces),
    }


def optimum_success_mass(weights: tuple[int, ...], budget: int) -> int:
    """Independent exact Bayesian decision tree oracle, unnormalized integer mass.

    Each query divides an ordered *interval* into nonempty left/right outcomes.
    Terminal MAP chooses the largest posterior weight. Explicitly caches the
    finite Bellman recursion for arbitrary positive integer priors.
    """
    if not weights or any((not isinstance(w, int)) or w < 1 for w in weights):
        raise ValueError('positive integer prior weights required')
    if budget < 0:
        raise ValueError('negative query budget')
    size = len(weights)

    @lru_cache(None)
    def value(lo: int, hi: int, remaining: int) -> int:
        if lo == hi or remaining == 0:
            return max(weights[lo:hi + 1])
        return max(value(lo, t, remaining - 1) + value(t + 1, hi, remaining - 1)
                   for t in range(lo, hi))

    return value(0, size - 1, budget)


def static_success_bound(size: int, budget: int) -> int:
    return min(size, budget + 1)


def adaptive_success_bound(size: int, budget: int) -> int:
    return min(size, 2 ** budget)


def validate_series(size: int = DEFAULT_HYPOTHESES) -> dict:
    """Cross-check two counting algorithms and the claimed prefix collision."""
    try:
        from .rrc02 import coefficients_from_exponents, exponent_mask
    except ImportError:
        from rrc02 import coefficients_from_exponents, exponent_mask
    end = PREFIX_DEGREE + size - 1
    f = baseline(end)
    g = gap_coefficients(end, gap=2, minimum_part=1)
    if f != tuple(g):
        raise AssertionError('reference Rogers-Ramanujan identity fails finite check')
    comparisons = 0
    for idx in range(size):
        k = candidate_defect(idx, size)
        a = exponent_mask(5, (1, 4), end)
        if k is not None:
            a[k] += 1
        exponents = coefficients_from_exponents(a)
        for n in range(end + 1):
            observed = f[n] if k is None else sum(f[n-j*k] for j in range(n//k+1))
            if observed != exponents[n]:
                raise AssertionError(f'convolution/Euler disagreement at candidate {idx}, degree {n}')
            if n <= PREFIX_DEGREE and observed != f[n]:
                raise AssertionError('observed prefix collision broken')
            comparisons += 1
        for t in range(size - 1):
            if query_bit(idx, size, t) != int(idx < size - 1 and idx <= t):
                raise AssertionError('query not a threshold bit')
    return {'coefficient_comparisons': comparisons, 'independent_oracles_match': True,
            'prefix_identical_through': PREFIX_DEGREE,
            'binary_comparisons_checked': size * (size-1)}


def experiment_without_students() -> dict:
    series = validate_series()
    evaluations = []
    for budget in range(MAX_QUERIES+1):
        static = all_policy_metrics(DEFAULT_HYPOTHESES, budget, static_policy)
        adapt = all_policy_metrics(DEFAULT_HYPOTHESES, budget, midpoint_policy)
        optimal = optimum_success_mass((1,)*DEFAULT_HYPOTHESES, budget)
        if not (static['correct'] == static_success_bound(DEFAULT_HYPOTHESES, budget)
                and adapt['correct'] == adaptive_success_bound(DEFAULT_HYPOTHESES, budget)
                and adapt['correct'] == optimal):
            raise AssertionError(f'adaptive/static certificate failed at B={budget}')
        evaluations.append({'budget': budget, 'adaptive': adapt, 'static': static, 'bellman_optimum_success_mass': optimal})
    return {'schema': 'rrc03.active.v1', 'series_audit': series,
            'candidate_family': {'base_residues_mod_5': [1,4], 'first_mutant_degree': 65,
                                 'last_mutant_degree': 96, 'unchanged_base_is_last_hypothesis': True,
                                 'hypotheses': DEFAULT_HYPOTHESES, 'uniform_prior': True},
            'baseline': evaluations}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    result = experiment_without_students()
    result['sources_sha256'] = {
        file: hashlib.sha256((HERE / file).read_bytes()).hexdigest()
        for file in ('rrc.py', 'rrc02.py', 'active03.py')
    }
    payload = json.dumps(result, sort_keys=True, indent=2)+'\n'
    if args.check:
        if args.out is None or not args.out.exists() or args.out.read_text() != payload:
            raise SystemExit('missing or stale exact active oracle receipt')
        print('RRC-03 exact active oracle receipt verified')
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f'wrote {args.out}')
    else:
        print(payload)


if __name__ == '__main__':
    main()
