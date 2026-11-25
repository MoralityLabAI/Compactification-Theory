"""Independent verification of fg.py. The reference here is a naive full-joint
product over all n variables (no component restriction, no elimination), written
separately from every routine under test."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fg  # noqa: E402
from gen import FAMILIES, make_problem  # noqa: E402


def naive(prob, tables=None):
    tables = prob.tables if tables is None else tables
    num = den = 0.0
    for x in itertools.product((0, 1), repeat=prob.n):
        w = 1.0
        for s, t in zip(prob.scopes, tables):
            w *= t[tuple(x[v] for v in s)]
        den += w
        if sum(x[v] for v in prob.query) % 2:
            num += w
    return num / den


CASES = [(f, n, s, q) for f in FAMILIES for n in (6, 9) for s in (1, 2) for q in (1, 2, 3)]


@pytest.mark.parametrize('family,n,seed,q', CASES)
def test_exact_paths_agree_with_naive(family, n, seed, q):
    prob = make_problem(family, n, 1000 * seed + n, noise='high' if seed == 2 else 'low', query_size=q)
    ref = naive(prob)
    for h in ('min_degree', 'min_fill', 'weighted_min_fill'):
        r = fg.variable_elimination(prob, heuristic=h)
        assert abs(r['p'] - ref) < 1e-9, (h, r['p'], ref)
        assert r['ops'] == fg.symbolic_ve(*_sym_args(prob, h))['ops']
    assert abs(fg.enumerate_exact(prob)['p'] - ref) < 1e-9
    assert abs(fg.ve_broadcast(prob)['p'] - ref) < 1e-9
    assert abs(fg.cutset_exact(prob)['p'] - ref) < 1e-9
    if fg.gf2_applicable(prob):
        assert abs(fg.gf2_exact(prob)['p'] - ref) < 1e-9


def _sym_args(prob, h):
    comp, variables = fg.query_component(prob)
    order, _ = fg.elimination_order(variables, comp.scopes, comp.query, h)
    return comp.scopes, order, comp.query


@pytest.mark.parametrize('family', ['chain', 'tree', 'star'])
def test_bp_exact_on_trees_for_single_queries(family):
    for seed in range(5):
        prob = make_problem(family, 10, seed, query_size=1)
        r = fg.loopy_bp(prob, max_iters=200, damping=0.0, tol=1e-12)
        assert abs(r['p'] - naive(prob)) < 1e-8


def test_mini_bucket_symbolic_ops_and_exact_at_full_ibound():
    for seed in range(6):
        prob = make_problem('grid', 9, seed, query_size=2)
        for ib in (1, 2, 3, 8):
            r = fg.mini_bucket(prob, ib)
            assert r['ops'] == fg.mini_bucket_ops(prob, ib)
        assert abs(fg.mini_bucket(prob, 20)['p'] - naive(prob)) < 1e-9


def test_budget_declines_before_work():
    prob = make_problem('dense', 12, 3, query_size=2)
    r = fg.variable_elimination(prob, budget_entries=4)
    assert r['declined'] and r['ops'] == 0


def test_particles_converge():
    prob = make_problem('sparse', 10, 4, query_size=2)
    r = fg.likelihood_weighting(prob, n_particles=200000, seed=1)
    assert abs(r['p'] - naive(prob)) < 4 * r['half_width'] / 2 + 1e-3


def test_misspecified_model_differs_from_truth_reference():
    prob = make_problem('grid', 9, 5, misspec=True, query_size=1)
    assert prob.true_tables is not None
    a, b = naive(prob), naive(prob, prob.true_tables)
    assert abs(fg.variable_elimination(prob)['p'] - a) < 1e-9
    assert abs(fg.variable_elimination(prob.with_tables(prob.true_tables))['p'] - b) < 1e-9


def test_decision_threshold():
    assert fg.decide(0.8) == 1 and fg.decide(0.2) == 0 and fg.decide(0.5) == -1
    assert fg.decide(0.85, penalty=9) == -1 and fg.decide(0.9, penalty=9) == 1
    for p in np.linspace(0, 1, 101):
        a = fg.decide(p)
        assert fg.expected_utility(a, p) >= max(fg.expected_utility(b, p) for b in (-1, 0, 1)) - 1e-12


def test_many_operand_buckets():
    prob = make_problem('star', 80, 11, query_size=2)
    r = fg.variable_elimination(prob)
    assert abs(r['p'] - fg.cutset_exact(prob)['p']) < 1e-9
    assert abs(r['p'] - fg.ve_broadcast(prob)['p']) < 1e-9
    small = make_problem('star', 12, 11, query_size=2)
    assert abs(fg._ein_labels([(t, list(s)) for s, t in zip(small.scopes, small.tables)], [0], max_ops=3).sum()
               - fg._ein_labels([(t, list(s)) for s, t in zip(small.scopes, small.tables)], [0]).sum()) < 1e-12
