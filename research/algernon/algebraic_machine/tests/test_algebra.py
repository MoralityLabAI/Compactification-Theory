"""Finite, exact and first-order tests for Algernon's circuit-algebra program."""
import itertools
import random
import unittest

import numpy as np

from research.algernon.algebraic_machine.machine import (
    congruence_defects, exhaustive_word_check, lifted_actions, minimal_period,
    run_word, semiconjugacy_defects, words,
)
from research.algernon.algebraic_machine.sheaf import (
    affine_cocycle, coboundary, cycle_cohomology, eye, matmul, matvec,
    monodromy, rank, solve,
)
from research.algernon.algebraic_machine.sparse import (
    intertwining_defect, sparse_repair, step_error_bound,
)


class TestInstructionMonoid(unittest.TestCase):
    def test_exhaustive_composition_for_every_three_state_map(self):
        maps = list(itertools.product(range(3), repeat=3))
        for t in maps:
            actions = (t, (1, 2, 0))
            hidden, pi = lifted_actions(actions, tag_count=2)
            self.assertFalse(semiconjugacy_defects(hidden, actions, pi))
            self.assertFalse(congruence_defects(hidden, pi))
            report = exhaustive_word_check(hidden, actions, pi, max_length=5)
            self.assertEqual(report['program_checks'], 6 * sum(2 ** k for k in range(6)))
            self.assertEqual(report['program_violations'], 0)

    def test_corrupted_hidden_transition_exposed(self):
        actions = ((1, 2, 0), (2, 0, 1))
        hidden, pi = lifted_actions(actions)
        broken = [list(f) for f in hidden]
        broken[0][0] = 0  # semantic state 0 rather than correct state 1
        self.assertIn((0, 0), semiconjugacy_defects(broken, actions, pi))
        self.assertGreater(exhaustive_word_check(broken, actions, pi, 4)['program_violations'], 0)
        self.assertTrue(congruence_defects(broken, pi))

    def test_cycle_period_divisibility(self):
        t = (1, 2, 3, 4, 0)
        f, pi = lifted_actions((t,), tag_count=2)
        self.assertEqual(minimal_period(t, 0), 5)
        self.assertEqual(minimal_period(f[0], 0), 10)
        self.assertEqual(minimal_period(f[0], 0) % minimal_period(t, pi[0]), 0)
        self.assertNotEqual(minimal_period(t, 0) % 2, 0)

    def test_bad_word(self):
        with self.assertRaises(ValueError):
            run_word([(1, 0)], [1], 0)
        with self.assertRaises(ValueError):
            list(words(2, -1))


class TestCyclicSheaf(unittest.TestCase):
    def test_three_edge_affine_obstruction_mod_five(self):
        edges = [[[1]] for _ in range(3)]
        c = cycle_cohomology(edges, 5)
        self.assertEqual(c['H0_dimension'], 1)
        self.assertEqual(c['H1_dimension'], 1)
        no = affine_cocycle(edges, [[1], [1], [1]], 5)
        self.assertFalse(no['solvable'])
        self.assertEqual(no['obstruction'], [3])
        yes = affine_cocycle(edges, [[1], [1], [3]], 5)
        self.assertTrue(yes['solvable'])
        self.assertEqual(len(yes['states']), 3)

    def test_twisted_cycle_has_vanishing_cohomology(self):
        c = cycle_cohomology([[[1]], [[1]], [[2]]], 5)
        self.assertEqual((c['H0_dimension'], c['H1_dimension']), (0, 0))
        self.assertTrue(affine_cocycle([[[1]], [[1]], [[2]]], [[1], [1], [1]], 5)['solvable'])

    def test_square_rank_and_monodromy_exhaustive_small(self):
        for p in (2, 3, 5, 7):
            for a, b, c in itertools.product(range(p), repeat=3):
                edges = [[[a]], [[b]], [[c]]]
                rep = cycle_cohomology(edges, p)
                self.assertEqual(rep['H0_dimension'], int((a*b*c) % p == 1))
                self.assertEqual(rep['H1_dimension'], rep['H0_dimension'])

    def test_two_dimensional_random_monodromy(self):
        rng = random.Random(20261104)
        for p in (5, 7):
            for _ in range(100):
                edges = [[[rng.randrange(p) for _ in range(2)] for _ in range(2)] for _ in range(4)]
                rep = cycle_cohomology(edges, p)
                self.assertEqual(rep['H0_dimension'], 2 - rep['rank_identity_minus_monodromy'])
                b = [[rng.randrange(p) for _ in range(2)] for _ in edges]
                sol = affine_cocycle(edges, b, p)
                if sol['solvable']:
                    states = sol['states']
                    for i, a in enumerate(edges):
                        actual = matvec(a, states[i], p)
                        expected = [(v + b[i][j]) % p for j, v in enumerate(actual)]
                        self.assertEqual(states[(i+1) % len(edges)], expected)

    def test_gauge_invariance(self):
        p = 7
        g = [[0, 1], [1, 1]]
        gi = [[6, 1], [1, 0]]  # g @ gi = I mod 7
        self.assertEqual(matmul(g, gi, p), eye(2))
        edges = [[[1, 2], [3, 0]], [[0, 1], [2, 1]], [[2, 1], [1, 1]]]
        shifted = [matmul(g, matmul(a, gi, p), p) for a in edges]
        self.assertEqual(cycle_cohomology(edges, p)['H0_dimension'],
                         cycle_cohomology(shifted, p)['H0_dimension'])
        self.assertEqual(cycle_cohomology(edges, p)['H1_dimension'],
                         cycle_cohomology(shifted, p)['H1_dimension'])
        self.assertEqual(monodromy(shifted, p), matmul(g, matmul(monodromy(edges, p), gi, p), p))

    def test_single_self_loop(self):
        self.assertEqual(coboundary([[[1]]], 5), [[0]])
        self.assertEqual(coboundary([[[2]]], 5), [[4]])

    def test_unsolvable_linear_system(self):
        self.assertIsNone(solve([[0]], [1], 5))
        self.assertEqual(rank([[1, 0], [0, 1]], 5), 2)


class TestSparseIntertwining(unittest.TestCase):
    def test_exact_support_repair(self):
        P = np.eye(3)
        A = np.eye(3)
        M = A.copy()
        M[0, 2] = 3.0
        U = np.array([[1.0], [0.0], [0.0]])
        rep = sparse_repair(P, M, A, U)
        self.assertTrue(rep['exactly_repairable'])
        self.assertLess(rep['actual_residual_norm'], 1e-9)
        self.assertEqual(rep['rank_PU'], 1)
        self.assertAlmostEqual(rep['explained_energy_fraction'], 1.0)

    def test_orthogonal_support_cannot_repair(self):
        P = np.eye(3)
        A = np.eye(3)
        M = A.copy()
        M[0, 2] = 3.0
        U = np.array([[0.0], [1.0], [0.0]])
        rep = sparse_repair(P, M, A, U)
        self.assertFalse(rep['exactly_repairable'])
        self.assertAlmostEqual(rep['optimal_residual_norm'], 3.0)
        self.assertAlmostEqual(rep['explained_energy_fraction'], 0.0)

    def test_rank_zero_site_and_dimensions(self):
        P=np.array([[1.0, 0.0]])
        M=np.eye(2)
        A=np.eye(1)
        U=np.zeros((2,0))
        rep=sparse_repair(P,M,A,U)
        self.assertTrue(rep['exactly_repairable'])
        with self.assertRaises(ValueError):
            intertwining_defect(P,np.eye(3),A)

    def test_error_bound(self):
        A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=float)
        q=np.array([1., 0., 0.])
        states=[q]
        for _ in range(8):
            q = 0.98 * A@q + 0.02 * np.ones(3)/3
            states.append(q)
        rep = step_error_bound(A, states, np.array([1., 0., 0.]))
        self.assertEqual(rep['n'], 8)
        self.assertGreater(rep['truth_error'], 0)
        self.assertLessEqual(rep['truth_error'], rep['certified_bound']+1e-10)
        self.assertTrue(rep['correct_argmax_guaranteed'])

    def test_nonstochastic_rejected(self):
        with self.assertRaises(ValueError):
            step_error_bound(np.array([[2.0]]), [np.array([1.0])], np.array([1.0]))


if __name__ == '__main__':
    unittest.main()
