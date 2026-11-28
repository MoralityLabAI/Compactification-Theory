"""Exhaustive finite-state checks plus exact rational counterexamples."""
import unittest
from fractions import Fraction as Q
from itertools import product
from closure import closure_risk, closure_witness, partitions
from task_closure import (FiniteSystem, bayes_risk, canonical, is_task_closed,
                          minimax_risk, quotient, refinement_history, refines,
                          run_word, shortest_witness, task_loss_witness, minimum_bits)


def words(actions, horizon):
    for length in range(horizon + 1):
        yield from product(range(actions), repeat=length)


class TaskClosureTests(unittest.TestCase):
    def test_exhaustive_two_action_minimality_and_horizon(self):
        systems = candidates = 0
        for n in range(1, 4):
            maps = tuple(product(range(n), repeat=n))
            for transitions in product(maps, repeat=2):
                for task in product(range(2), repeat=n):
                    system = FiniteSystem(transitions, task)
                    history = refinement_history(system)
                    p = history[-1]
                    self.assertLessEqual(len(history) - 1, n - len(set(task)))
                    for h, stage in enumerate(history):
                        signatures = tuple(tuple(task[run_word(system, x, w)]
                                                 for w in words(2, h)) for x in range(n))
                        self.assertEqual(stage, canonical(signatures))
                    self.assertTrue(is_task_closed(system, p))
                    macro = quotient(system, p)
                    for x in range(n):
                        for w in words(2, n):
                            self.assertEqual(p[run_word(system, x, w)], run_word(macro, p[x], w))
                    for c in partitions(n):
                        candidates += 1
                        witness = task_loss_witness(system, c)
                        self.assertEqual(witness is None, refines(c, p))
                        if witness is not None:
                            x, y, w = witness
                            self.assertEqual(c[x], c[y])
                            self.assertNotEqual(task[run_word(system, x, w)], task[run_word(system, y, w)])
                        if is_task_closed(system, c):
                            self.assertTrue(refines(c, p))
                    systems += 1
        self.assertEqual(systems, 5898)
        self.assertEqual(candidates, 29290)

    def test_exhaustive_witness_soundness_shortestness_and_bound(self):
        for n in range(1, 4):
            maps = tuple(product(range(n), repeat=n))
            for transitions in product(maps, repeat=2):
                for task in product(range(2), repeat=n):
                    system = FiniteSystem(transitions, task)
                    p = refinement_history(system)[-1]
                    for x in range(n):
                        for y in range(x):
                            witness = shortest_witness(system, x, y)
                            if p[x] == p[y]:
                                self.assertIsNone(witness)
                            else:
                                self.assertIsNotNone(witness)
                                self.assertLessEqual(len(witness), n - len(set(task)))
                                self.assertNotEqual(task[run_word(system, x, witness)],
                                                    task[run_word(system, y, witness)])
                                for w in words(2, len(witness) - 1):
                                    self.assertEqual(task[run_word(system, x, w)], task[run_word(system, y, w)])

    def test_nonconstant_task_preserving_reclosure(self):
        f, task = (0, 2, 0, 3), (0, 0, 0, 1)
        system = FiniteSystem((f,), task)
        path = ((0, 1, 2, 3), (0, 0, 1, 2), task)
        self.assertEqual([closure_risk(f, p) for p in path], [0, Q(1, 8), 0])
        self.assertEqual([closure_witness(f, p) is None for p in path], [True, False, True])
        for p in path:
            self.assertTrue(refines(p, refinement_history(system)[-1]))
            for w in words(1, 8):
                targets = tuple(task[run_word(system, x, w)] for x in range(4))
                self.assertEqual(bayes_risk(p, targets), 0)
        with self.assertRaises(ValueError):
            quotient(system, path[1])

    def test_intervention_reveals_hidden_bit(self):
        idle, reveal, task = (0, 1, 2, 3), (0, 3, 0, 3), (0, 0, 1, 1)
        self.assertEqual(len(set(refinement_history(FiniteSystem((idle,), task))[-1])), 2)
        system = FiniteSystem((idle, reveal), task)
        self.assertEqual(refinement_history(system)[-1], (0, 1, 2, 3))
        self.assertEqual(shortest_witness(system, 0, 1), (1,))
        self.assertEqual(minimum_bits(FiniteSystem((idle,), task)), 1)
        self.assertEqual(minimum_bits(system), 2)
        self.assertIsNotNone(task_loss_witness(system, task))
        future = tuple(task[reveal[x]] for x in range(4))
        self.assertEqual(bayes_risk(task, task), 0)
        self.assertEqual(bayes_risk(task, future), Q(1, 4))
        self.assertEqual(minimax_risk(task, future), Q(1, 4))
        self.assertEqual(bayes_risk(task, future, (1, 0, 0, 1)), 0)

    def test_exhaustive_action_family_refinement(self):
        for transitions in product(tuple(product(range(3), repeat=3)), repeat=2):
            for task in product(range(2), repeat=3):
                small = refinement_history(FiniteSystem(transitions[:1], task))[-1]
                large = refinement_history(FiniteSystem(transitions, task))[-1]
                self.assertTrue(refines(large, small))

    def test_exhaustive_fixed_target_monotonicity(self):
        for n in range(1, 5):
            ps = tuple(partitions(n))
            for targets in product(range(2), repeat=n):
                for fine in ps:
                    for coarse in ps:
                        if refines(fine, coarse):
                            for weights in ((1,) * n, tuple(range(1, n + 1))):
                                self.assertLessEqual(bayes_risk(fine, targets, weights),
                                                     bayes_risk(coarse, targets, weights))
                            self.assertLessEqual(minimax_risk(fine, targets), minimax_risk(coarse, targets))

    def test_exact_minimax_midpoint_and_weighted_variance(self):
        self.assertEqual(minimax_risk((0, 0), (-2, 4)), 9)
        self.assertEqual(bayes_risk((0, 0), (-2, 4), (1, 3)), Q(27, 4))
        self.assertEqual(bayes_risk((0, 0), (Q(1, 3), Q(2, 3))), Q(1, 36))

    def test_finite_horizon_can_require_more_state(self):
        system = FiniteSystem(((1, 2, 3, 3),), (0, 0, 0, 1))
        history = refinement_history(system)
        self.assertEqual(tuple(len(set(p)) for p in history), (2, 3, 4))
        self.assertEqual(shortest_witness(system, 0, 1), (0, 0))
        self.assertFalse(is_task_closed(system, history[1]))

    def test_empty_action_family_and_constant_task(self):
        self.assertEqual(refinement_history(FiniteSystem((), (0, 0, 1))), ((0, 0, 1),))
        system = FiniteSystem(((1, 2, 0),), (0, 0, 0))
        self.assertEqual(refinement_history(system), ((0, 0, 0),))
        self.assertIsNone(shortest_witness(system, 0, 1))
        self.assertEqual(len(quotient(system, (0, 0, 0)).task), 1)
        self.assertEqual(minimum_bits(system), 0)

    def test_invalid_inputs(self):
        for transitions, task in (((), ()), (((0, 2),), (0, 1)), (((0,),), (0, 1))):
            with self.assertRaises(ValueError):
                FiniteSystem(transitions, task)
        system = FiniteSystem(((0, 1),), (0, 1))
        for state, word in ((-1, ()), (2, ()), (0, (-1,)), (0, (1,))):
            with self.assertRaises(ValueError):
                run_word(system, state, word)
        for p, ys, ws in (((0,), (0, 1), None), ((0, 0), (0, 1), (0, 0)),
                          ((0, 0), (0, 1), (-1, 2)), ((0,), (0.5,), None)):
            with self.assertRaises(ValueError):
                bayes_risk(p, ys, ws)
        with self.assertRaises(ValueError):
            canonical(([0],))
        with self.assertRaises(ValueError):
            canonical((float('nan'),))


if __name__ == '__main__':
    unittest.main()
