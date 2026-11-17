"""Exact quotient and risk checks, numerical Fourier-character sanity checks."""
import unittest
from fractions import Fraction as Q
from math import isclose

from cyclic_addition import (minimum_state_bits, onehot_risk, phase, rotate,
                             shift, states, task)
from task_closure import FiniteSystem, minimum_bits, refinement_history


class CyclicAdditionTests(unittest.TestCase):
    def test_all_additive_interventions_have_sum_as_minimal_quotient(self):
        for p in (3, 5, 7):
            xs = states(p)
            index = {x: i for i, x in enumerate(xs)}
            maps = tuple(tuple(index[shift(*x, u, v, p)] for x in xs)
                         for u in range(p) for v in range(p))
            system = FiniteSystem(maps, tuple(task(*x, p) for x in xs))
            q = refinement_history(system)[-1]
            for i, x in enumerate(xs):
                for j, y in enumerate(xs):
                    self.assertEqual(q[i] == q[j], task(*x, p) == task(*y, p))
            self.assertEqual(len(set(q)), p)
            self.assertEqual(minimum_bits(system), minimum_state_bits(p))

    def test_phase_closure_for_every_character_and_shift(self):
        for p in (3, 5, 7):
            for j in range(1, p):
                observed = [phase(s, j, p) for s in range(p)]
                self.assertEqual(len({tuple(round(v, 9) for v in z) for z in observed}), p)
                for s in range(p):
                    for c in range(p):
                        lhs, rhs = phase((s + c) % p, j, p), rotate(phase(s, j, p), c, j, p)
                        self.assertTrue(all(isclose(a, b, abs_tol=2e-14) for a, b in zip(lhs, rhs)))

    def test_onehot_risk_identical_for_any_shift_and_code(self):
        p = 5
        xs = states(p)
        codes = [tuple((a % 2, b // 2) for a, b in xs),
                 tuple(a for a, b in xs),
                 tuple(task(a, b, p) // 2 for a, b in xs)]
        weights = tuple((a + 1) * (b + 2) for a, b in xs)
        for code in codes:
            base = onehot_risk(code, [task(*x, p) for x in xs], weights)
            for u in range(p):
                for v in range(p):
                    target = [task(*shift(*x, u, v, p), p) for x in xs]
                    self.assertEqual(onehot_risk(code, target, weights), base)
        self.assertEqual(onehot_risk(tuple(task(*x, p) for x in xs),
                                         tuple(task(*x, p) for x in xs)), Q(0))

    def test_alias_and_noninjective_projection(self):
        p = 5
        xs = states(p)
        labels = [task(*x, p) for x in xs]
        self.assertEqual(onehot_risk([0] * len(xs), labels), Q(4, 5))
        self.assertEqual(onehot_risk(labels, labels), 0)
        with self.assertRaises(ValueError):
            onehot_risk([0], [0], [0])


if __name__ == '__main__':
    unittest.main()
