"""Exact UG-LAT-01 local quadratic block checks; no model weights or GPU."""
from fractions import Fraction
from itertools import product
import unittest

from unique_games_latent_v01 import (
    add, block_formula, dot, exact_block_change, hyperplane, mul,
    polar, projective_lines, quadratic, recursive_change_formula,
    scalar, update_space, vectors,
)


class QuadraticBlockTests(unittest.TestCase):
    def test_small_field_arithmetic_is_a_field(self):
        for d in (1, 2, 3):
            q = 1 << d
            for a, b, c in product(range(q), repeat=3):
                self.assertEqual(mul(a, b, d), mul(b, a, d))
                self.assertEqual(mul(mul(a, b, d), c, d),
                                 mul(a, mul(b, c, d), d))
                self.assertEqual(mul(a, b ^ c, d),
                                 mul(a, b, d) ^ mul(a, c, d))
            for a in range(1, q):
                self.assertTrue(any(mul(a, b, d) == 1 for b in range(1, q)))

    def test_projective_line_and_hyperplane_dimensions(self):
        for d in (1, 2, 3):
            q = 1 << d
            lines = projective_lines(d)
            self.assertEqual(len(lines), q * q + q + 1)
            for v in lines:
                self.assertEqual(len(hyperplane(v, d)), q * q)
                self.assertEqual(len(update_space(v, d)), q ** 3)

    def test_quadratic_polar_annihilator_and_line_additivity(self):
        for d in (1, 2):
            q = 1 << d
            for v in projective_lines(d):
                for t in range(q):
                    a = scalar(t, v, d)
                    for x in vectors(d):
                        self.assertEqual(dot(v, polar(x, a, d), d), 0)
                    for s in range(q):
                        b = scalar(s, v, d)
                        self.assertEqual(quadratic(add(a, b), d),
                                         add(quadratic(a, d), quadratic(b, d)))

    def test_update_space_is_binary_linear(self):
        for d in (1, 2):
            zero = (0, 0, 0)
            for v in projective_lines(d):
                U = set(update_space(v, d))
                self.assertIn((zero, zero), U)
                for a, b in U:
                    for c, f in U:
                        self.assertIn((add(a, c), add(b, f)), U)

    def test_exact_nonzero_update_probability(self):
        for d in (1, 2, 3):
            lines = projective_lines(d)
            for v in lines[:min(5, len(lines))]:
                self.assertEqual(exact_block_change(v, d),
                                 block_formula(d))
                self.assertEqual(exact_block_change(v, d, (1, 0, 1)),
                                 block_formula(d))
        self.assertEqual(block_formula(1), Fraction(6, 7))
        self.assertEqual(recursive_change_formula(1, 0), Fraction(7, 8))
        self.assertEqual(recursive_change_formula(1, 1), Fraction(3, 4))
        self.assertGreater(recursive_change_formula(2, 2),
                           recursive_change_formula(2, 3))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            mul(4, 0, 2)
        with self.assertRaises(ValueError):
            update_space((0, 0, 0), 2)
        with self.assertRaises(ValueError):
            recursive_change_formula(2, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
