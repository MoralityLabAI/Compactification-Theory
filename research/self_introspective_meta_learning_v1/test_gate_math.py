"""Finite arithmetic and counterexample checks for the draft's proposed gates."""
from fractions import Fraction as F
import math
import unittest

from gate_math import attempt_alpha, endpoint_alpha, finite_class_radius, paired_interval, passes_declared_floors


class GateMathTests(unittest.TestCase):
    def test_attempt_spending_telescopes(self):
        alpha = F(1, 20)
        for n in (1, 2, 10, 1000):
            self.assertEqual(sum((attempt_alpha(alpha, t) for t in range(1, n + 1)), F()), alpha * n / (n + 1))

    def test_endpoint_allocation(self):
        for t in (1, 2, 99):
            for m in (1, 5, 30):
                self.assertEqual(m * endpoint_alpha(F(1, 20), t, m), attempt_alpha(F(1, 20), t))

    def test_reject_bad_allocation(self):
        for t in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                attempt_alpha(F(1, 20), t)
        for alpha in (F(0), F(1), F(-1), 0.05):
            with self.assertRaises(ValueError):
                attempt_alpha(alpha, 1)
        with self.assertRaises(ValueError):
            endpoint_alpha(F(1, 20), 1, 0)

    def test_paired_range_factor(self):
        n, alpha = 1000, 0.01
        lower, upper = paired_interval([0.0] * n, alpha)
        r = math.sqrt(2 * math.log(2 / alpha) / n)
        self.assertAlmostEqual(lower, -r)
        self.assertAlmostEqual(upper, r)
        self.assertAlmostEqual(r, 2 * math.sqrt(math.log(2 / alpha) / (2 * n)))

    def test_small_batch_is_inconclusive(self):
        interval = paired_interval([0.1] * 32, 0.005)
        self.assertLess(interval[0], 0)
        self.assertFalse(passes_declared_floors({'gain': interval}, {'gain': 0.02}))

    def test_large_gain_can_pass(self):
        interval = paired_interval([0.2] * 10000, 0.005)
        self.assertTrue(passes_declared_floors({'gain': interval}, {'gain': 0.1}))

    def test_reject_bad_scores(self):
        for xs in ([], [math.nan], [math.inf], [-1.1], [1.1]):
            with self.assertRaises(ValueError):
                paired_interval(xs, 0.05)
        for alpha in (0, 1, -0.1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                paired_interval([0.1], alpha)

    def test_clips_only_at_known_range(self):
        self.assertEqual(paired_interval([1.0], 0.05)[1], 1.0)
        self.assertEqual(paired_interval([-1.0], 0.05)[0], -1.0)

    def test_missing_endpoint_fails(self):
        with self.assertRaises(ValueError):
            passes_declared_floors({'gain': (0.1, 0.2)}, {'gain': 0.02, 'retention': -0.01})

    def test_aggregate_cannot_mask_regression(self):
        self.assertFalse(passes_declared_floors({'gain': (0.2, 0.3), 'anchor': (-0.1, -0.05)}, {'gain': 0.02, 'anchor': -0.01}))

    def test_anchor_prevents_incremental_forgetting(self):
        anchor, tolerance = F(9, 10), F(1, 100)
        scores = [anchor - k * F(1, 200) for k in range(4)]
        self.assertTrue(all(scores[i] - scores[i-1] >= -tolerance for i in range(1, 4)))
        self.assertLess(scores[-1] - anchor, -tolerance)

    def test_shear_joint_capacity(self):
        p = 31
        states = [(a, b) for a in range(p) for b in range(p)]
        present = {(a + b) % p for a, b in states}
        joint = {((a + b) % p, (2*a + b) % p) for a, b in states}
        self.assertEqual((len(present), len(joint)), (31, 961))
        self.assertEqual(((len(present)-1).bit_length(), (len(joint)-1).bit_length()), (5, 10))
        for s in range(p):
            targets = [(2*a + b) % p for a, b in states if (a+b) % p == s]
            self.assertEqual(sorted(targets), list(range(p)))
        self.assertEqual(1 - sum((F(1, p)**2 for _ in range(p)), F()), F(30, 31))

    def test_restoration_tube(self):
        rho, eta, radius = F(3, 4), F(1, 50), F(2, 25)
        self.assertLessEqual(eta, (1-rho)*radius)
        e = radius
        for _ in range(100):
            e = rho*e + eta
            self.assertLessEqual(e, radius)

    def test_description_penalty(self):
        r0 = finite_class_radius(0, 1, 1000, 0.05)
        self.assertGreater(finite_class_radius(10, 1, 1000, 0.05), r0)
        self.assertGreater(finite_class_radius(0, 10, 1000, 0.05), r0)
        self.assertAlmostEqual(finite_class_radius(0, 1, 4000, 0.05), r0 / 2)
        with self.assertRaises(ValueError):
            finite_class_radius(-1, 1, 1000, 0.05)

    def test_frozen_potential_telescopes(self):
        utilities = [F(2, 5), F(43, 100), F(1, 2), F(53, 100)]
        gains = [utilities[i] - utilities[i-1] for i in range(1, len(utilities))]
        self.assertEqual(sum(gains), utilities[-1] - utilities[0])
        self.assertTrue(all(g >= F(3, 100) for g in gains))


if __name__ == '__main__':
    unittest.main()
