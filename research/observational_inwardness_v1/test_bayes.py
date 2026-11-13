"""Uniform-prior exact Bayesian identification curve tests."""
import unittest
from fractions import Fraction

from bayes import bayes_error, curve, optimal_fixed_schedule, predicted_error, total_variation


class BayesCurveTests(unittest.TestCase):
    def test_first_probe_is_uninformative(self):
        for n in range(2, 20):
            self.assertEqual(total_variation(n, (1,)), 0)
            self.assertEqual(bayes_error(n, (1,)), Fraction(1, 2))

    def test_optimal_curve_and_fixed_schedule_search(self):
        for n in range(3, 9):
            for t in range(n + 1):
                p, word = optimal_fixed_schedule(n, t)
                self.assertEqual(p, predicted_error(n, t))
                self.assertEqual(bayes_error(n, (1,) * t), p)
                if t >= 2:
                    self.assertEqual(total_variation(n, (1,) * t), Fraction(t, n))
                    self.assertEqual(word, (1,) * t)

    def test_exact_curve(self):
        for n in range(2, 10):
            records = curve(n)
            self.assertEqual(records[0]["bayes_error"], "1/2")
            self.assertEqual(records[1]["bayes_error"], "1/2")
            self.assertEqual(records[-1]["bayes_error"], "0")

    def test_validation(self):
        with self.assertRaises(ValueError):
            predicted_error(1, 1)
        with self.assertRaises(ValueError):
            predicted_error(3, -1)
        with self.assertRaises(ValueError):
            optimal_fixed_schedule(3, -1)


if __name__ == "__main__":
    unittest.main()
