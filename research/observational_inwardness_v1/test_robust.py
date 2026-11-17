"""Exact OIC-06 robustness and witness tests."""
import unittest

from robust import (
    ambiguous_receipt, corruption_cost, exhaustive_best_distance,
    guaranteed_distance, interfamily_ambiguity, interfamily_distance,
    minimum_robust_horizon, reference, signatures,
)


class RobustObservabilityTests(unittest.TestCase):
    def test_full_probing_minimum_distance_formula(self):
        for n in range(2, 12):
            for t in range(4 * n + 1):
                self.assertEqual(interfamily_distance(n, (1,) * t), t // n)
                self.assertEqual(guaranteed_distance(n, t), t // n)

    def test_exhaustive_best_schedule(self):
        for n in range(2, 5):
            for t in range(0, 9):
                d, word = exhaustive_best_distance(n, t)
                self.assertEqual(d, t // n)
                self.assertEqual(interfamily_distance(n, word), d)

    def test_exact_error_erasure_thresholds(self):
        for n in range(2, 7):
            for e in range(3):
                for r in range(3):
                    t = minimum_robust_horizon(n, e, r)
                    self.assertEqual(t, n * (2 * e + r + 1))
                    before = interfamily_ambiguity(n, (1,) * (t - 1), e, r)
                    after = interfamily_ambiguity(n, (1,) * t, e, r)
                    self.assertIsNotNone(before)
                    self.assertIsNone(after)
                    a, b, observed = before
                    self.assertLessEqual(corruption_cost(a, observed)[0], e)
                    self.assertLessEqual(corruption_cost(b, observed)[0], e)
                    self.assertLessEqual(corruption_cost(a, observed)[1], r)
                    self.assertEqual(corruption_cost(a, observed)[1], corruption_cost(b, observed)[1])

    def test_erasures_and_flip_constructive_witness(self):
        a, b = (0, 0, 0, 0), (1, 1, 1, 0)
        got = ambiguous_receipt(a, b, flips=1, erasures=1)
        self.assertIsNotNone(got)
        self.assertEqual(corruption_cost(a, got), (1, 1))
        self.assertEqual(corruption_cost(b, got), (1, 1))
        self.assertIsNone(ambiguous_receipt(a, b, flips=0, erasures=2))

    def test_no_probe_is_never_helpful(self):
        for n in range(2, 7):
            for t in range(0, 7):
                self.assertEqual(signatures(n, "fixed", (0,) * t), {tuple([0] * t)})
                self.assertEqual(interfamily_distance(n, (0,) * t), 0)

    def test_input_validation(self):
        for a in ((1, 0, 0), (4, -1, 0), (4, 0, -1), (3, 0.5, 1)):
            with self.assertRaises(ValueError):
                minimum_robust_horizon(*a)
        for n, typ, probe in ((2, "whatever", (1,)), (1, "fixed", (1,)), (2, "fixed", (2,))):
            with self.assertRaises(ValueError):
                signatures(n, typ, probe)

    def test_reference_is_internally_consistent(self):
        rows = reference()["rows"]
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(r["ambiguous_before"] and r["separable_at"] for r in rows))


if __name__ == "__main__":
    unittest.main()
