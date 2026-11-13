"""SUD-01 finite reference checks; no dependencies beyond Python stdlib."""
import unittest
from reference import (
    binary_entropy, balanced_masks, exhaustive_balanced, parity_error,
    stats, truth_mask
)

class TestSUD01(unittest.TestCase):
    def test_entropy_endpoints(self):
        self.assertEqual(binary_entropy(0), 0)
        self.assertEqual(binary_entropy(1), 0)
        self.assertAlmostEqual(binary_entropy(.5), 1)

    def test_dictators_saturate(self):
        for n in [1, 2, 3]:
            mask = truth_mask(n, lambda x: x & 1)
            for p in [0, .03, .1, .25, .5]:
                a = stats(n, mask, p)
                self.assertTrue(a["balanced"])
                self.assertAlmostEqual(a["conditional_entropy_bits"],
                                       binary_entropy(p), places=11)
                self.assertAlmostEqual(a["bayes_error"], p, places=11)
                self.assertAlmostEqual(a["posterior_dimension"],
                                       a["CK_dimension_floor"], places=11)

    def test_parity_closed_form(self):
        for n in range(1, 5):
            mask = truth_mask(n, lambda x: x.bit_count() % 2)
            for p in [.05, .1, .25, .4]:
                a = stats(n, mask, p)
                pn = parity_error(n, p)
                self.assertAlmostEqual(a["conditional_entropy_bits"],
                                       binary_entropy(pn), places=10)
                self.assertAlmostEqual(a["bayes_error"], pn, places=10)

    def test_exhaustive_small_balanced_families(self):
        for n, count in [(1, 2), (2, 6), (3, 70)]:
            for p in [.01, .1, .3]:
                a = exhaustive_balanced(n, p)
                self.assertEqual(a["balanced_functions_checked"], count)
                self.assertAlmostEqual(a["smallest_entropy"], binary_entropy(p),
                                       places=10)
                self.assertAlmostEqual(a["smallest_bayes_error"], p, places=10)

    def test_all_12870_balanced_four_bit_functions(self):
        a = exhaustive_balanced(4, .1)
        self.assertEqual(a["balanced_functions_checked"], 12870)
        self.assertAlmostEqual(a["smallest_entropy"], binary_entropy(.1),
                               places=10)
        self.assertAlmostEqual(a["smallest_bayes_error"], .1, places=10)

    def test_reconstruction_probability_ceiling(self):
        p, n, T = .1, 3, 12
        for mask in balanced_masks(n):
            a = stats(n, mask, p)
            self.assertLessEqual((1-a["bayes_error"])**T, (1-p)**T+1e-12)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            stats(2, 1, .1, .5)
        with self.assertRaises(ValueError):
            stats(2, 1, .6)
        with self.assertRaises(ValueError):
            binary_entropy(-.01)

if __name__ == "__main__":
    unittest.main()
