"""Constructive finite checks for the uniform-prior information bound."""
import itertools
from math import log2
import unittest

from counting import conditional_entropy_bits, max_exact_recovery_probability, uniform_label_space


class CountingTests(unittest.TestCase):
    def test_six_binary_axes_have_six_bit_fiber(self):
        self.assertEqual(uniform_label_space([2] * 6), (64, 1 / 64, 6.0))

    def test_bruteforce_every_two_bucket_summary_of_four_states(self):
        for summary in itertools.product(range(2), repeat=4):
            with self.subTest(summary=summary):
                self.assertLessEqual(max_exact_recovery_probability(summary), 2 / 4)
                self.assertGreaterEqual(conditional_entropy_bits(summary), 1.0)

    def test_constant_vs_identity_summary(self):
        self.assertEqual(max_exact_recovery_probability([0, 0, 0, 0]), .25)
        self.assertEqual(conditional_entropy_bits([0, 0, 0, 0]), 2.0)
        self.assertEqual(max_exact_recovery_probability([0, 1, 2, 3]), 1.0)
        self.assertEqual(conditional_entropy_bits([0, 1, 2, 3]), 0.0)

    def test_partial_partition_entropy(self):
        self.assertEqual(conditional_entropy_bits([0, 0, 1, 1]), 1.0)
        self.assertEqual(max_exact_recovery_probability([0, 0, 1, 1]), .5)

    def test_reject_invalid_spaces(self):
        for fn, arg in ((max_exact_recovery_probability, []),
                        (conditional_entropy_bits, []),
                        (uniform_label_space, [2, 0])):
            with self.assertRaises(ValueError):
                fn(arg)


if __name__ == '__main__':
    unittest.main()
