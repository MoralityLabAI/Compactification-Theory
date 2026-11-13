import unittest

from research.shaft_compactification.behavioral_budget import (
    example_uniform_16_goal_bound, maximum_uniform_exact_recovery,
    necessary_exact_bits,
)


class BehavioralCodeBudgetTests(unittest.TestCase):
    def test_uniform_16_goal_ceiling(self):
        self.assertEqual([maximum_uniform_exact_recovery(16, k)[0] for k in range(5)],
                         [1, 2, 4, 8, 16])
        self.assertEqual(necessary_exact_bits(16), 4)

    def test_arbitrary_exact_minimum(self):
        for size in range(1, 129):
            bits = necessary_exact_bits(size)
            self.assertGreaterEqual(2 ** bits, size)
            if bits > 0:
                self.assertLess(2 ** (bits - 1), size)

    def test_uniform_bound_achievable_with_simple_code(self):
        for bits in range(5):
            mask = (1 << bits) - 1
            decoded = [(goal & mask) for goal in range(16)]
            successes = sum(dec == goal for goal, dec in enumerate(decoded))
            self.assertEqual(successes, maximum_uniform_exact_recovery(16, bits)[0])

    def test_receipt(self):
        self.assertEqual(example_uniform_16_goal_bound()['minimum_exact_bits'], 4)


if __name__ == '__main__':
    unittest.main()
