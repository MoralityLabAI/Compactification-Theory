import unittest

from research.shaft_compactification.depth_quotient import (
    exact_quotient, observed_signature, report, scalar_deviation,
)


class DepthQuotientTests(unittest.TestCase):
    def test_horizon_refinement_exact(self):
        self.assertEqual([exact_quotient(h, (0,))['behavior_classes'] for h in range(5)], [1, 2, 4, 8, 16])

    def test_joint_substrates_refine_more_quickly(self):
        self.assertEqual([exact_quotient(h, (0, 2))['behavior_classes'] for h in range(5)], [1, 4, 16, 16, 16])

    def test_code_length_is_log_behavior_classes(self):
        self.assertEqual([exact_quotient(h, (0,))['minimal_fixed_length_code_bits'] for h in range(5)], [0, 1, 2, 3, 4])

    def test_substrate_coverage_only_refines_classes(self):
        for h in range(12):
            self.assertGreaterEqual(exact_quotient(h, (0, 2))['behavior_classes'],
                                    exact_quotient(h, (0,))['behavior_classes'])

    def test_complete_goal_erasure_collapses_signatures(self):
        for h in range(7):
            self.assertEqual(len({observed_signature(0, h, (0, 2)) for _ in range(16)}), 1)

    def test_scalar_depth_bound_is_sharp(self):
        self.assertEqual(scalar_deviation(0, 1, 5), 1)
        self.assertEqual(scalar_deviation(1, 1, 5), 5)
        self.assertEqual(scalar_deviation(2, 1, 5), 31)
        self.assertAlmostEqual(scalar_deviation(0.5, 1, 50), 2, places=12)

    def test_contractivity_vs_expansion(self):
        self.assertLess(scalar_deviation(0.8, 0.01, 100), 0.051)
        self.assertGreater(scalar_deviation(1.2, 0.01, 100), 1000)

    def test_receipt_schema(self):
        r = report()
        self.assertEqual(r['goals'], 16)
        self.assertEqual(r['joint_substrates_offsets_0_2'][2]['behavior_classes'], 16)


if __name__ == '__main__':
    unittest.main()
