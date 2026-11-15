"""NOR-09: source-locked dual captured GRU orbit clipping controls."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor09_dual_history as gym


class DualCapturedHistories(unittest.TestCase):
    def test_both_natural_source_captures_and_exact_ball_clearance(self):
        c=gym.source_constants()
        self.assertEqual(c["capture_1"],1308)
        self.assertEqual(c["capture_2"],660)
        self.assertEqual(c["first_ball_radius"],Q(1,10**11))
        self.assertEqual(c["second_ball_radius"],Q(1,10**6))
        self.assertGreater(c["gap"],5)
        self.assertGreater(c["first_source_amplitude"],0)
        self.assertGreater(c["second_source_amplitude"],0)
        self.assertLess(c["first_K24"],1)
        self.assertLess(c["second_K24"],1)

    def test_monotone_certified_tail_and_correct_residual_floor(self):
        c=gym.source_constants()
        previous=None
        floor=2*max(c["first_fixed_offset"],c["second_fixed_offset"])
        for k in (0,1,16,32,64,96,128,256):
            v=gym.horizon_record(k,c)
            bound=Q(v["two_anchor_clipped_tail_pairwise_error_upper_fraction"])
            self.assertGreaterEqual(bound,floor)
            if previous is not None:
                self.assertLessEqual(bound,previous)
            previous=bound
        self.assertLess(previous,Q(4,10**14))
        self.assertGreater(previous,Q(3,10**14))

    def test_two_codes_retain_large_certified_margin(self):
        a=gym.run((0,32,64,128))
        self.assertEqual(a["required_terminal_label_count"],2)
        self.assertEqual(a["required_fixed_width_label_bits"],1)
        self.assertEqual(a["sampled_clock_stride"],24)
        self.assertTrue(a["zero_input_real_analytic_map"])
        self.assertFalse(a["source_truth_or_private_future_used"])
        by_k={row["extra_24_step_blocks"]:row for row in a["rows"]}
        self.assertEqual(by_k[64]["signed_original_zero_input_steps"],2844)
        self.assertEqual(by_k[64]["squared_original_zero_input_steps"],2196)
        self.assertLess(Q(by_k[64]["two_anchor_clipped_tail_pairwise_error_upper_fraction"]),Q(12,10**12))
        self.assertGreater(Q(by_k[64]["one_code_cross_basin_additive_error_lower_fraction"]),Q(5))

    def test_rational_rounding_outward_and_bad_horizons(self):
        self.assertEqual(gym.upward(Q(1,3),10),Q(4,10))
        self.assertEqual(gym.upward(Q(1,2),10),Q(1,2))
        with self.assertRaises(ValueError):
            gym.upward(Q(-1,2))
        with self.assertRaises(ValueError):
            gym.horizon_record(-1,gym.source_constants())
        with self.assertRaises(ValueError):
            gym.run((4,4))


if __name__=="__main__":
    unittest.main()
