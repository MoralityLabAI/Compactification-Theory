"""NOR-04 rigorous nonlinear invariant-ball controls for trained RRC25."""
import unittest
from fractions import Fraction as Q

from invariant_ball import (
    CENTER, CENTER_DECIMALS, RADIUS, DEPTH, centered_ball,
    fixed_ball_certificate,
)
from trained_gru_cert import (
    Interval, build_certificate, get_weights, gates, sqrt_upper, ZERO,
)
from research.rrc_compactification import bench25


class NonlinearFixedBallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck, dig = bench25.checkpoints()
        cls.weights = get_weights(ck["recurrent"])
        cls.dig = dig["recurrent"]

    def test_source_center_and_cube(self):
        self.assertEqual(len(CENTER), 12)
        self.assertEqual(len(CENTER_DECIMALS), 12)
        self.assertEqual(RADIUS, Q(1, 10**11))
        self.assertEqual(DEPTH, 12)
        box = centered_ball(CENTER, RADIUS)
        self.assertEqual(len(box), 12)
        for x, c in zip(box, CENTER, strict=True):
            self.assertLessEqual(x.lo, c)
            self.assertGreaterEqual(x.hi, c)
            self.assertGreater(x.lo, -1)
            self.assertLess(x.hi, 1)

    def test_invalid_ball_is_rejected(self):
        with self.assertRaises(ValueError):
            centered_ball(CENTER, Q(0))
        with self.assertRaises(ValueError):
            centered_ball([Q(0)], Q(1, 10**11))
        with self.assertRaises(ValueError):
            centered_ball([Q(1)]*12, Q(1, 10**11))

    def test_numeric_anchor_enclosed_by_rational_fixed_block(self):
        state = [Interval.point(x) for x in CENTER]
        for _ in range(DEPTH):
            state = gates(self.weights, state)[0]
        residuals = [max(abs(x.lo-c), abs(x.hi-c))
                     for x,c in zip(state, CENTER, strict=True)]
        s = sqrt_upper(sum((v*v for v in residuals), ZERO), 10**24)
        self.assertGreater(s, 0)
        self.assertLess(s, Q(1, 10**11))

    def test_uniform_block_jacobian_contraction_and_self_map(self):
        cert = fixed_ball_certificate()
        self.assertEqual(cert["checkpoint_sha256"], self.dig)
        self.assertEqual(cert["checkpoint_parameters"], 1000)
        self.assertLess(cert["operator_upper_float"], 1.0)
        self.assertTrue(cert["closed_ball_invariant"])
        self.assertTrue(cert["banach_fixed_point_certificate"])
        self.assertGreater(Q(cert["invariance_margin_fraction"]), 0)
        self.assertLess(Q(cert["fixed_point_distance_from_center_upper_fraction"]),
                        RADIUS)
        self.assertEqual(cert["fixed_point_of_one_step_F_proved"], False)
        self.assertEqual(cert["trained_source_family_in_basin_proved"], False)


if __name__ == "__main__":
    unittest.main()
