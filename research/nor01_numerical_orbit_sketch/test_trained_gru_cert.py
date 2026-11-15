"""NOR-03: exact-interval mathematical regression gates on frozen RRC25.

These tests use the real source-locked checkpoint. The principal
positive witness is a certified 12-step Jacobian product after eight
zero-input GRU steps from hidden zero, not a synthetic Jordan matrix.
"""
import math
import unittest
from fractions import Fraction as Q

from trained_gru_cert import (
    Interval, exponential_bounds, sigmoid_at, tanh,
    sqrt_upper, round_rational, ceil_rational,
    hessian_bound, interval_forward, get_weights,
    build_certificate, run,
)
from research.rrc_compactification import bench25


class CertifiedGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck, digest = bench25.checkpoints()
        cls.ck = ck["recurrent"]
        cls.digest = digest["recurrent"]
        cls.weights = get_weights(cls.ck)

    def test_exact_taylor_exponential_enclosures(self):
        for x in (Q(-15, 2), Q(-3), Q(-1, 10), Q(0), Q(1, 7), Q(5), Q(19, 2)):
            v = exponential_bounds(x)
            truth = math.exp(float(x))
            self.assertLessEqual(float(v.lo) - 1e-13, truth)
            self.assertLessEqual(truth, float(v.hi) + 1e-13)
            self.assertGreater(v.lo, 0)
            self.assertGreaterEqual(v.hi, v.lo)

    def test_high_precision_adaptive_exponential_tail(self):
        # At 16 decimal digits the fixed 72-term Taylor tail could
        # dominate long interval recurrences. The new 32-digit engine
        # proves tight absolute enclosures even at large arguments.
        from trained_gru_cert import DIGITS, EXP_TAIL_TARGET
        self.assertGreaterEqual(DIGITS, 32)
        self.assertLess(EXP_TAIL_TARGET, Q(1, 10**40))
        for x in (Q(0), Q(7), Q(12), Q(18), Q(25)):
            bound = exponential_bounds(x)
            self.assertLessEqual(bound.hi - bound.lo, Q(3, 10**DIGITS))
            self.assertGreater(bound.lo, 0)

    def test_sigmoid_tanh_monotonic_enclosures(self):
        for x in (Q(-10), Q(-4), Q(-1, 100), Q(0), Q(3, 10), Q(3), Q(11)):
            z = sigmoid_at(x)
            q = 1 / (1+math.exp(-float(x)))
            self.assertLessEqual(float(z.lo)-1e-14, q)
            self.assertGreaterEqual(float(z.hi)+1e-14, q)
            t = tanh(Interval.point(x))
            self.assertLessEqual(float(t.lo)-1e-14, math.tanh(float(x)))
            self.assertGreaterEqual(float(t.hi)+1e-14, math.tanh(float(x)))

    def test_outward_sqrt_and_rounding(self):
        for x in (Q(0), Q(1, 4), Q(2), Q(89, 10), Q(1, 10**22)):
            y = sqrt_upper(x)
            self.assertGreaterEqual(y*y, x)
            self.assertLess(y, sqrt_upper(x, 10**9) + Q(1, 10**9))
        self.assertEqual(round_rational(Q(7, 2), 1), Q(4))
        self.assertEqual(ceil_rational(Q(7, 2), 1), Q(4))

    def test_trained_source_lock_and_hessian_budget(self):
        self.assertEqual(self.ck["parameters"], 1000)
        self.assertEqual(len(self.digest), 64)
        H = hessian_bound(self.weights)
        self.assertGreater(H, 0)
        self.assertTrue(math.isfinite(float(H)))

    def test_amplifying_early_and_contracting_late_trained_block(self):
        early = build_certificate(self.weights,
                                  [Interval.point(0)]*12, length=8,
                                  label="early_negative")
        box = interval_forward(self.weights, [[0]*8 for _ in range(8)])
        late = build_certificate(self.weights, box, length=12,
                                 label="late_positive")
        self.assertFalse(early["contracting_disk"])
        self.assertTrue(late["contracting_disk"])
        self.assertEqual(late["disk_center_fraction"], "1/32")
        self.assertGreater(late["block_nonlinear_remainder_upper"], 0)
        self.assertLess(late["anchor_box_max_radius"], 1e-10)

    def test_closed_checkpoint_replay_is_deterministic(self):
        one = run(include_heldout=False)
        two = run(include_heldout=False)
        self.assertEqual(one, two)
        self.assertTrue(one["zero_origin_positive"]["contracting_disk"])
        self.assertFalse(one["early_negative_control"]["contracting_disk"])


if __name__ == "__main__":
    unittest.main()
