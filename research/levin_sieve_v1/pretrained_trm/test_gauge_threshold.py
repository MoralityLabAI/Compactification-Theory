"""Exact-rational optimum sufficient-radius algebra, independent of Arb."""
import unittest
from fractions import Fraction as Q
from test_gauge_quotient import mock_inputs
from gauge_threshold import exact_thresholds
from gauge_quotient import certificate_for_case

class TestExactRadius(unittest.TestCase):
    def test_strict_threshold_proves_below_and_can_fail_above(self):
        case,w=mock_inputs()
        proof=exact_thresholds(case,w)
        top1=Q(proof['winner_threshold_strict'])
        self.assertGreater(top1,0)
        lo=str(top1*Q(99,100));hi=str(top1*Q(101,100))
        c=certificate_for_case(case,w,(lo,hi))
        self.assertTrue(c['transverse_radii'][lo]['top1_certified'])
        self.assertFalse(c['transverse_radii'][hi]['top1_certified'])
        self.assertLessEqual(Q(proof['top_two_threshold_strict']),top1)

    def test_zero_head_sensitivity_radius_reaches_sigma(self):
        case,w=mock_inputs()
        w['head_weight_hex']=[['0x0.0p+0']*48 for _ in range(10)]
        a=exact_thresholds(case,w)
        self.assertEqual(Q(a['winner_threshold_strict']),Q(a['sigma_lower_bound']))
        self.assertEqual(Q(a['top_two_threshold_strict']),Q(a['sigma_lower_bound']))

    def test_unprovable_ordering_rejected(self):
        case,w=mock_inputs()
        w['head_bias_hex'][3]=(100.).hex()
        with self.assertRaises(ValueError):exact_thresholds(case,w)

if __name__=='__main__':unittest.main()
