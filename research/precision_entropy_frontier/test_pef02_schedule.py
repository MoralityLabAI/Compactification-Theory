"""PEF-02S exhaustive order-sensitive mixed contraction schedule results."""
from fractions import Fraction
from itertools import product, permutations
import unittest

from pef02_schedule import (
    Schedule, audit, full_history_precision_threshold,
    independent_exhaustive_gap, mixed_schedule_gap,
    per_step_error_gain, suite, visible_suffix_bits,
)


class SchedulePrecisionTests(unittest.TestCase):
    def test_all_mixed_schedules_mixed_precision_and_gap(self):
        for depth in range(1,6):
            for widths in product((1,2,3),repeat=depth):
                K=sum(widths)
                for bits in range(1,K+2):
                    actual=audit(Schedule(widths,bits))
                    self.assertEqual(
                        actual["exact_cross_suffix_gap"],
                        str(mixed_schedule_gap(widths,visible_suffix_bits(widths,bits))))
                    self.assertEqual(actual["distinct_codes"],
                                     1 << actual["retained_most_recent_inputs"])
                    self.assertTrue(actual["stagewise_integer_floor_commutes_with_endpoint_quantization"])

    def test_exact_capacity_schedule_permutations(self):
        a,b,c=(audit(Schedule(widths,6)) for widths in (
            (2,2,4),(2,4,2),(4,2,2)))
        self.assertEqual(a["total_contraction_exponent"],8)
        self.assertEqual(b["total_contraction_exponent"],8)
        self.assertEqual(c["total_contraction_exponent"],8)
        self.assertEqual((a["retained_most_recent_inputs"],
                          b["retained_most_recent_inputs"],
                          c["retained_most_recent_inputs"]), (2,2,3))
        self.assertEqual((a["exact_cross_suffix_gap"],b["exact_cross_suffix_gap"]),
                         ("9/256","57/256"))
        self.assertEqual((a["sharp_full_history_precision"],
                          b["sharp_full_history_precision"],
                          c["sharp_full_history_precision"]), (7,7,5))
        self.assertEqual(Fraction(57,9),Fraction(19,3))

    def test_threshold_switch_near_first_old_bit(self):
        for widths in product((1,2,3),repeat=4):
            P=full_history_precision_threshold(widths)
            self.assertEqual(visible_suffix_bits(widths,P),len(widths))
            self.assertEqual(visible_suffix_bits(widths,P-1),len(widths)-1)

    def test_per_step_noise_gain_order_sensitive(self):
        self.assertEqual(per_step_error_gain((2,2,4)),
                         Fraction(1)+Fraction(1,16)+Fraction(1,64))
        self.assertEqual(per_step_error_gain((2,4,2)),
                         Fraction(1)+Fraction(1,4)+Fraction(1,64))
        self.assertEqual(suite()["gap_ratio_b_over_a"],"19/3")

    def test_empty_readout_zero_bits(self):
        a=audit(Schedule((1,2,3),0))
        self.assertEqual(a["retained_most_recent_inputs"],0)
        self.assertEqual(a["distinct_codes"],1)
        self.assertIsNone(a["exact_cross_suffix_gap"])

    def test_guards(self):
        for s in (Schedule((),3),Schedule((0,2),2),
                  Schedule((2,-1),2),Schedule((2,2),-1)):
            with self.assertRaises(ValueError):
                audit(s)


if __name__=="__main__":
    unittest.main()
