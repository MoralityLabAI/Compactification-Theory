"""NOR-10: frozen one-bit terminal metric and source-receipt controls."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor10_binary_terminal as m


class OneBitCertifiedTerminal(unittest.TestCase):
    def test_center_distance_has_exact_rational_sqrt_bracket(self):
        x=m.center_gap_interval()
        lower=Q(x["lower_fraction"])
        upper=Q(x["upper_fraction"])
        sq=Q(x["distance_squared_fraction"])
        self.assertLessEqual(lower*lower,sq)
        self.assertGreaterEqual(upper*upper,sq)
        self.assertEqual(upper-lower,Q(1,10**18))
        self.assertGreater(lower,5)

    def test_two_distinct_rational_codes(self):
        q=m.center_gap_interval()
        self.assertEqual(m.encoded_terminal_code("signed",q),Q(0))
        self.assertGreater(m.encoded_terminal_code("squared",q),Q(5))
        with self.assertRaises(ValueError):
            m.encoded_terminal_code("unseen",q)

    def test_exact_two_captures_and_error_floor(self):
        x=m.run((0,32,64,96,128,256))
        self.assertEqual(x["source_capture_clocks"],[1308,660])
        self.assertEqual(x["sampled_clock_stride"],24)
        self.assertEqual(x["terminal_runtime_basin_label_bits"],1)
        self.assertTrue(x["shared_rational_distance_metadata_stored_once"])
        self.assertFalse(x["claim_boundaries"]["all_source_basin_labels_inferred_by_model"])
        last=None
        for row in x["rows"]:
            a=Q(row["signed_center_radius_upper_fraction"])
            b=Q(row["squared_center_radius_upper_fraction"])
            c=Q(row["cross_basin_pair_error_upper_fraction"])
            combined=Q(row["one_bit_terminal_all_pair_error_upper_fraction"])
            self.assertEqual(c,a+b+Q(1,10**18))
            self.assertEqual(combined,max(2*a,2*b,c))
            self.assertGreater(Q(row["zero_bit_cross_basin_error_lower_fraction"]),5)
            if last is not None:
                self.assertLessEqual(combined,last)
            last=combined
        self.assertLess(last,Q(4,10**14))

    def test_code_capacity_ratio_large_at_cutoff64(self):
        x=m.run((64,))
        row=x["rows"][0]
        self.assertGreater(row["rigorous_zero_bit_to_one_bit_worst_error_ratio_lower_float"],1e11)
        self.assertLess(Q(row["one_bit_terminal_all_pair_error_upper_fraction"]),Q(12,10**12))

    def test_invalid_epochs(self):
        with self.assertRaises(ValueError):
            m.run((12,12))
        with self.assertRaises(ValueError):
            m.run((-1,))


if __name__=="__main__":
    unittest.main()
