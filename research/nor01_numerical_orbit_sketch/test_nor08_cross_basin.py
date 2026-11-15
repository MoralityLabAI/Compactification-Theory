"""NOR-08D source-bound rational basin separation checks."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor08_cross_basin as m


class BasinCodeGapTests(unittest.TestCase):
    def test_exact_rational_square_lower(self):
        self.assertEqual(m.lower_sqrt_rational(Q(4)),Q(2))
        self.assertLess(m.lower_sqrt_rational(Q(2)),Q(3,2))
        self.assertLessEqual(m.lower_sqrt_rational(Q(2))**2,Q(2))

    def test_source_locked_two_basin_margin(self):
        r=m.audit()
        self.assertEqual(r["source_checkpoint_sha256"],m.SHA)
        self.assertGreater(Q(r["first_second_ball_separation_lower_fraction"]),Q(5))
        self.assertEqual(r["minimum_distinct_terminal_code_labels"],2)
        self.assertEqual(r["minimum_fixed_width_terminal_label_bits"],1)
        self.assertTrue(r["signed_natural_source_capture"])
        self.assertFalse(r["squared_natural_source_capture_certified_by_this_audit"])

    def test_invalid_square(self):
        with self.assertRaises(ValueError):
            m.lower_sqrt_rational(Q(-1))
        with self.assertRaises(ValueError):
            m.lower_sqrt_rational(Q(1),scale=0)


if __name__=="__main__":
    unittest.main()
