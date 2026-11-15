"""NOR-08B source-locked squared-family rational capture smoke tests."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor08_squared_capture as sq


class SquaredSourceEntryTests(unittest.TestCase):
    def test_second_basin_frozen_exact_radius_and_checkpoint(self):
        center,R,K,a,b=sq.certified_target()
        self.assertEqual(len(center),12)
        self.assertEqual(R,Q(3,10**6))
        self.assertLess(K,1)
        self.assertGreater(K,0)
        self.assertEqual(len(a),len(b))
        self.assertEqual(len(a),64)

    def test_short_rational_probe_is_deterministic_and_inconclusive(self):
        r=sq.run(max_blocks=2,record_stride=1)
        t=sq.run(max_blocks=2,record_stride=1)
        self.assertEqual(r,t)
        self.assertEqual(r["checkpoint_sha256"],sq.SOURCE_SHA)
        self.assertEqual(r["public_source"]["family"],"squared")
        self.assertEqual(r["target_second_ball_disjoint_from_first"],True)
        self.assertEqual(r["processed_blocks"],2)
        self.assertEqual(r["capture_status"],"not_established")
        self.assertIsNone(r["capture_witness"])

    def test_invalid_budget_is_rejected(self):
        for n in (0,513):
            with self.assertRaises(ValueError):
                sq.run(max_blocks=n)
        with self.assertRaises(ValueError):
            sq.run(max_blocks=2,record_stride=0)


if __name__=="__main__":
    unittest.main()
