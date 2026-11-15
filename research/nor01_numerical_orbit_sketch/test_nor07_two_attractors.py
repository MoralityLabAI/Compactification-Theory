"""NOR-07 exact trained-GRU two-attractor control tests."""
import unittest
from fractions import Fraction as Q

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor07_two_attractors as m
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER


class TwoAttractorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,dig=bench25.checkpoints()
        cls.ck=ck["recurrent"]
        cls.digest=dig["recurrent"]

    def test_fixed_checkpoint_and_seed_geometry(self):
        self.assertEqual(self.digest,m.EXPECTED_CK)
        self.assertEqual(self.ck["parameters"],1000)
        self.assertEqual(len(m.SECOND_SEED),12)
        self.assertEqual(m.BLOCK,24)
        self.assertGreater(abs(Q(m.SECOND_SEED[1])-CENTER[1]),Q(1))

    def test_decimal_newton_reproducible(self):
        c1,strings1,res1=m.decimal_newton_center(self.ck)
        c2,strings2,res2=m.decimal_newton_center(self.ck)
        self.assertEqual(c1,c2)
        self.assertEqual(strings1,strings2)
        self.assertEqual(res1,res2)
        self.assertEqual(len(c1),12)
        self.assertGreater(len(res1),0)
        self.assertGreater(abs(c1[1]-CENTER[1]),Q(1))

    def test_exact_source_locked_invariant_balls_disjoint(self):
        result=m.two_fixed_points_certificate()
        self.assertEqual(result["source_checkpoint_sha256"],self.digest)
        self.assertTrue(result["second_invariant_block"])
        self.assertTrue(result["second_period_one_certified"])
        self.assertTrue(result["disjoint_fixed_point_balls"])
        self.assertLess(Q(result["second_block_operator_norm_upper_fraction"]),1)
        self.assertGreater(Q(result["second_invariance_margin_fraction"]),0)
        self.assertGreater(Q(result["certified_coordinate_separation_lower_fraction"]),Q(1))
        self.assertFalse(result["squared_input_entry_to_second_basin_certified"])

    def test_survey_candidate_is_just_a_candidate(self):
        result=m.two_fixed_points_certificate()
        self.assertEqual(result["candidate_source"],
            "squared-family NOR-06 FP64 terminal hidden vector after 4800 zero-input steps")
        self.assertTrue(result["two_distinct_attracting_fixed_points"])
        self.assertFalse(result["squared_input_entry_to_second_basin_certified"])


if __name__=="__main__":
    unittest.main()
