"""NOR-05: exact-rational period-one fixed point from NOR-04 ball."""
import unittest
from fractions import Fraction as Q

from invariant_ball import CENTER, RADIUS, fixed_ball_certificate
from period_one import period_one_certificate, point_residual
from trained_gru_cert import get_weights
from research.rrc_compactification import bench25


class PeriodOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck, digest=bench25.checkpoints()
        cls.weights=get_weights(ck["recurrent"])
        cls.digest=digest["recurrent"]

    def test_source_locked_nonlinear_block_prerequisite(self):
        prior=fixed_ball_certificate()
        self.assertEqual(prior["checkpoint_sha256"],self.digest)
        self.assertTrue(prior["banach_fixed_point_certificate"])
        self.assertLess(Q(prior["fixed_point_distance_from_center_upper_fraction"]), RADIUS)

    def test_one_step_center_residual_is_small(self):
        s=point_residual(self.weights,CENTER,1)
        self.assertGreater(s,0)
        self.assertLess(s,RADIUS / 100)

    def test_period_one_commutation_gate(self):
        r=period_one_certificate()
        self.assertEqual(r["checkpoint_sha256"],self.digest)
        self.assertTrue(r["one_step_fixed_point_image_distance_upper_fraction"])
        self.assertTrue(r["one_step_fixed_point_proved"])
        self.assertLess(Q(r["one_step_fixed_point_image_distance_upper_fraction"]),RADIUS)
        self.assertGreater(Q(r["global_one_step_cube_lipschitz_fraction"]),0)

    def test_scope_flags(self):
        r=period_one_certificate()
        self.assertFalse(r["heldout_input_basin_entry_proved"])
        self.assertFalse(r["native_proof_search_gain_proved"])
        self.assertEqual(r["schema"],"nor05.trained_gru.one_step_fixed_point.v1")


if __name__=="__main__":
    unittest.main()
