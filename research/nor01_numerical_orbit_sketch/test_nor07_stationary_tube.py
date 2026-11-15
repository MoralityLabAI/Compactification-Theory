"""NOR-07S: quick independent stationary-tube mathematical regression tests."""
import unittest
from fractions import Fraction as Q

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor07_two_attractors as two
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER,RADIUS
from nor07_stationary_tube import stationary_tube,midpoint_matrix_and_error


class StationaryCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,dig=bench25.checkpoints()
        cls.weights=two.get_weights(ck["recurrent"])
        cls.ck_sha=dig["recurrent"]

    def test_exact_checkpoint_and_point_jacobian(self):
        self.assertEqual(self.ck_sha,two.EXPECTED_CK)
        matrix,error=midpoint_matrix_and_error(self.weights,CENTER)
        self.assertEqual(len(matrix),12)
        self.assertEqual(len(matrix[0]),12)
        self.assertGreater(error,0)

    def test_known_first_attractor_independently_certifiable(self):
        proof=stationary_tube(self.weights,CENTER,RADIUS,12)
        self.assertEqual(proof["block_length"],12)
        self.assertTrue(proof["contracting_invariant_ball"])
        self.assertLess(Q(proof["block_operator_norm_upper_fraction"]),1)
        self.assertGreater(Q(proof["invariance_margin_fraction"]),0)
        self.assertLess(Q(proof["center_residual_upper_fraction"]),RADIUS)

    def test_non_normal_one_step_and_twelve_step_behavior(self):
        early=stationary_tube(self.weights,CENTER,RADIUS,1)
        late=stationary_tube(self.weights,CENTER,RADIUS,12)
        self.assertGreater(Q(early["nominal_product_frobenius_upper_fraction"]),1)
        self.assertLess(Q(late["nominal_product_frobenius_upper_fraction"]),1)

    def test_invalid_neighborhood_rejected(self):
        with self.assertRaises(ValueError):
            stationary_tube(self.weights,CENTER,Q(-1,10),4)
        with self.assertRaises(ValueError):
            stationary_tube(self.weights,[Q(0)],RADIUS,4)
        with self.assertRaises(ValueError):
            stationary_tube(self.weights,CENTER,Q(1,10),0)


if __name__=="__main__":
    unittest.main()
