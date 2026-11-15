"""NOR-07T: independent controls for recentered Hessian/Volterra bound."""
import unittest
from fractions import Fraction as Q

from research.rrc_compactification import bench25
from research.nor01_numerical_orbit_sketch import nor07_two_attractors as proof
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER, RADIUS
from nor07_recentered_tube import (
    identity, matrix_product, frobenius_upper,
    nominal_trace, segment_norms, recentered_tube,
)


class RigorousNominalTubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint, dig = bench25.checkpoints()
        cls.w = proof.get_weights(checkpoint["recurrent"])
        cls.checkpoint_sha = dig["recurrent"]

    def test_rational_order_of_matrix_products(self):
        left = identity()
        left[0][0] = Q(2)
        left[0][1] = Q(3)
        right = identity()
        right[0][0] = Q(5)
        right[1][0] = Q(7)
        prod = matrix_product(left,right)
        self.assertEqual(prod[0][0],Q(31))
        self.assertEqual(prod[0][1],Q(3))
        self.assertEqual(prod[1][0],Q(7))

    def test_nominal_jacobian_products_retain_nonnormal_structure(self):
        states,matrices,eps,residual=nominal_trace(self.w,CENTER,2)
        self.assertEqual(len(states),3)
        self.assertEqual(len(matrices),2)
        self.assertEqual(len(eps),2)
        self.assertTrue(all(z>0 for z in eps))
        products,norms=segment_norms(matrices)
        self.assertEqual(products[(2,0)],matrix_product(matrices[1],matrices[0]))
        self.assertEqual(norms[(0,0)],Q(1))
        self.assertEqual(norms[(2,2)],Q(1))
        self.assertGreaterEqual(norms[(2,0)],0)

    def test_small_reference_tube_and_taylor_error_are_sound(self):
        result=recentered_tube(self.w,CENTER,RADIUS,2)
        self.assertEqual(result["block_length"],2)
        self.assertEqual(result["initial_radius_fraction"],str(RADIUS))
        self.assertGreater(Q(result["block_operator_norm_upper_fraction"]),0)
        self.assertGreater(Q(result["center_residual_upper_fraction"]),0)
        self.assertGreater(Q(result["max_point_jacobian_error_fraction"]),0)
        self.assertGreaterEqual(Q(result["block_derivative_variation_upper_fraction"]),0)
        self.assertGreaterEqual(Q(result["max_tube_radius_upper_fraction"]),RADIUS)

    def test_reject_invalid_radius_and_dimensions(self):
        with self.assertRaises(ValueError):
            recentered_tube(self.w,CENTER,Q(-1,100),2)
        with self.assertRaises(ValueError):
            recentered_tube(self.w,CENTER,Q(1,100),0)
        with self.assertRaises(ValueError):
            recentered_tube(self.w,[Q(0)],Q(1,100),2)


if __name__=="__main__":
    unittest.main()
