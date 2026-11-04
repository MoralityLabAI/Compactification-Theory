import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.intrinsic_entropy import (
    latent_grid_partition, bit_capacity_upper, rate_radius,
    linear_rates, finite_lipschitz_check, exact_packing_lower_bound,
    packing_bit_lower_bound, metric_entropy_certificate, grid_axis_count,
)
from research.ingression_dynamics_v1.geometric_quotient import (
    minimum_cover, minimum_label_bits,
)
from research.ingression_dynamics_v1.multi_action_coloring import maximum_rate_radius


class IntrinsicEntropyTests(unittest.TestCase):
    def test_12_action_one_latent_coordinate(self):
        u = tuple((Q(i,40),) for i in range(41))
        rows = tuple((Q(j,12),) for j in range(1,13))
        result = metric_entropy_certificate(u,rows,Q(1,8))
        self.assertEqual(result["intrinsic_dim"],1)
        self.assertEqual(result["actions"],12)
        self.assertEqual(result["global_lipschitz_L"],"1")
        self.assertEqual(result["grid_cells_per_axis"],4)
        self.assertEqual(result["used_transient_classes"],4)
        self.assertEqual(result["macro_label_bits_upper"],3)
        self.assertLessEqual(Q(result["observed_one_step_TV_defect"]),Q(1,8))
        p,L=linear_rates(u,rows)
        self.assertTrue(finite_lipschitz_check(u,p,L))
        self.assertEqual(
            rate_radius(p,result["labels"]),
            maximum_rate_radius({f"a_{j}":tuple(pt[j] for pt in p)
                                 for j in range(12)},result["labels"]))

    def test_two_dimensional_grid_matches_upper_and_packing_lower(self):
        u=tuple((Q(i,2),Q(j,2)) for i,j in product(range(3),repeat=2))
        rows=((Q(1),Q(0)),(Q(0),Q(1)))
        pts,L=linear_rates(u,rows)
        tau=Q(1,5)
        result=metric_entropy_certificate(u,rows,tau)
        self.assertEqual(result["grid_cells_per_axis"],3)
        self.assertEqual(result["used_transient_classes"],9)
        self.assertEqual(result["macro_label_bits_upper"],4)
        self.assertEqual(len(exact_packing_lower_bound(pts,tau)),9)
        self.assertEqual(packing_bit_lower_bound(pts,tau),4)
        self.assertEqual(len(minimum_cover(pts,2*tau)),9)
        self.assertEqual(minimum_label_bits(pts,2*tau),4)

    def test_mixed_affine_map_and_grid_bound(self):
        u=tuple((Q(i,4),Q(j,4)) for i,j in product(range(5),repeat=2))
        rows=((Q(1,2),Q(1,2)),(Q(1,3),Q(1,3)),
              (Q(1,4),Q(1,2)),(Q(0),Q(1,3)))
        out,L=linear_rates(u,rows)
        self.assertEqual(L,Q(1))
        labels,cells,_=latent_grid_partition(u,L,Q(1,6))
        self.assertEqual(cells,3)
        self.assertLessEqual(len(set(labels)),9)
        self.assertLessEqual(rate_radius(out,labels),Q(1,6))
        self.assertTrue(finite_lipschitz_check(u,out,L))

    def test_zero_lipschitz_is_constant_rate(self):
        u=((Q(0),),(Q(1,2),),(Q(1),))
        rate,L=linear_rates(u,((Q(0),),(Q(0),)),(Q(1,3),Q(2,3)))
        self.assertEqual(L,0)
        result=metric_entropy_certificate(u,((Q(0),),(Q(0),)),Q(1,10),
                                           (Q(1,3),Q(2,3)))
        self.assertEqual(result["used_transient_classes"],1)
        self.assertEqual(result["macro_label_bits_upper"],1)
        self.assertEqual(result["observed_one_step_TV_defect"],"0")
        self.assertEqual(grid_axis_count(Q(0),Q(1,10)),1)

    def test_strict_packing_threshold(self):
        points=((Q(0),Q(0)),(Q(1,2),Q(0)))
        self.assertEqual(len(exact_packing_lower_bound(points,Q(1,4))),1)
        self.assertEqual(len(exact_packing_lower_bound(points,Q(1,5))),2)

    def test_monotonic_label_capacity_in_tolerance(self):
        values=[bit_capacity_upper(2,Q(1),Q(1,d))
                for d in (2,3,4,5,8,16)]
        self.assertEqual(values,sorted(values))

    def test_out_of_range_affine_map_rejected_globally(self):
        # Even though the one supplied sample at u=0 lies in [0,1],
        # an affine model leaving the probability cube must be refused.
        with self.assertRaises(ValueError):
            linear_rates(((Q(0),),),((Q(1),),),(Q(4,5),))

    def test_invalid_contract(self):
        with self.assertRaises(ValueError):
            bit_capacity_upper(0,Q(1),Q(1,4))
        with self.assertRaises(ValueError):
            grid_axis_count(-1,Q(1))
        with self.assertRaises(ValueError):
            grid_axis_count(Q(1),Q(0))
        with self.assertRaises(ValueError):
            latent_grid_partition(((Q(-1),),),Q(1),Q(1,2))
        with self.assertRaises(ValueError):
            linear_rates(((Q(0),),(Q(0),Q(0))),((Q(1),),))
        with self.assertRaises(ValueError):
            exact_packing_lower_bound(((Q(0),),)*17,Q(1,4))


if __name__ == "__main__":
    unittest.main()
