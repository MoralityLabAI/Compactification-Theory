"""FHC-01 exact finite controls; no external Python dependencies."""
import itertools
import random
import unittest
from fractions import Fraction

from reference import (
    control_report,
    coarse_embedding_1d_certificate,
    doubling_cover_bound,
    equilateral_packing_obstruction,
    exact_l2_to_l4_certificate,
    exact_line_equilateral_distortion,
    first_forbidden_equilateral_size,
    fourth_gaussian_moment,
    gaussian_isometry_moment_identity,
    net_transfer_distortion,
    robust_net_threshold,
)


class TestFHC01(unittest.TestCase):
    def test_wick_identity_for_rational_vectors(self):
        values = [
            [], [0], [Fraction(1, 3)], [1, 2], [-2, 3, 4],
            [Fraction(-7, 11), Fraction(4, 9), Fraction(3, 5)],
        ]
        for vector in values:
            self.assertTrue(gaussian_isometry_moment_identity(vector))
            expected = 3 * sum((Fraction(x)**2 for x in vector), Fraction())**2
            self.assertEqual(fourth_gaussian_moment(vector), expected)

    def test_random_wick_controls(self):
        rng = random.Random(940198)
        for n in range(1, 6):
            for _ in range(7):
                xs = [Fraction(rng.randrange(-6, 7), rng.randrange(1, 8))
                      for _ in range(n)]
                self.assertTrue(gaussian_isometry_moment_identity(xs))

    def test_exact_volume_packing_witnesses(self):
        for d in (1, 2, 3, 4):
            for D in (Fraction(1), Fraction(6, 5), Fraction(3, 2), Fraction(2)):
                n = first_forbidden_equilateral_size(d, D)
                self.assertTrue(equilateral_packing_obstruction(n, d, D))
                self.assertFalse(equilateral_packing_obstruction(n-1, d, D))
        self.assertEqual(first_forbidden_equilateral_size(2, Fraction(6, 5)), 12)

    def test_equilateral_optimal_one_dim_distortion(self):
        for n in range(2, 30):
            self.assertEqual(exact_line_equilateral_distortion(n), n-1)
        with self.assertRaises(ValueError):
            exact_line_equilateral_distortion(1)

    def test_robust_epsilon_ratio_exact(self):
        for D in (Fraction(1), Fraction(5,4), Fraction(17,7)):
            delta = Fraction(3, 4)
            Dprime = 2 * D
            epsilon = robust_net_threshold(delta, D, Dprime)
            self.assertEqual(epsilon, delta/6)
            self.assertEqual(net_transfer_distortion(delta, epsilon, D), Dprime)
            self.assertLess(
                net_transfer_distortion(delta, epsilon/2, D), Dprime)
            self.assertGreater(
                net_transfer_distortion(delta, epsilon*Fraction(6,5), D), Dprime)

    def test_robust_general_factors(self):
        delta = Fraction(7, 5)
        D = Fraction(4, 3)
        for alpha in (Fraction(3,2), Fraction(7,4), Fraction(5)):
            e = robust_net_threshold(delta, D, alpha*D)
            self.assertEqual(
                net_transfer_distortion(delta, e, D), alpha*D)
            self.assertLess(e, delta/2)

    def test_epsilon_cover_refinement(self):
        cap = doubling_cover_bound(4, 1, Fraction(1,8))
        self.assertEqual(cap["steps"], 5)
        self.assertEqual(cap["net_count_upper"], 1024)
        for eps in (Fraction(1,2), Fraction(1,4), Fraction(1,8)):
            c = doubling_cover_bound(8, 1, eps)
            self.assertLessEqual(c["final_radius"], eps/2)
            self.assertEqual(c["net_count_upper"], 8**c["steps"])

    def test_rational_l2_to_l4_pair_certificate(self):
        source = [[0, 0], [1, 0], [0, 1], [1, 1]]
        good = exact_l2_to_l4_certificate(source, source, Fraction(1,2), 2)
        self.assertTrue(good["passed"])
        self.assertEqual(good["pairs_checked"], 6)
        bad = exact_l2_to_l4_certificate(source, [[0,0]]*4, Fraction(1,2), 2)
        self.assertFalse(bad["passed"])
        self.assertEqual(len(bad["failed_pairs"]), 6)

    def test_coarse_additive_distortion_toy(self):
        states = [Fraction(i,2) for i in range(9)]
        net = [Fraction(i) for i in range(5)]
        code = {x: x for x in net}
        result = coarse_embedding_1d_certificate(
            states,net,code,Fraction(1,2),1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["pairs_checked"], 36)

    def test_rejects_invalid_geometry_and_source(self):
        with self.assertRaises(ValueError):
            doubling_cover_bound(0,1,Fraction(1,4))
        with self.assertRaises(ValueError):
            net_transfer_distortion(1,Fraction(1,2),1)
        with self.assertRaises(ValueError):
            robust_net_threshold(1,2,2)
        with self.assertRaises(ValueError):
            exact_l2_to_l4_certificate([[0],[0]], [[0],[0]], 1, 2)
        with self.assertRaises(ValueError):
            coarse_embedding_1d_certificate(
                [0,1,2],[0,2],{Fraction(0):0,Fraction(2):2},
                Fraction(1,4),1)
        with self.assertRaises(ValueError):
            coarse_embedding_1d_certificate(
                [0,1,2],[0,2],{Fraction(0):0,Fraction(2):1},
                1,1)

    def test_fixed_reference_receipt(self):
        r = control_report()
        self.assertEqual(r["status"], "CONTROL_ONLY")
        self.assertEqual(r["equilateral_N_first_forbidden_d2_D6over5"], 12)
        self.assertEqual(r["robust_epsilon_at_delta1_D1_Dprime2"], "1/6")
        self.assertFalse(r["constructs_compact_doubling_witness"])
        self.assertFalse(r["runs_trained_models"])


if __name__ == "__main__":
    unittest.main()
