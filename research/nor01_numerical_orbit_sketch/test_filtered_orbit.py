"""NOR-02: exact disk-geometry and noncommuting-filter controls."""
from fractions import Fraction
import math
import unittest

from filtered_orbit import (
    A, R, Q, curvature_gain, nested_disk_model,
    matrix_filtered_state, convolution_filtered_state, run_case,
)
from orbit_sketch import JordanBlock, norm, subtract


class FilteredOrbitTests(unittest.TestCase):
    def test_nested_disk_exact_rational(self):
        model = nested_disk_model(32)
        self.assertEqual(model.q, Fraction(7, 8))
        self.assertEqual(model.dimension, 64)
        self.assertEqual(model.blocks[0].numerical_radius, Q)
        for block in model.blocks:
            self.assertLessEqual(abs(block.a - A) + block.b / 2, R)
        with self.assertRaises(ValueError):
            nested_disk_model(33)

    def test_two_matrix_coefficients_do_not_commute(self):
        # B0=[[0,1],[0,0]], B1=[[0,0],[1,0]].
        b0 = ((0, 1), (0, 0))
        b1 = ((0, 0), (1, 0))
        def multiply(x, y):
            return tuple(tuple(sum(x[i][k] * y[k][j] for k in (0, 1))
                               for j in (0, 1)) for i in (0, 1))
        self.assertEqual(multiply(b0, b1), ((1, 0), (0, 0)))
        self.assertEqual(multiply(b1, b0), ((0, 0), (0, 1)))

    def test_independent_polynomial_evaluation(self):
        model = nested_disk_model(8)
        x = [((i % 4) - 1.5) / 7 for i in range(model.dimension)]
        u = (0.6, -0.8)
        for order in (1, 2, 3, 4):
            for t in range(0, 14):
                analytic = matrix_filtered_state(model, x, t, order, u)
                binomial = convolution_filtered_state(model, x, t, order, u)
                self.assertLess(norm(subtract(analytic, binomial)), 3e-12)

    def test_disk_curvature_inequality_exact_rational_grid(self):
        # Exact rational identity used in the analytic proof:
        # |z-q|^2 <= (b/a) (q^2-|z|^2) for z in disk D(a,b).
        for i in range(-16, 17):
            for j in range(-16, 17):
                u, v = R * Fraction(i, 16), R * Fraction(j, 16)
                if u * u + v * v <= R * R:
                    abs_z_squared = (A + u) ** 2 + v ** 2
                    dist_q_squared = (u - R) ** 2 + v ** 2
                    self.assertLessEqual(dist_q_squared,
                                         (R / A) * (Q * Q - abs_z_squared))
                    for t in (0, 1, 2, 8, 20):
                        for order in (1, 2, 3):
                            beta_squared = Fraction(
                                (t ** t if t else 1) * order ** order,
                                (t + order) ** (t + order),
                            )
                            bound_squared = ((R / A) ** order *
                                             Q ** (2 * (t + order)) * beta_squared)
                            self.assertLessEqual(abs_z_squared ** t *
                                                 dist_q_squared ** order,
                                                 bound_squared)

    def test_extremum_is_reachable_on_boundary(self):
        # The analytic one-dimensional optimizer lies in the radius range
        # of the boundary circle for t>=1 and orders 1..4.
        q2 = Q * Q
        lower_sq = (A - R) ** 2
        for t in (1, 2, 8, 18, 40):
            for order in (1, 2, 3, 4):
                s_star = q2 * Fraction(t, t + order)
                self.assertGreaterEqual(s_star, lower_sq)
                self.assertLessEqual(s_star, q2)
                # Hence cos(theta)=(s_star-a^2-b^2)/(2ab) is in [-1,1].
                cos_theta = (s_star - A * A - R * R) / (2 * A * R)
                self.assertLessEqual(abs(cos_theta), 1)

    def test_bound_monotone_for_late_times(self):
        for order in (1, 2, 3, 4):
            previous = curvature_gain(0, order)
            for t in range(1, 65):
                next_value = curvature_gain(t, order)
                self.assertLessEqual(next_value, previous + 1e-16)
                previous = next_value

    def test_matrix_filtered_tail_bound(self):
        model = nested_disk_model(32)
        x = [z for _ in range(32) for z in (0.0, 1 / math.sqrt(32))]
        u = (1 / math.sqrt(2), 1 / math.sqrt(2))
        for order in (1, 2, 3, 4):
            for t in range(0, 32):
                filtered = matrix_filtered_state(model, x, t, order, u)
                self.assertLessEqual(norm(filtered),
                                     2 * curvature_gain(t, order) + 1e-13)

    def test_sharp_factor_two_witness_for_scalar_case(self):
        # J(0,2q) has W=closed radius-q disk, with ||J||=2q.
        q = Fraction(7, 8)
        block = JordanBlock(Fraction(0), 2 * q)
        self.assertEqual(block.numerical_radius, q)
        self.assertEqual(block.operator_norm, float(2 * q))

    def test_stitching_strictly_improves_old_budget(self):
        result = run_case()
        self.assertLess(result["sharp_stitch_bound"], result["legacy_stitch_bound"])
        self.assertLess(result["curvature_tail_radius"],
                        result["complete_crouzeix_disk_only_tail_radius"])
        self.assertLess(result["complete_crouzeix_disk_only_tail_radius"],
                        result["coefficient_triangle_tail_radius"])
        self.assertTrue(result["finite_probe_below_bound"])
        self.assertLess(result["sketch_dimension"], result["ambient_hilbert_dimension"])

    def test_other_seed_and_filter_order(self):
        for order in (1, 3):
            result = run_case(block_count=12, cutoff=12, k=16, order=order,
                              p=2, seed=100 + order, tail_checks=12)
            self.assertTrue(result["finite_probe_below_bound"])
            self.assertLessEqual(result["sharp_stitch_bound"],
                                 result["legacy_stitch_bound"])


if __name__ == "__main__":
    unittest.main()
