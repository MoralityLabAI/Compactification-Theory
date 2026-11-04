"""QO-04 exact independent finite checks for square cover / hit-rate quotients."""
import random
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.geometric_quotient import (
    candidates, cover_with_k, minimum_cover, optimum_for_k,
    one_dimensional_greedy, minimum_label_bits,
    normalize_cover_instance, pad_budget_to_binary,
    points_as_action_rates, class_assignment,
)
from research.ingression_dynamics_v1.hit_rate_dp import optimal_rate_groups
from research.ingression_dynamics_v1.multi_action_coloring import maximum_rate_radius


def independent_assignment_radius(points, assignment):
    """Direct partition error, independently computed from coordinate ranges."""
    dimension = len(points[0])
    blocks = {label: [points[i] for i in range(len(points))
                      if assignment[i] == label] for label in set(assignment)}
    return max(((max(p[j] for p in block)-min(p[j] for p in block))/2
                for block in blocks.values() for j in range(dimension)), default=Q(0))


def exhaustive_radius(points, k):
    """Independent brute-force assignment oracle; no candidate boxes."""
    return min(independent_assignment_radius(points, labels)
               for labels in product(range(k), repeat=len(points)))


class GeometryAudit(unittest.TestCase):
    def test_mixed_coordinate_corner_required(self):
        pts = ((Q(0), Q(1)), (Q(1), Q(0)), (Q(1), Q(1)))
        witness = minimum_cover(pts, Q(1))
        self.assertEqual(len(witness), 1)
        self.assertEqual(witness[0][0], (Q(0), Q(0)))
        self.assertNotIn(witness[0][0], pts)
        self.assertEqual(witness[0][1], 0b111)

    def test_all_729_three_point_two_action_rate_matrices(self):
        options = (Q(0), Q(1,2), Q(1))
        count = 0
        for coords in product(options, repeat=6):
            pts = tuple((coords[i], coords[3+i]) for i in range(3))
            for k in (1, 2, 3):
                expected = exhaustive_radius(pts, k)
                got, witness, labels = optimum_for_k(pts, k)
                self.assertEqual(got, expected)
                self.assertLessEqual(len(witness), k)
                self.assertEqual(labels[-1], len(witness))
                self.assertEqual(independent_assignment_radius(pts, labels[:-1]), got)
                for error in (Q(0), Q(1,4), Q(1,2)):
                    covered = cover_with_k(pts, 2*error, k)
                    self.assertEqual(covered is not None, expected <= error)
                count += 1
        self.assertEqual(count, 2187)  # 729 matrices * 3 class budgets

    def test_1d_greedy_matches_polynomial_rate_dp(self):
        rng = random.Random(20261104)
        for n in range(1, 11):
            for _ in range(8):
                rates = [Q(rng.randrange(12),11) for _ in range(n)]
                points = tuple((p,) for p in rates)
                for k in (1, 2, 3, 4):
                    exact, _, _ = optimal_rate_groups(rates, k)
                    proposed, _, _ = optimum_for_k(points, k)
                    self.assertEqual(proposed, exact)
                for side in (Q(0), Q(1, 5), Q(1, 2), Q(1)):
                    greedy = one_dimensional_greedy(points, side)
                    checked = minimum_cover(points, side)
                    self.assertEqual(len(greedy), len(checked))

    def test_complete_square_needs_four_classes_at_half_side(self):
        pts = tuple((Q(x), Q(y)) for x, y in product((0,1), repeat=2))
        self.assertEqual(len(minimum_cover(pts, Q(1,2))), 4)
        self.assertIsNone(cover_with_k(pts, Q(1,2), 3))
        self.assertIsNotNone(cover_with_k(pts, Q(1,2), 4))
        self.assertEqual(minimum_label_bits(pts, Q(1,2)), 3)

    def test_constant_half_side_has_four_square_cover(self):
        rng = random.Random(4811)
        for n in range(1, 8):
            for _ in range(10):
                pts = tuple((Q(rng.randrange(101),100),
                             Q(rng.randrange(101),100)) for _ in range(n))
                self.assertIsNotNone(cover_with_k(pts, Q(1,2), 4))

    def test_exact_normalization_preserves_linf_coverage(self):
        raw = ((Q(-2), Q(5)), (Q(-1), Q(5)), (Q(4), Q(6)),
               (Q(4), Q(9)))
        width = Q(3,2)
        new, scaled = normalize_cover_instance(raw, width)
        self.assertTrue(all(0 <= v <= 1 for p in new for v in p))
        for k in range(1, 5):
            self.assertEqual(cover_with_k(raw, width, k) is not None,
                             cover_with_k(new, scaled, k) is not None)

    def test_bit_budget_padding_exact_equivalence(self):
        rng = random.Random(2468)
        trials = 0
        for n in range(2, 5):
            for _ in range(6):
                pts = tuple((Q(rng.randrange(7), 3),
                             Q(rng.randrange(7), 3)) for _ in range(n))
                for width in (Q(1,3), Q(1,2)):
                    for k in range(1,n+1):
                        original = cover_with_k(
                            normalize_cover_instance(pts, width)[0],
                            normalize_cover_instance(pts, width)[1], k) is not None
                        padded, side, bits, extra = pad_budget_to_binary(pts, width, k)
                        new_budget = (1 << bits)-1
                        padded_feasible = cover_with_k(
                            padded, side, new_budget) is not None
                        self.assertEqual(original, padded_feasible)
                        self.assertEqual(extra, new_budget-k)
                        self.assertEqual(len(padded), n+extra)
                        trials += 1
        self.assertEqual(trials, 108)

    def test_rate_interpretation_matches_cover_grouping(self):
        pts = ((Q(0), Q(0)), (Q(1,2), Q(1,4)), (Q(1), Q(1)))
        error, witness, labels = optimum_for_k(pts, 2)
        rates = points_as_action_rates(pts)
        self.assertEqual(
            maximum_rate_radius(rates, labels[:-1]), error)
        self.assertLessEqual(len(witness)+1, 3)

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            candidates(((Q(-1), Q(0)),), Q(1))
        with self.assertRaises(ValueError):
            cover_with_k(((Q(0),),), Q(1), 0)
        with self.assertRaises(ValueError):
            minimum_cover(((Q(0), Q(1)), (Q(0),)), Q(1))
        with self.assertRaises(ValueError):
            normalize_cover_instance(((0, 0),), 0)
        with self.assertRaises(ValueError):
            pad_budget_to_binary(((0,0),), 1, 2)
        with self.assertRaises(ValueError):
            one_dimensional_greedy(((Q(0), Q(0)),), Q(1))


if __name__ == "__main__":
    unittest.main()
