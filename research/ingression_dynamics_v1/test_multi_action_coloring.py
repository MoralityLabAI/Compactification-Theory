import random
import unittest
from fractions import Fraction
from itertools import combinations, product

from research.ingression_dynamics_v1.multi_action_coloring import (
    normalize_edges, edge_hit_rates, maximum_rate_radius,
    exact_three_colorings, best_radius_with_three_transient_classes,
    witness_from_coloring, actual_macro_kernels_from_rates,
    joint_action_obstruction, QUARTER,
    conflict_edges_for_tolerance, coloring_witness_for_tolerance,
    optimal_radius_by_conflict_graph,
)


class MultiActionTests(unittest.TestCase):
    def test_all_four_vertex_graphs(self):
        n = 4
        possible = tuple(combinations(range(n), 2))
        count = 0
        for edge_on in product((False, True), repeat=len(possible)):
            edges = [e for e, include in zip(possible, edge_on) if include]
            colorable = next(exact_three_colorings(n, edges), None) is not None
            best = best_radius_with_three_transient_classes(n, edges)
            self.assertEqual(best <= QUARTER, colorable)
            self.assertEqual(witness_from_coloring(n, edges) is not None, colorable)
            if colorable:
                witness = witness_from_coloring(n, edges)
                self.assertEqual(witness[-1], 3)
                self.assertLessEqual(
                    maximum_rate_radius(edge_hit_rates(n, edges), witness[:-1]),
                    QUARTER)
            count += 1
        self.assertEqual(count, 64)

    def test_seeded_five_vertex_graphs(self):
        n = 5
        possible = tuple(combinations(range(n), 2))
        rng = random.Random(20261104)
        for _ in range(60):
            edges = [e for e in possible if rng.randrange(2)]
            colorable = next(exact_three_colorings(n, edges), None) is not None
            best = best_radius_with_three_transient_classes(n, edges)
            self.assertEqual(best <= QUARTER, colorable)
        self.assertGreater(
            best_radius_with_three_transient_classes(n, possible), QUARTER)

    def test_complete_four_vs_cycle_four(self):
        for n, edges, want in (
            (4, tuple(combinations(range(4), 2)), False),
            (4, ((0, 1), (1, 2), (2, 3), (3, 0)), True),
            (1, (), True),
        ):
            has = next(exact_three_colorings(n, edges), None) is not None
            self.assertEqual(has, want)
            self.assertEqual(
                best_radius_with_three_transient_classes(n, edges) <= QUARTER,
                want)

    def test_joint_actions_require_more_states_than_either_alone(self):
        rates = joint_action_obstruction()
        candidates = list(product(range(2), repeat=4))
        x = min(maximum_rate_radius({"x": rates["x"]}, p) for p in candidates)
        y = min(maximum_rate_radius({"y": rates["y"]}, p) for p in candidates)
        xy = min(maximum_rate_radius(rates, p) for p in candidates)
        self.assertEqual((x, y, xy), (0, 0, Fraction(1, 2)))
        self.assertEqual(
            min(maximum_rate_radius(rates, p)
                for p in product(range(4), repeat=4)), 0)

    def test_constructive_macro_kernel(self):
        n = 3
        edges = ((0, 1), (1, 2))
        colors = (0, 1, 0)
        rates = edge_hit_rates(n, edges)
        macro = actual_macro_kernels_from_rates(rates, colors)
        goal = max(colors) + 1
        self.assertEqual(goal, 2)
        for action, row in rates.items():
            for s in range(n):
                z = colors[s]
                projected = {z: 1-row[s], goal: row[s]}
                center = macro[action][z]
                self.assertEqual(sum(center.values()), 1)
                tv = sum(abs(projected[t]-center[t])
                         for t in (z, goal)) / 2
                self.assertLessEqual(tv, QUARTER)

    def test_invalid_graph(self):
        for n, edges in ((0, ()), (3, ((0, 0),)), (3, ((0, 3),))):
            with self.assertRaises(ValueError):
                edge_hit_rates(n, edges)


class ConflictGraphTests(unittest.TestCase):
    def test_threshold_characterization_all_three_state_rate_vectors(self):
        values = (Fraction(0), Fraction(1, 2), Fraction(1))
        n = 3
        cases = 0
        for flat in product(values, repeat=6):
            rates = {"a": flat[:n], "b": flat[n:]}
            for k in (1, 2, 3):
                exact = min(maximum_rate_radius(rates, labels)
                            for labels in product(range(k), repeat=n))
                computed, assignment = optimal_radius_by_conflict_graph(rates, k)
                self.assertEqual(exact, computed)
                self.assertLessEqual(maximum_rate_radius(rates, assignment),
                                     computed)
                for eps in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
                    witness = coloring_witness_for_tolerance(rates, k, eps)
                    self.assertEqual(witness is not None, exact <= eps)
                cases += 1
        self.assertEqual(cases, 2187)

    def test_conflict_graph_edge_exactness(self):
        rates = edge_hit_rates(4, ((0, 1), (2, 3)))
        self.assertEqual(set(conflict_edges_for_tolerance(rates, QUARTER)),
                         {(0, 1), (2, 3)})

    def test_negative_tolerance_is_invalid(self):
        with self.assertRaises(ValueError):
            conflict_edges_for_tolerance({"a": (Fraction(0),)}, -1)


if __name__ == "__main__":
    unittest.main()
