"""Exact reference tests. Run with unittest (also with python -O)."""
import unittest

from model import (
    State, benchmark, first_separating_fixed_word, freeze_count, full_reference,
    odd_triangle_contextuality, output_classes, passive_cycle_lengths, step,
    trace_set, trajectory,
)


class OICTests(unittest.TestCase):
    def test_parameter_validation(self):
        for args in ((1, "fixed", State(0, 0), 0),
                     (4, "bad", State(0, 0), 0),
                     (4, "fixed", State(2, 0), 0),
                     (4, "rotating", State(0, 4), 0),
                     (4, "rotating", State(0, 0), 2)):
            with self.assertRaises(ValueError):
                step(*args)

    def test_perfect_passive_indistinguishability(self):
        for n in range(3, 10):
            for x in (0, 1):
                for z in range(n):
                    for t in (0, 1, n, 2 * n):
                        a = trajectory(n, "fixed", z, (0,) * t, x)
                        b = trajectory(n, "rotating", z, (0,) * t, x)
                        self.assertEqual(a, b)
                        self.assertEqual(a, tuple(x ^ (i % 2) for i in range(t + 1)))

    def test_passive_quotient_always_two_classes(self):
        for n in range(3, 12):
            for variant in ("fixed", "rotating"):
                for horizon in (0, 1, 2 * n):
                    classes = output_classes(n, variant, horizon)
                    self.assertEqual(len(classes), 2)
                    self.assertEqual(sorted(map(len, classes.values())), [n, n])

    def test_transition_graph_cycle_topology(self):
        for n in range(3, 13):
            a, b = passive_cycle_lengths(n, "fixed"), passive_cycle_lengths(n, "rotating")
            self.assertEqual(a, (2,) * n)
            self.assertEqual(b, (n,) * 2 if n % 2 == 0 else (2 * n,))
            self.assertNotEqual(a, b)

    def test_unknown_phase_minimal_fixed_schedule(self):
        # Independently enumerate every possible binary open-loop word up to horizon n.
        for n in range(3, 9):
            t, word = first_separating_fixed_word(n)
            self.assertEqual(t, n)
            self.assertEqual(word, (1,) * n)
            self.assertIsNone(first_separating_fixed_word(n, max_horizon=n - 1))
            for x in (0, 1):
                a = trace_set(n, "fixed", word, visible_initial=x)
                b = trace_set(n, "rotating", word, visible_initial=x)
                self.assertTrue(a.isdisjoint(b))
                self.assertEqual({freeze_count(y) for y in a}, {0, n})
                self.assertEqual({freeze_count(y) for y in b}, {1})

    def test_adaptive_lower_bound_explicit_pair(self):
        # For T<n, B starts at z=1 and cannot hit 0 during the probed transitions.
        # A starts at z=1, which never reaches 0. Equality holds for EVERY word,
        # hence also along the common branch of any adaptive policy.
        from itertools import product
        for n in range(3, 8):
            for t in range(n):
                for word in product((0, 1), repeat=t):
                    a = trajectory(n, "fixed", 1, word)
                    b = trajectory(n, "rotating", 1, word)
                    self.assertEqual(a, b)

    def test_contextuality_is_a_separate_obstruction(self):
        result = odd_triangle_contextuality()
        self.assertEqual(result["local_sections_per_context"], [2, 2, 2])
        self.assertTrue(result["uniform_overlap_marginals"])
        self.assertEqual(result["global_section_count"], 0)
        self.assertEqual(result["z2_cycle_holonomy"], 1)

    def test_deterministic_receipt_shape(self):
        result = benchmark(4)
        self.assertEqual(result["passive_cycle_lengths_fixed"], [2, 2, 2, 2])
        self.assertEqual(result["passive_cycle_lengths_rotating"], [4, 4])
        self.assertEqual(result["earliest_separating_fixed_action_horizon"], 4)
        self.assertEqual(result["first_separating_fixed_action_word"], "1111")
        self.assertEqual(result["possible_freeze_counts_fixed"], [0, 4])
        self.assertEqual(result["possible_freeze_counts_rotating"], [1])
        self.assertEqual(len(full_reference()["results"]), 6)


if __name__ == "__main__":
    unittest.main()
