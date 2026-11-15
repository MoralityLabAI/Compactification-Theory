"""Deterministic exact search checks and explicit extension tests for COE-01."""
import itertools
import math
import random
import unittest

try:
    from .observer_codes import (check_graph, optimal_coloring, minimum_observer,
        validate_code, hamming, complete_edges, clique_bound_holds,
        extend_probe, vertex_point, metric_quotient_observer)
except ImportError:
    from observer_codes import (check_graph, optimal_coloring, minimum_observer,
        validate_code, hamming, complete_edges, clique_bound_holds,
        extend_probe, vertex_point, metric_quotient_observer)


class TestChromaticObserver(unittest.TestCase):
    def test_chromatic_families(self):
        cases = [
            ("complete", 6, complete_edges(6), 6, 3),
            ("path", 8, [(i, i + 1) for i in range(7)], 2, 1),
            ("odd cycle", 5, [(i, (i + 1) % 5) for i in range(5)], 3, 2),
            ("edgeless", 6, [], 1, 0),
        ]
        for label, n, edges, chromatic, k_expected in cases:
            with self.subTest(label=label):
                coloring = optimal_coloring(n, edges)
                self.assertEqual(max(coloring) + 1, chromatic)
                k, words = minimum_observer(n, edges)
                self.assertEqual(k, k_expected)
                self.assertTrue(validate_code(n, edges, k, words))

    def test_all_graphs_on_four_vertices(self):
        edges = complete_edges(4)
        for mask in range(1 << len(edges)):
            chosen = [edge for i, edge in enumerate(edges) if mask & (1 << i)]
            colors = optimal_coloring(4, chosen)
            k, words = minimum_observer(4, chosen)
            self.assertEqual(1 << k >= max(colors) + 1, True)
            self.assertTrue(validate_code(4, chosen, k, words))
            if k:
                self.assertLess(1 << (k - 1), max(colors) + 1)

    def test_robust_codes_and_packing(self):
        for n, edges in [(2, [(0, 1)]), (4, complete_edges(4)),
                         (5, complete_edges(5)),
                         (6, [(i, i + 1) for i in range(5)])]:
            k, words = minimum_observer(n, edges, margin=3)
            self.assertTrue(validate_code(n, edges, k, words, margin=3))
            # Deleting any two coordinates still distinguishes adjacent nodes.
            for a, b in edges:
                xor = words[a] ^ words[b]
                for removed in itertools.combinations(range(k), 2):
                    mask = sum(1 << i for i in removed)
                    self.assertNotEqual(xor & ~mask, 0)
            # Balls of one bit-flip around adjacent codes are disjoint.
            for a, b in edges:
                ball_a = {words[a]} | {words[a] ^ (1 << i) for i in range(k)}
                ball_b = {words[b]} | {words[b] ^ (1 << i) for i in range(k)}
                self.assertFalse(ball_a & ball_b)
            if len(edges) == len(complete_edges(n)):
                self.assertTrue(clique_bound_holds(n, k, 1))

    def test_mcshane_extension(self):
        n = 6
        k, words = minimum_observer(n, complete_edges(n))
        self.assertEqual(extend_probe((0.0,) * n, n, k, words), (0.5,) * k)
        for v in range(n):
            actual = extend_probe(vertex_point(v, n), n, k, words)
            expected = tuple(float((words[v] >> j) & 1) for j in range(k))
            self.assertEqual(actual, expected)
        rng = random.Random(2198)
        for _ in range(300):
            x = tuple(rng.uniform(-2, 2) for _ in range(n))
            y = tuple(rng.uniform(-2, 2) for _ in range(n))
            fx, fy = extend_probe(x, n, k, words), extend_probe(y, n, k, words)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(fx, fy)),
                                 math.dist(x, y) + 1e-12)


    def test_metric_quotient_chain_and_boundary(self):
        from fractions import Fraction as F
        def line_metric(xs):
            return tuple(tuple(abs(a - b) for b in xs) for a in xs)
        xs = [F(0), F(3, 5), F(6, 5)]
        d = line_metric(xs)
        # The endpoints are more than one unit apart, but the intermediate
        # node forces both into the same locally constant bit-observer class.
        with self.assertRaises(ValueError):
            metric_quotient_observer(d, [(0, 2)], 1)
        k, code, comps = metric_quotient_observer(d, [], 1)
        self.assertEqual((k, comps), (0, (0, 0, 0)))
        self.assertEqual(code, (0, 0, 0))
        k, code, comps = metric_quotient_observer(d, [(0, 2)], 2)
        self.assertEqual(k, 1)
        self.assertEqual(len(set(comps)), 3)
        # Equality d=1/L permits a one-bit step: the short-edge relation
        # uses strict < rather than <=.
        k, words, comps = metric_quotient_observer(line_metric([F(0), F(1)]),
                                                   [(0, 1)], 1)
        self.assertEqual(k, 1)
        self.assertNotEqual(*words)
        self.assertEqual(comps, (0, 1))

    def test_metric_quotient_robust_code(self):
        from fractions import Fraction as F
        xs = [F(0), F(2, 5), F(4, 5), F(2), F(13, 5)]
        d = tuple(tuple(abs(a - b) for b in xs) for a in xs)
        k, words, comps = metric_quotient_observer(d, [(0, 3)], 1, margin=3)
        self.assertEqual(k, 3)
        self.assertEqual(comps, (0, 0, 0, 1, 1))
        self.assertEqual(hamming(words[0], words[3]), 3)
        with self.assertRaises(ValueError):
            metric_quotient_observer(d, [(0, 2)], 1, margin=3)

    def test_invalid_metrics_rejected(self):
        from fractions import Fraction as F
        with self.assertRaises(ValueError):
            metric_quotient_observer(((F(0), F(1)), (F(2), F(0))), [], 1)
        with self.assertRaises(ValueError):
            metric_quotient_observer(((F(0), F(1), F(3)),
                                     (F(1), F(0), F(1)),
                                     (F(3), F(1), F(0))), [], 1)

    def test_invalid_graph_rejected(self):
        with self.assertRaises(ValueError):
            check_graph(4, [(1, 1)])
        with self.assertRaises(ValueError):
            minimum_observer(2, [(0, 1)], margin=0)


if __name__ == "__main__":
    unittest.main()
