"""Exhaustive q=2 witnesses, all-depth recurrence checks, and corruption controls."""
from collections import Counter
from fractions import Fraction as F
from itertools import product
import copy
import json
import random
import unittest

from unique_games_recursive_v02 import (
    aggregate, basis, children, dot, evolution, gl3, height_one,
    height_one_lift_detection, linear, noise_row, quadratic, rank_options,
    receipt, restriction_rows, spaces, span, update_basis, verify_receipt,
)


class RecursiveLatentTests(unittest.TestCase):
    def test_fixed_tree_and_uniform_dual_orientation(self):
        self.assertEqual(Counter(len(basis(s)) for s in spaces()), {0: 1, 1: 7, 2: 7, 3: 1})
        self.assertEqual(len(gl3()), 168)
        self.assertEqual(len(children()), 1176)
        for s in spaces():
            counts = Counter(tuple(sorted(sum(dot(z, col) << i for i, col in enumerate(g))
                                          for z in s)) for g in gl3())
            targets = [t for t in spaces() if len(t) == len(s)]
            self.assertEqual(set(counts), set(targets))
            self.assertEqual(set(counts.values()), {168 // len(targets)})

    def test_every_extension_kernel_matches_direct_restriction(self):
        maps = checks = 0
        for s in spaces():
            sb, r = basis(s), len(basis(s))
            for gamma in product(range(8), repeat=r):
                maps += 1
                gz = {linear(sb, k): linear(gamma, k) for k in range(1 << r)}
                for v in range(1, 8):
                    loss = int(v in gz and dot(v, gz[v]) == int(v == 7))
                    got = len(basis(restriction_rows(s, gamma, v)))
                    self.assertEqual(got, r - loss)
                    checks += 1
            if r:
                self.assertEqual(max(h[r - 1] for h in rank_options(s)), (0, 1, 3, 6)[r])
        self.assertEqual((maps, checks), (1017, 7119))

    def test_actual_height_one_map_and_full_lift(self):
        rng = random.Random(20261104)
        values = tuple(rng.randrange(8) for _ in children())
        initial = aggregate(values)
        self.assertEqual(height_one(values), (initial >> 3) ^ quadratic(initial & 7))
        for _ in range(16):
            i, delta = rng.randrange(len(values)), rng.randrange(8)
            changed = list(values)
            changed[i] ^= delta
            final = initial ^ linear(children()[i], delta)
            self.assertEqual(height_one(tuple(changed)), (final >> 3) ^ quadratic(final & 7))
        self.assertEqual(height_one_lift_detection((0, 0, 0)), F(43, 56))

    def test_enumerated_noise_kernel(self):
        self.assertEqual(noise_row(), (F(1, 7),) + (F(6, 49),) * 7)
        self.assertEqual(sum(noise_row()), 1)

    def test_closed_forms_against_bellman_recursion(self):
        for row in evolution(64):
            t = row['depth']
            p, d = F(row['nonlinear_change']), F(row['minimum_full_lift_detection'])
            self.assertEqual(p, F(7, 8) * F(6, 7) ** t)
            self.assertEqual(d, F(9, 10) * F(6, 7) ** t - F(1, 40) * F(1, 7) ** t)
            self.assertEqual(d / p, F(36, 35) - F(1, 35 * 6 ** t))
            self.assertGreaterEqual(d, p)
            self.assertLess(d, F(36, 35) * p)
        self.assertLess(F(evolution(9)[-1]['minimum_full_lift_detection']), F(1, 4))

    def test_future_informed_gamma_is_a_strictly_invalid_shortcut(self):
        s = tuple(range(8))
        clairvoyant = sum(min(F(1) - F(1, len(span(restriction_rows(s, g, v))))
                              for g in product(range(8), repeat=3))
                          for v in range(1, 8)) / 7
        self.assertEqual(clairvoyant, F(3, 4))
        self.assertGreater(F(evolution(1)[-1]['minimum_full_lift_detection']), clairvoyant)

    def test_receipt_mutations_and_input_checks(self):
        data = json.loads(json.dumps(receipt()))
        verify_receipt(data)
        corrupt = copy.deepcopy(data)
        corrupt['depths'][2]['minimum_full_lift_detection'] = '1/4'
        with self.assertRaises(ValueError):
            verify_receipt(corrupt)
        for depth in (-1, 129, True, 1.5):
            with self.assertRaises(ValueError):
                evolution(depth)
        with self.assertRaises(ValueError):
            height_one(())
        with self.assertRaises(ValueError):
            update_basis(0)
        with self.assertRaises(ValueError):
            rank_options((0, 1, 2))
        with self.assertRaises(ValueError):
            height_one_lift_detection((0, 0, 8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
