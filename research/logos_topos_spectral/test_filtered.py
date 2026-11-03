from fractions import Fraction as Q
from itertools import combinations, product
import unittest

from filtered import (MarkedSpace, Interpretation, seminorm, mass_block, multiply,
                      block_commutator, eigenvalue_square, resolvent_square_bound)


class FilteredTests(unittest.TestCase):
    def setUp(self):
        self.source = MarkedSpace(tuple(product((0, 1), repeat=2)))
        self.target = MarkedSpace(((0, 0), (0, 1), (1, 0)))

    def test_exact_depth_formula_all_small_maps(self):
        # Includes maps with image a proper subset, and constant maps.
        for image in product(range(3), repeat=4):
            f = Interpretation(self.source, self.target, image)
            self.assertEqual(f.lipschitz_constant(), f.profile_constant())
            for c in range(4):
                self.assertEqual(f.has_delay(c), f.lipschitz_constant() <= 2**c)

    def test_composition_bounds_all_small_maps(self):
        z = MarkedSpace(((0,), (1,)))
        for image_f in product(range(3), repeat=4):
            f = Interpretation(self.source, self.target, image_f)
            for image_g in product(range(2), repeat=3):
                g = Interpretation(self.target, z, image_g)
                h = f.then(g)
                self.assertLessEqual(h.least_delay(), f.least_delay()+g.least_delay())
                self.assertLessEqual(h.lipschitz_constant(), f.lipschitz_constant()*g.lipschitz_constant())
                for n, cost in enumerate(g.depth_profile()):
                    self.assertLessEqual(h.depth_profile()[n], f.depth_profile()[cost])

    def test_optimal_pullback_all_small_maps(self):
        for image in product(range(3), repeat=4):
            f = Interpretation(self.source, self.target, image)
            _, achieved = f.optimal_pullback_witness()
            self.assertEqual(achieved, f.lipschitz_constant())
            for a in product((Q(-1), Q(0), Q(1)), repeat=3):
                self.assertLessEqual(seminorm(self.source, f.pullback(a)),
                                     f.lipschitz_constant()*seminorm(self.target, a))

    def test_conservative_locus_tests(self):
        for image in product(range(3), repeat=4):
            f = Interpretation(self.source, self.target, image)
            for mask in range(8):
                locus = {j for j in range(3) if mask & (1 << j)}
                preimage = {i for i, j in enumerate(image) if j in locus}
                if f.conservative():
                    self.assertEqual(len(locus) == 3, len(preimage) == 4)
                    self.assertEqual(not locus, not preimage)

    def test_sharp_two_observation_delay(self):
        x = MarkedSpace(tuple(product((0, 1), repeat=5)))
        y = MarkedSpace(tuple(product((0, 1), repeat=3)))
        f = Interpretation(x, y, tuple(y.points.index(p[2:]) for p in x.points))
        self.assertEqual(f.depth_profile(), (0, 3, 4, 5))
        self.assertEqual(f.least_delay(), 2)
        self.assertEqual(f.lipschitz_constant(), 4)
        self.assertEqual(f.optimal_pullback_witness()[1], 4)
        self.assertTrue(f.conservative())

    def test_zero_delay_isometries(self):
        for image in product(range(4), repeat=4):
            if len(set(image)) == 4:
                f = Interpretation(self.source, self.source, image)
                inv = Interpretation(self.source, self.source, tuple(image.index(i) for i in range(4)))
                if f.has_delay(0) and inv.has_delay(0):
                    for i, j in combinations(range(4), 2):
                        self.assertEqual(self.source.dist(i, j), self.source.dist(image[i], image[j]))

    def test_contractions_and_constant_maps(self):
        x = MarkedSpace(((0,), (1,)))
        y = MarkedSpace(((0, 0, 0), (0, 0, 1)))
        f = Interpretation(x, y, (0, 1))
        self.assertEqual(f.depth_profile(), (0, 0, 0, 1))
        self.assertEqual(f.profile_constant(), Q(1, 4))
        const = Interpretation(x, y, (0, 0))
        self.assertEqual(const.profile_constant(), 0)
        self.assertEqual(const.lipschitz_constant(), 0)

    def test_singleton_and_empty_marking(self):
        one = MarkedSpace(((),))
        f = Interpretation(one, self.target, (2,))
        self.assertEqual(f.lipschitz_constant(), 0)
        self.assertEqual(f.profile_constant(), 0)
        self.assertEqual(f.optimal_pullback_witness()[1], 0)

    def test_mass_commutator_identity(self):
        for distance, mass, a, b in product((Q(1), Q(1, 2), Q(1, 8)), (Q(0), Q(3), Q(32)), (Q(-2), Q(0)), (Q(1), Q(5))):
            actual = block_commutator(distance, mass, a, b)
            self.assertEqual(actual, ((Q(0), (b-a)/distance), ((a-b)/distance, Q(0))))
            self.assertEqual(actual, block_commutator(distance, 0, a, b))

    def test_mass_square_identity(self):
        for d, m in product((Q(1), Q(1, 2), Q(1, 16)), (Q(0), Q(2), Q(128))):
            block = mass_block(d, m)
            eigen_sq = eigenvalue_square(d, m)
            self.assertEqual(multiply(block, block), ((eigen_sq, Q(0)), (Q(0), eigen_sq)))

    def test_resolvent_tail_certificate(self):
        for first in range(12):
            bound = resolvent_square_bound(first)
            for j in range(first, first+5):
                for d in (Q(1), Q(1, 8), Q(1, 1024)):
                    self.assertLessEqual(1/(1+eigenvalue_square(d, Q(2**j))), bound)
            if first:
                self.assertLess(bound, resolvent_square_bound(first-1))

    def test_mass_changes_spectrum_preserves_observable_geometry(self):
        for i, j in combinations(range(4), 2):
            d = self.source.dist(i, j)
            self.assertNotEqual(eigenvalue_square(d, 2), eigenvalue_square(d, 8))
            self.assertEqual(block_commutator(d, 2, 0, 1), block_commutator(d, 8, 0, 1))

    def test_invalid_data(self):
        for points in ((), ((0,), (0,)), ((0,), (1, 0)), ((2,),)):
            with self.assertRaises(ValueError): MarkedSpace(points)
        for image in ((0,), (0, 0, 0, 3), (0, 0, 0, -1)):
            with self.assertRaises(ValueError): Interpretation(self.source, self.target, image)
        with self.assertRaises(ValueError): mass_block(Q(0), Q(1))
        with self.assertRaises(ValueError): resolvent_square_bound(-1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
