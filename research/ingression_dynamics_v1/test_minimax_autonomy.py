import unittest
from fractions import Fraction as Q
from itertools import product
from research.ingression_dynamics_v1.minimax_autonomy import (
    radius, optimum, direct_optimum, tv, midpoint, stochastic_fibre_diameters,
    composed_defect, tv_path_disagreement_bound, tv_return_bound, tv_minimax_center,
)

class MinimaxAutonomyTests(unittest.TestCase):
    def test_exhaustive_binary_quotients(self):
        states = (0, 1, 2)
        zs = (0, 1)
        dist = lambda x, y: Q(int(x != y))
        count = 0
        for encoding in product(zs, repeat=3):
            if len(set(encoding)) != 2:
                continue
            projection = dict(zip(states, encoding))
            for successors in product(states, repeat=3):
                updates = {"a": dict(zip(states, successors))}
                optimal, macro = optimum(states, projection, updates, zs, dist)
                self.assertEqual(optimal, direct_optimum(states, projection, updates, zs, dist))
                for s in states:
                    self.assertLessEqual(dist(projection[updates["a"][s]],
                                             macro[projection[s], "a"]), optimal)
                count += 1
        self.assertEqual(count, 162)

    def test_discrete_center_rejects_half_diameter_conjecture(self):
        self.assertEqual(radius({0, 1}, (0, 1), lambda x, y: Q(int(x != y)))[0], 1)

    def test_real_interval_midpoint(self):
        values = {Q(-2), Q(3), Q(7), Q(4)}
        candidate = (min(values) + max(values)) / 2
        self.assertEqual(max(abs(v - candidate) for v in values), Q(9, 2))

    def test_two_distributions_tv_midpoint(self):
        p = {0: Q(3, 4), 1: Q(1, 4)}
        q = {0: Q(1, 4), 1: Q(3, 4)}
        m = midpoint(p, q)
        self.assertEqual(tv(p, m), tv(q, m))
        self.assertEqual(tv(p, m), tv(p, q)/2)

    def test_three_distributions_tv_needs_two_thirds(self):
        point_masses = [{i: Q(1)} for i in range(3)]
        center = {i: Q(1, 3) for i in range(3)}
        self.assertTrue(all(tv(p, center) == Q(2, 3) for p in point_masses))
        self.assertEqual(max(tv(p, q) for p in point_masses for q in point_masses), 1)

    def test_stochastic_fibres(self):
        states = (0, 1, 2)
        projection = {0: 0, 1: 0, 2: 1}
        kernel = {"a": {0: {0: Q(3, 4), 2: Q(1, 4)},
                        1: {1: Q(1, 2), 2: Q(1, 2)}, 2: {2: Q(1)}}}
        diam = stochastic_fibre_diameters(states, projection, kernel)
        self.assertEqual(diam[(0, "a")], Q(1, 4))
        self.assertEqual(diam[(1, "a")], Q(0))

    def test_composition_sharp_and_coupling(self):
        self.assertEqual(composed_defect(Q(1, 3), Q(1, 5), Q(2)), Q(13, 15))
        self.assertEqual(tv_path_disagreement_bound(Q(1, 4), 0), 0)
        self.assertEqual(tv_path_disagreement_bound(Q(1, 4), 3), Q(37, 64))
        self.assertEqual(tv_return_bound(Q(1, 4), Q(0), 3),
                         Q(0) + Q(1, 4) + Q(7, 16))
        self.assertLessEqual(tv_return_bound(Q(1, 4), Q(0), 3), Q(3, 4))

    def test_tv_lp_three_dirac_center(self):
        p = [{i: Q(1)} for i in range(3)]
        radius_value, center = tv_minimax_center(p, (0, 1, 2))
        self.assertEqual(radius_value, Q(2, 3))
        self.assertEqual(center, {i: Q(1, 3) for i in range(3)})

    def test_tv_lp_two_bernoulli_rows(self):
        p = [{0: Q(3, 4), 1: Q(1, 4)}, {0: Q(1, 4), 1: Q(3, 4)}]
        radius_value, center = tv_minimax_center(p, (0, 1))
        self.assertEqual(radius_value, Q(1, 4))
        self.assertEqual(center, {0: Q(1, 2), 1: Q(1, 2)})

    def test_tv_lp_single_support(self):
        self.assertEqual(tv_minimax_center([{0: Q(1)}], (0,)),
                         (Q(0), {0: Q(1)}))

if __name__ == "__main__":
    unittest.main()
