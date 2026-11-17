"""Independent finite checks for the latent morphology threshold model."""

import unittest
from fractions import Fraction as Q
from itertools import combinations

try:
    from territory import (
        Tissue, boundary, critical_field, energy_gap, enumerate_ground_states,
        experiment, influence_cone, mincut_ground_state, square_grid,
    )
except ImportError:
    from .territory import (
        Tissue, boundary, critical_field, energy_gap, enumerate_ground_states,
        experiment, influence_cone, mincut_ground_state, square_grid,
    )


def direct_energy(t: Tissue, mask: int, field: Q) -> Q:
    """Independent Ising Hamiltonian, with fixed -1 exterior spins."""
    spin = lambda i: 1 if (mask >> i) & 1 else -1
    return (
        -t.coupling * sum(spin(i) * spin(j) for i, j in t.edges)
        +t.coupling * sum(t.ground[i] * spin(i) for i in range(t.n))
        -field * sum(spin(i) for i in t.program)
        +t.anchor * sum(spin(i) for i in range(t.n) if i not in t.program)
    )


class TerritoryTests(unittest.TestCase):
    def setUp(self):
        self.intact = square_grid()
        self.cut = self.intact.cut([(1, 4), (3, 4)])
        self.sham = self.intact.cut([(0, 1), (7, 8)])

    def test_hamiltonian_identity_for_every_state(self):
        for t in (self.intact, self.cut, self.sham):
            for h in (Q(0), Q(2), Q(3), Q(4), Q(7, 2)):
                e0 = direct_energy(t, 0, h)
                for mask in range(1 << t.n):
                    self.assertEqual(energy_gap(t, mask, h), direct_energy(t, mask, h) - e0)

    def test_exact_critical_field(self):
        self.assertEqual(critical_field(self.intact), (Q(4), 16))
        self.assertEqual(critical_field(self.cut), (Q(2), 16))
        self.assertEqual(critical_field(self.sham), (Q(4), 16))
        self.assertIn(16, enumerate_ground_states(self.intact, Q(4))[1])
        self.assertIn(0, enumerate_ground_states(self.intact, Q(4))[1])

    def test_latent_field_and_wound_probe(self):
        for t in (self.intact, self.cut, self.sham):
            self.assertEqual(enumerate_ground_states(t, Q(0)), (Q(0), (0,)))
        self.assertEqual(enumerate_ground_states(self.intact, Q(3)), (Q(0), (0,)))
        self.assertEqual(enumerate_ground_states(self.cut, Q(3)), (Q(-2), (16,)))
        self.assertEqual(enumerate_ground_states(self.sham, Q(3)), (Q(0), (0,)))

    def test_influence_cones(self):
        self.assertEqual([len(influence_cone(self.intact, 4, k)) for k in range(4)], [1, 5, 9, 9])
        self.assertEqual([len(influence_cone(self.cut, 4, k)) for k in range(4)], [1, 3, 6, 8])
        self.assertEqual([len(influence_cone(self.sham, 4, k)) for k in range(4)], [1, 5, 9, 9])
        self.assertEqual(influence_cone(self.intact, 4, 0), frozenset({4}))

    def test_all_possible_center_edge_deletions_monotone(self):
        incident = [e for e in self.intact.edges if 4 in e]
        self.assertEqual(len(incident), 4)
        for k in range(5):
            for removed in combinations(incident, k):
                t = self.intact.cut(removed)
                self.assertEqual(critical_field(t)[0], Q(4-k))

    def test_independent_mincut_matches_enumeration(self):
        for t in (self.intact, self.cut, self.sham):
            for h in (Q(0), Q(1), Q(2), Q(3), Q(4), Q(5), Q(7, 2)):
                direct_gap, masks = enumerate_ground_states(t, h)
                cut_gap, mask, _ = mincut_ground_state(t, h)
                self.assertEqual(cut_gap, direct_gap)
                self.assertIn(mask, masks)

    def test_mincut_small_graph_exhaustive_family(self):
        potential_edges = [(0, 1), (0, 2), (1, 2)]
        for code in range(8):
            edges = tuple(e for j, e in enumerate(potential_edges) if (code >> j) & 1)
            for program in range(3):
                t = Tissue(3, edges, (1, 1, 1), frozenset({program}), Q(3, 2), Q(2, 3))
                for field in (Q(0), Q(1), Q(3, 2), Q(3)):
                    best, witnesses = enumerate_ground_states(t, field)
                    got, mask, _ = mincut_ground_state(t, field)
                    self.assertEqual(got, best)
                    self.assertIn(mask, witnesses)

    def test_larger_graph_mincut_without_exhaustive_search(self):
        t = square_grid(9)
        center = 40
        cut = t.cut([(center-9, center), (center-1, center)])
        self.assertEqual(mincut_ground_state(t, Q(3))[:2], (Q(0), 0))
        self.assertEqual(mincut_ground_state(cut, Q(3))[:2], (Q(-2), 1 << center))

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            self.intact.cut([(0, 8)])
        with self.assertRaises(ValueError):
            square_grid(4)
        with self.assertRaises(ValueError):
            critical_field(square_grid(5), max_vertices=20)
        with self.assertRaises(ValueError):
            influence_cone(self.intact, 4, -1)

    def test_receipt(self):
        receipt = experiment()
        self.assertEqual(receipt["results"]["two_center_edges_cut"]["ground_state_masks"], [16])
        self.assertEqual(receipt["results"]["sham_distant_edges_cut"]["critical_field"], "4")


if __name__ == "__main__":
    unittest.main()
