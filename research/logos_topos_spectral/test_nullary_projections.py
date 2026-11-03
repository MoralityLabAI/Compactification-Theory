"""Exhaustive exact-rational checks of finite LTR logical Dirac observables."""
import unittest
from fractions import Fraction as Q
from itertools import combinations, product

from filtered import MarkedSpace, block_commutator
from nullary_projections import (
    TruthLocus,
    sentence_projection,
    dirac_sentence_commutator,
    dirac_sentence_seminorm,
    refutation_norm_gap,
)


class NullaryProjectionTests(unittest.TestCase):
    def test_every_three_bit_truth_table_has_01_norm_gap(self):
        state_count = 8
        for mask in range(1 << state_count):
            locus = TruthLocus(tuple(bool(mask & (1 << i)) for i in range(state_count)))
            projected = locus.projection()
            self.assertTrue(all(p in (Q(0), Q(1)) for p in projected))
            self.assertTrue(all(p * p == p for p in projected))
            self.assertEqual(locus.norm(), Q(1) if locus.satisfiable() else Q(0))
            norm, entails = refutation_norm_gap(locus)
            self.assertEqual(norm, Q(0) if entails else Q(1))

    def test_two_point_dirac_commutator_detects_every_labelled_split(self):
        space = MarkedSpace(tuple(product((0, 1), repeat=3)))
        zero = ((Q(0), Q(0)), (Q(0), Q(0)))
        pairs = tuple(combinations(range(len(space.points)), 2))
        for mask in range(1 << len(space.points)):
            locus = TruthLocus(tuple(bool(mask & (1 << i))
                                     for i in range(len(space.points))))
            for i, j in pairs:
                comm = dirac_sentence_commutator(space, locus, i, j, Q(4))
                self.assertEqual(comm == zero, locus.values[i] == locus.values[j])
                self.assertEqual(comm[0][1], -comm[1][0])
                self.assertEqual(abs(comm[0][1]),
                                 abs(Q(locus.values[i]) - Q(locus.values[j]))
                                 / space.dist(i, j))

    def test_mass_changes_spectrum_without_changing_logic_commutator(self):
        space = MarkedSpace(((0,), (1,)))
        locus = TruthLocus((False, True))
        self.assertEqual(dirac_sentence_commutator(space, locus, 0, 1, Q(0)),
                         dirac_sentence_commutator(space, locus, 0, 1, Q(65536)))

    def test_concrete_boolean_formulas(self):
        space = MarkedSpace(tuple(product((0, 1), repeat=4)))
        p0 = sentence_projection(space, lambda x: x[0])
        p1 = sentence_projection(space, lambda x: x[1])
        tautology = sentence_projection(space, lambda x: bool(x[0] or not x[0]))
        contradiction = sentence_projection(space, lambda x: bool(x[0] and not x[0]))
        self.assertEqual(p0.norm(), Q(1))
        self.assertEqual(p1.norm(), Q(1))
        self.assertEqual(tautology.refutation().norm(), Q(0))
        self.assertEqual(contradiction.norm(), Q(0))
        self.assertEqual(contradiction.refutation().norm(), Q(1))

    def test_invalid_truth_data(self):
        for values in ((), (True, 1), (0, False)):
            with self.assertRaises(ValueError):
                TruthLocus(values)
        s = MarkedSpace(((0,), (1,)))
        locus = TruthLocus((False,))
        with self.assertRaises(ValueError):
            dirac_sentence_commutator(s, locus, 0, 1)
        with self.assertRaises(ValueError):
            dirac_sentence_commutator(s, TruthLocus((False, True)), 0, 0)

    def test_exact_ratio_of_atomic_truth_observable(self):
        space = MarkedSpace(tuple(product((0, 1), repeat=5)))
        for n in range(5):
            locus = sentence_projection(space, lambda x, n=n: x[n])
            seminorm = max(
                abs(locus.projection()[i] - locus.projection()[j]) / space.dist(i, j)
                for i, j in combinations(range(len(space.points)), 2)
            )
            self.assertEqual(seminorm, Q(2**n))
            self.assertEqual(dirac_sentence_seminorm(space, locus, Q(8)), seminorm)
            self.assertEqual(dirac_sentence_seminorm(space, locus, Q(0)), seminorm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
