"""Exact exhaustive shear quotient, code nesting and risk curve tests."""
import unittest
from fractions import Fraction

from cyclic_addition import states
from shear_exact import (after_shear, balanced_code, balanced_curve, exact_bits,
                         oracle_code, oracle_curve, present, risks, shear)
from task_closure import FiniteSystem, refinement_history


class ShearExactTests(unittest.TestCase):
    def test_outputs_reconstruct_microstate(self):
        for p in (5, 31):
            observed = set()
            for a, b in states(p):
                s, t = present(a, b, p), after_shear(a, b, p)
                self.assertEqual(((t-s) % p, (2*s-t) % p), (a, b))
                self.assertEqual(present(*shear(a, b, p), p), t)
                observed.add((s, t))
            self.assertEqual(len(observed), p*p)
        self.assertEqual(exact_bits(31), (5, 10))

    def test_shear_task_partition_refines_to_singletons_in_one_round(self):
        p = 5
        xs = states(p)
        lookup = {x: i for i, x in enumerate(xs)}
        system = FiniteSystem((tuple(lookup[shear(*x, p)] for x in xs),),
                              tuple(present(*x, p) for x in xs))
        history = refinement_history(system)
        self.assertEqual([len(set(q)) for q in history], [p, p*p])

    def test_exact_curve_and_nested_codes(self):
        p = 31
        curve = oracle_curve(p)
        self.assertEqual([(r['present_risk'], r['after_shear_risk']) for r in (curve[0], curve[5], curve[10])],
                         [('30/31','30/31'), ('0','30/31'), ('0','0')])
        self.assertEqual([r['after_shear_risk'] for r in curve[5:]],
                         ['30/31','29/31','27/31','23/31','15/31','0'])
        xs = states(p)
        for b in range(10):
            coarse = [oracle_code(*x, b, p) for x in xs]
            fine = [oracle_code(*x, b+1, p) for x in xs]
            map_to_coarse = {}
            for z, c in zip(fine, coarse):
                if z in map_to_coarse:
                    self.assertEqual(map_to_coarse[z], c)
                map_to_coarse[z] = c
        self.assertEqual(risks([oracle_code(*x, 10, p) for x in xs], p),
                         (Fraction(0), Fraction(0)))
        balanced = balanced_curve(p)
        self.assertEqual((balanced[5]['present_risk'], balanced[5]['after_shear_risk']),
                         ('23/31','27/31'))
        self.assertEqual((balanced[10]['present_risk'], balanced[10]['after_shear_risk']),
                         ('0','0'))
        for b in range(10):
            coarse = [balanced_code(*x,b,p) for x in xs]
            fine = [balanced_code(*x,b+1,p) for x in xs]
            mapping = {}
            for z,c in zip(fine,coarse):
                self.assertEqual(mapping.setdefault(z,c), c)


if __name__ == '__main__':
    unittest.main()
