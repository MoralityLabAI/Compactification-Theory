"""Exact tests for OIC-T26, using independently enumerated finite phases."""
from __future__ import annotations
import unittest
from collections import Counter
from fractions import Fraction
from itertools import product
from math import gcd
from oic19_identifiability import (
    trace,distributions,first_n_pulse_probability,
    empirical_first_n_pulse_probability,brute_gauge_audit,
)


class ObservableGaugeTests(unittest.TestCase):
    def test_pulse_probability_exact(self):
        for n in range(2,30):
            for d in range(-2*n,2*n+1):
                self.assertEqual(first_n_pulse_probability(n,d),Fraction(1,gcd(n,d)))
                self.assertEqual(empirical_first_n_pulse_probability(n,d),Fraction(1,gcd(n,d)))

    def test_exact_trace_law_for_same_gcd(self):
        for n in range(2,13):
            checked=brute_gauge_audit(n,horizons=(0,1,2,3,4))
            self.assertGreater(checked['checked_equivalent_shift_pairs'],0)
            self.assertEqual(checked['counterexamples'],[])

    def test_all_units_observationally_equivalent(self):
        for n in (5,8,9,12,16):
            action=(1,0,1,1,0,1,1,0,1,1,1,1,0)
            common=distributions(n,1,action)
            for d in range(n):
                if gcd(n,d)==1:
                    self.assertEqual(distributions(n,d,action),common)

    def test_different_gcd_distinguishable_by_n_active_probes(self):
        for n in range(2,25):
            distributions_by_g={}
            for d in range(n):
                g=gcd(n,d)
                current=distributions(n,d,(1,)*n)
                if g in distributions_by_g:self.assertEqual(current,distributions_by_g[g])
                distributions_by_g[g]=current
            for a,b in product(distributions_by_g,repeat=2):
                if a!=b:self.assertNotEqual(distributions_by_g[a],distributions_by_g[b])

    def test_specific_n5_shift_ambiguity(self):
        self.assertEqual(distributions(5,1,(1,)*12),distributions(5,2,(1,)*12))
        self.assertEqual(distributions(5,2,(1,)*12),distributions(5,3,(1,)*12))
        self.assertNotEqual(distributions(6,1,(1,)*12),distributions(6,2,(1,)*12))

    def test_input_validation(self):
        for args in ((1,1,0,(1,)),(3,1,3,(1,)),(3,0,0,(2,))):
            with self.assertRaises(ValueError):trace(*args)


if __name__=='__main__':unittest.main()
