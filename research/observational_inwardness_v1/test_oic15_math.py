"""Pure standard-library proof witness and fault-distance tests."""
import math
import random
import unittest

from oic15_math import (raw_codebook,codebook_min_distance,codebook_min_error_correct,
    raw_filter_sufficient_beta,first_soft_collapse_n,rotating_pulse_likelihoods,
    rotating_pulse_classified,residual_phase_entropy,finite_exact_bayes_accuracy)


class OIC15MathTests(unittest.TestCase):
    def test_passive_trace_equivalence(self):
        for n in range(2,13):
            word=(0,) * (3*n)
            a=raw_codebook(n,0,word)
            b=raw_codebook(n,1,word)
            self.assertEqual(a,b)
            self.assertEqual(len(set(a)),1)

    def test_n_active_probes_identify_family(self):
        for n in range(2,13):
            word=(1,)*n
            self.assertTrue(codebook_min_error_correct(n,word,0,0))
            self.assertEqual(codebook_min_distance(n,word),1)

    def test_fixed_nonzero_phase_indistinguishable(self):
        rng=random.Random(1948)
        for n in range(3,12):
            for T in (n,n*2,n*4):
                word=tuple(rng.randrange(2) for _ in range(T))
                fixed=raw_codebook(n,0,word)
                self.assertEqual(len(set(fixed[1:])),1)
            self.assertAlmostEqual(residual_phase_entropy(n),math.log2(n-1))

    def test_raw_even_distance_and_known_fault_bounds(self):
        for n in (2,4,6,8):
            for e,r in ((0,0),(1,0),(0,1),(1,1),(2,0)):
                t=n+2*e+r
                word=(1,)*n+(0,)*(t-n)
                self.assertEqual(codebook_min_distance(n,word),2*e+r+1)
                self.assertTrue(codebook_min_error_correct(n,word,e,r))
                smaller=(1,)*(n-1) if t==n else (1,)*n+(0,)*(t-n-1)
                self.assertFalse(codebook_min_error_correct(n,smaller,e,r))

    def test_raw_likelihood_margin_condition(self):
        for n in (4,6,8):
            for e,r in ((1,0),(0,1),(1,1)):
                d=2*e+r+1
                self.assertAlmostEqual(raw_filter_sufficient_beta(n,d,e,r),math.log(n))
                self.assertTrue(5.0>raw_filter_sufficient_beta(n,d,e,r))
        with self.assertRaises(ValueError):
            raw_filter_sufficient_beta(4,2,1,0)

    def test_soft_audit_entropy_dilution(self):
        for beta in (.75,1.4,3.0,5.0,6.4):
            n=first_soft_collapse_n(beta)
            self.assertGreaterEqual(n,4)
            self.assertTrue(rotating_pulse_classified(n-1,beta))
            self.assertFalse(rotating_pulse_classified(n,beta))
            for check_n in (3,4,5,8):
                a,b=rotating_pulse_likelihoods(check_n,beta)
                self.assertGreater(a,0)
                self.assertGreater(b,0)

    def test_bayes_accuracy_curve(self):
        for n in range(3,14):
            self.assertEqual(finite_exact_bayes_accuracy(n,0),.5)
            self.assertEqual(finite_exact_bayes_accuracy(n,1),.5)
            self.assertEqual(finite_exact_bayes_accuracy(n,n),1)
            self.assertEqual(finite_exact_bayes_accuracy(n,n+1),1)
            for t in range(2,n):
                self.assertAlmostEqual(finite_exact_bayes_accuracy(n,t),1-(n-t)/(2*n))


if __name__=='__main__':unittest.main()
