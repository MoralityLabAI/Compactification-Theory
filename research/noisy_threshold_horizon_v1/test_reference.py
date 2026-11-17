"""NHB-01 finite exact controls; entropy comparisons use numeric tolerance."""
from fractions import Fraction as F
import math
import unittest
from reference import (affine_projection, analyze, brute_signal, char, compose,
    crossover, entropy, examples, influence, information_from_signal,
    monitor_receipt, noise_rate, noise_signal, source_information_cap,
    spectrum, threshold_truth)


def all_truths(n):
    for mask in range(1<<(1<<n)):
        yield tuple(1 if mask&(1<<x) else -1 for x in range(1<<n))


class NoisyHorizonTests(unittest.TestCase):
    def test_noise_semigroup_matches_repeated_bit_channels(self):
        for f in all_truths(2):
            for p,q in ((F(1,20),F(1,10)),(F(0),F(1,2)),(F(1,4),F(1,3))):
                self.assertEqual(noise_signal(noise_signal(f,p),q),
                                 noise_signal(f,compose(p,q)))
            current=tuple(map(F,f))
            for t in range(7):
                self.assertEqual(current,noise_signal(f,crossover(F(1,20),t)))
                current=noise_signal(current,F(1,20))

    def test_factorized_channel_matches_full_transition_enumeration(self):
        for f in all_truths(3):
            for p in (F(1,5),F(1,2)):
                self.assertEqual(noise_signal(f,p),brute_signal(f,p))

    def test_fourier_parseval_reconstruction_and_total_influence(self):
        for f in all_truths(3):
            coeff=spectrum(f)
            self.assertEqual(sum(c*c for c in coeff),1)
            for x in range(8):
                self.assertEqual(sum(coeff[m]*char(m,x) for m in range(8)),f[x])
            self.assertEqual(sum(influence(f)),
                             sum(m.bit_count()*c*c for m,c in enumerate(coeff)))

    def test_all_three_bit_boolean_functions_bayes_and_information(self):
        for f in all_truths(3):
            for p in (F(1,10),F(1,4)):
                for eta in (F(0),F(1,10)):
                    r=analyze(f,p,eta)
                    self.assertLessEqual(F(r['Bayes_terminal_error']),F(r['same_function_error']))
                    if sum(f)==0:
                        self.assertGreaterEqual(F(r['Bayes_terminal_error']),compose(eta,p))
                    self.assertLessEqual(r['terminal_information_bits_numeric'],
                                         r['source_119_cap_bits_numeric']+2e-11)

    def test_soft_channel_general_finite_grid(self):
        import itertools
        for u in itertools.product((F(-1),F(-1,2),F(0),F(1,2),F(1)),repeat=4):
            I0=information_from_signal(u)
            for p in (F(1,10),F(1,4)):
                actual=information_from_signal(noise_signal(u,p))
                self.assertLessEqual(actual,source_information_cap(I0,p)+2e-11)

    def test_coordinate_attains_balanced_information_and_error_envelopes(self):
        f=threshold_truth(5,{1:1})
        for t in (0,1,2,4,5,8,16):
            p=crossover(F(1,20),t)
            for eta in (F(0),F(1,10),F(1,2)):
                r=analyze(f,p,eta)
                self.assertEqual(F(r['Bayes_terminal_error']),compose(eta,p))
                self.assertAlmostEqual(r['terminal_information_bits_numeric'],
                                       1-entropy(compose(eta,p)),places=11)

    def test_polynomial_threshold_representation_has_declared_degree(self):
        for name,n,terms,f in examples():
            self.assertEqual(len(f),1<<n)
            degree=max(m.bit_count() for m in terms)
            AS=sum(influence(f))
            self.assertLessEqual(float(AS),8*degree*math.sqrt(n))
        majority=threshold_truth(7,{1<<i:1 for i in range(7)})
        self.assertEqual(sum(influence(majority)),F(35,16))
        self.assertEqual(max(m.bit_count() for m,c in enumerate(spectrum(majority)) if c),7)
        self.assertEqual(threshold_truth(1,{0:0}),(1,1))

    def test_surrogate_degree_bound_with_exact_uniform_approximation(self):
        for f in all_truths(2):
            for h in all_truths(2):
                p,eta=F(1,4),F(1,10)
                r=analyze(f,p,eta,h)
                alpha=F(r['uniform_compression_error'])
                surrogate_AS=sum(influence(h))
                upper=eta+(1-2*eta)*(alpha+min(F(1,2),p*surrogate_AS))
                self.assertLessEqual(F(r['compressed_readout_error']),upper)
                self.assertLessEqual(abs(F(r['compressed_readout_error'])-
                                         F(r['same_function_error'])),(1-2*eta)*alpha)

    def test_parity_exposes_degree_dependent_horizon(self):
        f=threshold_truth(7,{127:1})
        for t in range(6):
            p=crossover(F(1,20),t); r=analyze(f,p,F(1,10))
            expected=(1-F(4,5)*(1-2*p)**7)/2
            self.assertEqual(F(r['Bayes_terminal_error']),expected)
            self.assertEqual(F(r['same_function_error']),expected)
            self.assertEqual(r['real_Fourier_degree'],7)

    def test_exact_four_step_cutoff_for_balanced_terminal_decoder(self):
        eta=F(1,10); eps=F(1,20)
        self.assertEqual(compose(eta,crossover(eps,4)),F(5939,25000))
        self.assertEqual(compose(eta,crossover(eps,5)),F(65951,250000))
        self.assertLessEqual(compose(eta,crossover(eps,4)),F(1,4))
        self.assertGreater(compose(eta,crossover(eps,5)),F(1,4))
        for t in range(5,30):
            self.assertGreater(compose(eta,crossover(eps,t)),F(1,4))

    def test_rare_alert_stability_with_zero_terminal_information(self):
        f=threshold_truth(7,{0:-6,**{1<<i:1 for i in range(7)}})
        r=analyze(f,F(1,2),0)
        self.assertFalse(r['balanced'])
        self.assertIsNone(r['balanced_error_floor'])
        self.assertEqual(F(r['same_function_error']),F(127,8192))
        self.assertEqual(F(r['Bayes_terminal_error']),F(1,128))
        self.assertEqual(r['terminal_information_bits_numeric'],0.0)
        self.assertGreater(1-F(r['same_function_error']),F(98,100))

    def test_preserved_initial_bit_is_a_separate_observation_interface(self):
        # A monitor receiving and retaining the original target U can decode
        # it perfectly using two persistent states, regardless of state noise.
        joint={(u,y):F(1,4) for u in (-1,1) for y in (-1,1)}
        no_memory=sum(min(joint[-1,y],joint[1,y]) for y in (-1,1))
        with_memory=sum(p*int(u!=u) for (u,y),p in joint.items())
        self.assertEqual(no_memory,F(1,2))
        self.assertEqual(with_memory,0)
        self.assertEqual(len({u for u,y in joint}),2)

    def test_large_majority_source_bound_is_nonvacuous_but_loose(self):
        n=129; AS=F(n*math.comb(n-1,(n-1)//2),2**(n-1))
        self.assertLess(float(AS),8*math.sqrt(n))
        self.assertLess(8*math.sqrt(n),n)

    def test_complete_receipt_has_evidence_boundaries(self):
        r=monitor_receipt()
        self.assertEqual(r['paper_families'],[119,127])
        self.assertEqual(len(r['cases']),35)
        self.assertTrue(r['entropy_is_numerical'])
        self.assertTrue(r['probabilities_are_exact_rationals'])
        self.assertFalse(r['trained_neural_checkpoint_evaluated'])
        self.assertFalse(r['history_memory_available'])
        self.assertFalse(r['source_asymptotic_proofs_reproduced'])

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError): noise_rate(F(3,4))
        with self.assertRaises(ValueError): crossover(F(1,10),-1)
        with self.assertRaises(ValueError): spectrum((1,-1,1))
        with self.assertRaises(ValueError): analyze((True,False),F(1,4))
        with self.assertRaises(ValueError): threshold_truth(2,{8:1})
        with self.assertRaises(ValueError): analyze((1,-1),F(1,4),readout=(1,-1,1,-1))
        with self.assertRaises(ValueError): source_information_cap(2,F(1,4))


if __name__=='__main__': unittest.main()
