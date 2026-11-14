"""Finite decision-menu and posterior-refinement certificates."""
from fractions import Fraction as F
import unittest
from decision_memory import (all_threshold_certificate, comparison, encoded_channel, report)
from reference import crossover, noise_signal, threshold_truth


class DecisionMemoryTests(unittest.TestCase):
    def test_bayes_bit_matches_full_state_on_all_three_bit_targets(self):
        for mask in range(256):
            f=tuple(1 if mask&(1<<x) else -1 for x in range(8))
            r=comparison(f,F(1,5),F(1,10))
            a=r['channels']['optimal_decision_bit']; b=r['channels']['full_state']
            self.assertEqual(a['optimal_decoding_error'],b['optimal_decoding_error'])
            self.assertLessEqual(a['states'],2)
            self.assertLessEqual(a['information_bits_numeric'],b['information_bits_numeric']+1e-11)

    def test_all_thresholds_separate_every_distinct_posterior(self):
        values=(F(-3,4),F(-1,4),F(0),F(1,3),F(3,4),F(0))
        r=all_threshold_certificate(values)
        self.assertEqual(r['minimum_code_states'],5)
        self.assertEqual(r['minimum_fixed_length_bits'],3)
        self.assertEqual(r['code'][2],r['code'][5])
        levels=list(map(F,r['posterior_levels']))
        for a,b,t in zip(levels,levels[1:],map(F,r['separating_thresholds'])):
            self.assertLess(a,t); self.assertLess(t,b)

    def test_posterior_encoder_preserves_binary_channel_information(self):
        values=(F(-1,2),F(1,4),F(-1,2),F(1,4),F(3,4))
        r=all_threshold_certificate(values)
        compressed=encoded_channel(values,r['code'])
        full=encoded_channel(values,tuple(range(len(values))))
        self.assertEqual(compressed['optimal_decoding_error'],full['optimal_decoding_error'])
        self.assertAlmostEqual(compressed['information_bits_numeric'],full['information_bits_numeric'])

    def test_majority_seven_requires_three_bits_for_all_thresholds(self):
        f=threshold_truth(7,{1<<j:1 for j in range(7)})
        r=comparison(f,crossover(F(1,20),4))
        self.assertEqual(r['threshold_menu']['minimum_code_states'],8)
        self.assertEqual(r['threshold_menu']['minimum_fixed_length_bits'],3)
        self.assertEqual(r['channels']['optimal_decision_bit']['fixed_length_bits'],1)
        self.assertLess(r['channels']['optimal_decision_bit']['information_bits_numeric'],
                        r['channels']['full_state']['information_bits_numeric'])

    def test_parity_affine_projection_loses_its_recoverable_signal(self):
        f=threshold_truth(7,{127:1})
        r=comparison(f,F(1,20))
        self.assertEqual(F(r['channels']['affine_surrogate_bit']['optimal_decoding_error']),F(1,2))
        self.assertLess(F(r['channels']['full_state']['optimal_decoding_error']),F(1,2))
        self.assertEqual(r['channels']['affine_surrogate_bit']['information_bits_numeric'],0)
        self.assertEqual(r['threshold_menu']['minimum_code_states'],2)

    def test_rare_alert_best_decision_can_be_constant_with_information_present(self):
        f=threshold_truth(7,{0:-6,**{1<<j:1 for j in range(7)}})
        r=comparison(f,crossover(F(1,20),4))
        self.assertEqual(r['channels']['optimal_decision_bit']['states'],1)
        self.assertGreater(r['channels']['full_state']['information_bits_numeric'],0)
        self.assertEqual(r['threshold_menu']['minimum_code_states'],8)

    def test_receipt_and_invalid_inputs(self):
        r=report()
        self.assertEqual(len(r['cases']),5)
        self.assertFalse(r['trained_checkpoint_evaluated'])
        for bad in ([],[F(2)]):
            with self.assertRaises(ValueError): all_threshold_certificate(bad)
        with self.assertRaises(ValueError): encoded_channel([F(0)],[0,1])
        with self.assertRaises(ValueError): comparison((1,-1),F(2,3))


if __name__=='__main__':unittest.main()
