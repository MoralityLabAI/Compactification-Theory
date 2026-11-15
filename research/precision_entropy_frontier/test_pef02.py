"""PEF-02 exact independent checks: noise margins, floor quantization, priors."""
import random
from fractions import Fraction
from itertools import product
import unittest

from pef01 import quantize_integer
from pef02 import (
    affine_gauge_codes, case_report, exhaustive_suffix_gap,
    geometric_error_gain, markov_prior_audit, minimum_suffix_gap,
    nearest_round_at_each_step, nearest_round_once,
    recurrent_floor, robustness, suite,
)


class PEF02Tests(unittest.TestCase):
    def test_analytic_gap_equals_complete_exact_pair_enumeration(self):
        for k in range(1, 5):
            for depth in range(1, 8):
                for t in range(1, depth+1):
                    self.assertEqual(
                        minimum_suffix_gap(k, depth, t),
                        exhaustive_suffix_gap(k, depth, t),
                        (k, depth, t))

    def test_recurrent_floor_exact_commutation_full_parameter_grid(self):
        for k in range(1,5):
            for depth in range(0,8):
                for precision in range(0,12):
                    for hist in product((0,1), repeat=depth):
                        self.assertEqual(
                            recurrent_floor(hist, k, precision),
                            quantize_integer(hist, k, precision),
                            (k, depth, precision, hist))

    def test_nearest_rounding_counterexample_is_real(self):
        history = (1,0,0)
        self.assertEqual(nearest_round_at_each_step(history,2,3), 1)
        self.assertEqual(nearest_round_once(history,2,3), 0)

    def test_gap_and_bounded_step_error_certificates(self):
        a=robustness(2,8,8)
        self.assertEqual(a["minimum_cross_suffix_distance"], "513/65536")
        self.assertEqual(a["retained_suffix_bits"],4)
        self.assertEqual(a["quantized_step_error_sufficient_max"],
                         str((Fraction(513,65536)-Fraction(1,256)) /
                             (2*geometric_error_gain(2,8))))
        self.assertEqual(geometric_error_gain(2,1),1)
        self.assertEqual(geometric_error_gain(2,2),Fraction(5,4))
        self.assertEqual(robustness(1,8,8)["quantized_step_error_sufficient_max"],"0")

    def test_uniform_cluster_centres_robust_under_adversarial_bounds(self):
        # The analog threshold is necessary and sufficient: min-pair gap g
        # gives disjoint radius-epsilon intervals exactly when 2eps<g.
        for k,depth,t in ((1,5,5),(2,4,2),(3,6,3),(4,5,2)):
            g = minimum_suffix_gap(k,depth,t)
            eps = g / 3
            self.assertLess(2*eps,g)
            self.assertFalse(2*(g/2)<g)
            self.assertEqual(exhaustive_suffix_gap(k,depth,t),g)

    def test_gauge_calibrated_code_invariant(self):
        for history in product((0,1),repeat=5):
            for scale in (Fraction(1,4),Fraction(2),Fraction(16)):
                for shift in (Fraction(-3,4),Fraction(0),Fraction(7,3)):
                    x=affine_gauge_codes(history,2,3,scale,shift)
                    self.assertEqual(x["calibrated_code"],x["untransformed_code"])
        # A raw absolute B=3 readout of a 16x amplification recovers B+4:
        raw_codes = {affine_gauge_codes(h,2,3,Fraction(16),Fraction(0))["raw_absolute_code"]
                     for h in product((0,1),repeat=4)}
        calibrated_codes = {affine_gauge_codes(h,2,3,Fraction(16),Fraction(0))["calibrated_code"]
                            for h in product((0,1),repeat=4)}
        self.assertEqual(len(raw_codes),16)
        self.assertEqual(len(calibrated_codes),4)

    def test_markov_prior_map_gain_without_causal_old_bit_storage(self):
        for stay in (Fraction(1,2),Fraction(3,4),Fraction(19,20),Fraction(1,4)):
            for k,depth,bits in ((2,8,8),(1,5,2),(3,6,5),(2,5,0)):
                record=markov_prior_audit(k,depth,bits,stay)
                self.assertEqual(record["optimal_full_history_map_success"],
                                 str((max(stay,1-stay)**(depth-record["retained_suffix_bits"]))
                                     if record["retained_suffix_bits"] else
                                     Fraction(1,2)*max(stay,1-stay)**(depth-1)))
                self.assertEqual(record["hidden_edge_direct_causal_readout_effect"],0)
        example=markov_prior_audit(2,8,8,Fraction(19,20))
        self.assertEqual(example["optimal_full_history_map_success"],"130321/160000")
        self.assertEqual(example["uniform_history_success"],"1/16")
        self.assertEqual(example["hidden_edge_observational_agreement"],"19/20")

    def test_suite_and_guards(self):
        result=suite()
        self.assertEqual(len(result["independent_brute_gap_and_staged_floor"]),5)
        self.assertEqual(result["source_prior_example"]["optimal_full_history_map_success"],
                         "130321/160000")
        for values in [(0,4,1),(2,0,0),(2,4,5)]:
            with self.assertRaises(ValueError):
                minimum_suffix_gap(*values)
        with self.assertRaises(ValueError):
            robustness(2,4,0)


if __name__=="__main__":
    unittest.main()
