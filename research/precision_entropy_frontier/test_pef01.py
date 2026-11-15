"""PEF-01 differential tests and causal readout audit."""
import itertools
from fractions import Fraction
import random
import unittest

from pef01 import (
    Case, encode_integer, exact_audit, minimum_full_recovery_precision,
    quantize_independent, quantize_integer, rational_orbit, retained_bits,
    suite,
)


class PEF01Tests(unittest.TestCase):
    def test_all_small_precision_fibres_with_independent_oracle(self):
        # Every contraction/precision/depth combination in this finite grid.
        for k in range(1, 5):
            for depth in range(0, 7):
                for bits in range(0, k*depth + 3):
                    case = Case(k, depth, bits)
                    result = exact_audit(case, check_interventions=False)
                    t = min(depth, (bits + k - 1)//k)
                    self.assertEqual(result["observed_distinct_quantizer_codes"], 2**t)
                    self.assertEqual(result["histories_per_observed_code"], 2**(depth-t))
                    self.assertEqual(result["conditional_history_entropy_bits"], depth-t)
                    self.assertTrue(result["independent_fraction_oracle_checked"])

    def test_depth_four_fibre_pattern_matches_lean(self):
        for bits, size in [(3, 4), (6, 2), (7, 1)]:
            self.assertEqual(exact_audit(Case(2, 4, bits))[
                "histories_per_observed_code"], size)
        self.assertEqual(exact_audit(Case(1, 4, 2))[
            "histories_per_observed_code"], 4)

    def test_original_input_bits_intervention_visibility(self):
        for k in (1, 2, 3):
            for bits in (0, 1, 3, 7):
                result = exact_audit(Case(k, 5, bits))
                self.assertEqual(result["exact_intervention_checks"], 5 * 2**5)
                self.assertEqual(result["readout_visible_input_positions_zero_based"],
                                 list(range(5-retained_bits(k,5,bits),5)))

    def test_threshold_is_sharp(self):
        for k in range(1, 7):
            for depth in range(1, 9):
                minimum = minimum_full_recovery_precision(k,depth)
                self.assertEqual(retained_bits(k,depth,minimum),depth)
                self.assertLess(retained_bits(k,depth,minimum-1),depth)
        self.assertEqual(minimum_full_recovery_precision(2,8),15)

    def test_actual_ratio_and_chronological_digits(self):
        rng = random.Random(140148)
        for _ in range(300):
            k=rng.randrange(1,7)
            history=tuple(rng.randrange(2) for _ in range(rng.randrange(13)))
            bits=rng.randrange(30)
            x=rational_orbit(history,k)
            self.assertEqual(x, Fraction(encode_integer(history,k),
                                         (1<<k)**len(history)))
            self.assertEqual(quantize_integer(history,k,bits),
                             quantize_independent(history,k,bits))

    def test_suite_summary_replays_information_frontier(self):
        results = suite()["cases"]
        self.assertEqual([(r["k"],r["depth"],r["precision_bits"],
                           r["recovered_most_recent_history_bits"])
                          for r in results],
                         [(1,8,8,8),(2,8,8,4),(2,8,15,8),
                          (3,8,8,3),(3,8,22,8),(2,12,8,4)])
        self.assertEqual(results[-1]["optimal_exact_history_decoding_probability"],
                         {"numerator":1,"denominator":256})
        self.assertEqual(results[1]["histories_per_observed_code"],16)

    def test_input_guards(self):
        for obj in (Case(0,2,2),Case(1,-1,0),Case(1,1,-1),
                    Case(1,13,1), Case(1,1,129)):
            with self.assertRaises(ValueError):
                exact_audit(obj)
        with self.assertRaises(ValueError):
            quantize_integer((2,),2,3)


if __name__ == "__main__":
    unittest.main()
