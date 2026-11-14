"""SCT-01: complete finite SAT, expectation threshold, stopping-time and state tests."""
from __future__ import annotations
import itertools
import math
import random
import unittest
from fractions import Fraction as F

from reference import (
    K, R, M, ALL_CLAUSES, gadget_full_mask, core_masks, unsat,
    clause_false_assignment_bits, exhaustive_sat_oracle, brute_sat,
    bernoulli_unsat, histogram_probability, optimal_cover_cost,
    cover_threshold_float, median_threshold_float, complete_group_by_time,
    hit_cdf, stirling_second, hit_cdf_by_histogram, hitting_moments,
    hitting_moments_independent_dp_float, one_gadget_coupon_moments,
    minimal_monitor_states, monitor_bits, encode_monitor, step_monitor,
    separating_suffix, promised_unique_witness_success_ceiling, source_report,
)


class SATThresholdControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exhaustive = exhaustive_sat_oracle()

    def test_each_complete_three_clause_gadget_is_minimally_unsat(self):
        self.assertEqual(core_masks(),(255,65280))
        for core in core_masks():
            self.assertTrue(unsat(core))
            for i in range(M):
                if core & (1<<i):
                    self.assertFalse(unsat(core^(1<<i)))
                    self.assertTrue(brute_sat(core^(1<<i)))
            self.assertFalse(brute_sat(core))

    def test_independent_assignment_bitset_covers_all_65536_formulas(self):
        x=self.exhaustive
        self.assertEqual(x["all_formula_subsets"],65536)
        self.assertEqual(x["unsatisfiable_subsets"],511)
        self.assertEqual(x["satisfiable_subsets"],65025)
        self.assertEqual(x["minimal_unsat_cores"],[255,65280])
        self.assertEqual(x["unsat_size_histogram"][:8],[0]*8)
        self.assertEqual(x["unsat_size_histogram"][8],2)
        self.assertEqual(x["unsat_size_histogram"][16],1)

    def test_direct_assignment_search_matches_analytic_model(self):
        rng=random.Random(175235)
        cases=[0,ALL_CLAUSES,255,65280,254,65279,ALL_CLAUSES^255]
        cases.extend(rng.randrange(1<<M) for _ in range(2000))
        for mask in cases:
            self.assertEqual(not brute_sat(mask),unsat(mask))

    def test_each_clause_excludes_one_local_assignment_and_eight_global(self):
        for g in range(R):
            for pattern in range(8):
                forbidden=clause_false_assignment_bits(g,pattern)
                self.assertEqual(forbidden.bit_count(),8)
                for x in range(64):
                    self.assertEqual(bool(forbidden & (1<<x)),
                                     ((x>>(3*g))&7)==pattern)

    def test_exact_cover_and_bernoulli_reliability_polynomials(self):
        hist=self.exhaustive["unsat_size_histogram"]
        for p in (F(0),F(1,20),F(1,4),F(1,2),
                  F(3,4),F(4,5),F(9,10),F(1)):
            self.assertEqual(histogram_probability(hist,p),bernoulli_unsat(p))
            self.assertEqual(bernoulli_unsat(p),2*p**8-p**16)
            self.assertEqual(optimal_cover_cost(p),min(F(1),2*p**8))
        self.assertEqual(bernoulli_unsat(F(1,2)),F(511,65536))
        self.assertEqual(optimal_cover_cost(F(1,2)),F(1,128))

    def test_fractional_cover_lower_certificate_with_empty_subset(self):
        # The analytic proof first tests the two minimal UNSAT members.
        # Nonempty covering subsets of distinct cores cannot overlap.
        for p in (F(1,5),F(1,2),F(4,5),F(9,10),F(1)):
            optimum=optimal_cover_cost(p)
            for empty_weight in (F(0),F(1,4),F(1,2),F(3,4),F(1)):
                lower=empty_weight+2*(1-empty_weight)*p**8
                self.assertGreaterEqual(lower,optimum)
            self.assertEqual(optimum,min(F(1),2*p**8))

    def test_cover_and_median_threshold_are_distinct(self):
        q=cover_threshold_float()
        p=median_threshold_float()
        self.assertAlmostEqual(q,2**(-1/4),places=12)
        self.assertLess(q,p)
        self.assertAlmostEqual(2*p**8-p**16,0.5,places=12)
        self.assertAlmostEqual(2*q**8,0.5,places=12)

    def test_two_independent_coupon_CDF_calculations_agree(self):
        hist=tuple(self.exhaustive["unsat_size_histogram"])
        for t in (0,1,7,8,9,12,16,24,30,31,32,40,48,64):
            self.assertEqual(hit_cdf(t),hit_cdf_by_histogram(t,hist))
            self.assertTrue(0<=hit_cdf(t)<=1)
        self.assertEqual(hit_cdf(7),0)
        self.assertEqual(hit_cdf(8),F(315,16777216))
        self.assertLess(hit_cdf(30),F(1,2))
        self.assertGreaterEqual(hit_cdf(31),F(1,2))

    def test_stirling_surjection_counts_small_t(self):
        self.assertEqual(stirling_second(0,4)[:5],(1,0,0,0,0))
        self.assertEqual(stirling_second(4,4)[:5],(0,1,7,6,1))
        self.assertEqual(stirling_second(8,8)[8],1)

    def test_exact_hitting_time_moments_independent_markov_oracle(self):
        a=hitting_moments()
        b=hitting_moments_independent_dp_float()
        self.assertEqual(a["mean"],"1481069/45045")
        self.assertEqual(a["variance"],"80211997256/676350675")
        self.assertAlmostEqual(a["mean_float"],b["mean"],places=9)
        self.assertAlmostEqual(a["variance_float"],b["variance"],places=7)
        self.assertEqual(b["state_count"],65536)

    def test_full_uniform_n3_random_sat_is_8_coupon_collector(self):
        x=one_gadget_coupon_moments()
        self.assertEqual(x["mean"],"761/35")
        self.assertEqual(x["variance"],"838034/11025")

    def test_exact_online_monitor_states_and_bit_budget(self):
        self.assertEqual(minimal_monitor_states(),65026)
        self.assertEqual(monitor_bits(),16)
        self.assertEqual(minimal_monitor_states(1,8),256)
        self.assertEqual(monitor_bits(1,8),8)
        self.assertEqual(monitor_bits(3,8),24)
        self.assertEqual(len({encode_monitor(mask) for mask in range(1<<M)}),
                         65026)
        self.assertEqual(encode_monitor(255),65025)
        self.assertEqual(encode_monitor(65280),65025)

    def test_online_automaton_matches_every_clause_transition(self):
        for mask in range(1<<M):
            state=encode_monitor(mask)
            for clause in range(M):
                self.assertEqual(step_monitor(state,clause),
                                 encode_monitor(mask|(1<<clause)))

    def test_arbitrary_repetitions_and_order_do_not_change_decision(self):
        rng=random.Random(235175)
        for _ in range(200):
            sequence=[rng.randrange(M) for j in range(rng.randrange(1,120))]
            mask=0
            state=0
            for clause in sequence:
                mask|=1<<clause
                state=step_monitor(state,clause)
                self.assertEqual(state,encode_monitor(mask))
            self.assertEqual(state==65025,unsat(mask))

    def test_exhaustive_small_Myhill_Nerode_distinguishing_suffix(self):
        r,k=2,3
        losing=[mask for mask in range(1<<(r*k)) if not unsat(mask,r,k)]
        self.assertEqual(len(losing),49)
        for i,a in enumerate(losing):
            for b in losing[i+1:]:
                proof=separating_suffix(a,b,r,k)
                self.assertTrue(proof["A_unsat_after_suffix"])
                self.assertTrue(proof["B_sat_after_suffix"])
        rng=random.Random(1)
        for _ in range(1000):
            a=rng.randrange(1<<16)
            b=rng.randrange(1<<16)
            if a!=b and not unsat(a) and not unsat(b):
                separating_suffix(a,b)

    def test_unique_core_promise_witness_information_lower_bound(self):
        for r in (2,3,4,8,16):
            for bits in (0,1,2,3,4):
                rate=promised_unique_witness_success_ceiling(r,bits)
                self.assertEqual(rate,min(F(1),F(1<<bits,r)))
            self.assertEqual(promised_unique_witness_success_ceiling(r,0),
                             F(1,r))
        self.assertEqual(promised_unique_witness_success_ceiling(8,2),F(1,2))
        self.assertEqual(promised_unique_witness_success_ceiling(8,3),1)

    def test_report_source_scope_and_independent_receipts(self):
        report=source_report()
        self.assertEqual(report["source_families"],[175,235])
        self.assertEqual(report["status"],"EXACT_FINITE_SAT_CORE_CONTROL")
        self.assertEqual(report["with_replacement_first_UNSAT_median_draw"],31)
        self.assertEqual(report["streaming_exact_min_bits"],16)
        self.assertFalse(report["general_source_175_constant_gap_proved_here"])
        self.assertFalse(report["general_source_235_limiting_density_proved_here"])
        self.assertFalse(report["trained_reasoner_evaluated"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError): unsat(-1)
        with self.assertRaises(ValueError): unsat(1<<M)
        with self.assertRaises(ValueError): gadget_full_mask(-1)
        with self.assertRaises(ValueError): hit_cdf(-1)
        with self.assertRaises(ValueError): optimal_cover_cost(F(3,2))
        with self.assertRaises(ValueError): step_monitor(-1,0)
        with self.assertRaises(ValueError): step_monitor(0,M)
        with self.assertRaises(ValueError): separating_suffix(0,0)
        with self.assertRaises(ValueError): promised_unique_witness_success_ceiling(0,3)


if __name__=="__main__":
    unittest.main()
