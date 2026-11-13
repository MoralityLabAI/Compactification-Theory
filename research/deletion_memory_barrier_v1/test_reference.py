"""DMB-01 exact-rational regression suite; Python standard library only."""
from __future__ import annotations
import itertools
import math
import unittest
from fractions import Fraction as Q
from reference import (
    words, deletion_law, deletion_deck, deck_equal_through,
    total_variation, retained_length_tail, product_law, hard_pair,
    optimal_terminal_success, streaming_state_laws,
    optimal_full_trace_success, terminal_state_ceiling,
    hard_pair_success_ceiling, universal_information_ceiling,
    gaussian_cap_upper, gaussian_terminal_ceiling,
    binomial_tail, thinning_markov_cap, structured_two_word_success, control_report,
)


class DeletionReferenceTests(unittest.TestCase):
    def test_laws_normalize_for_all_binary_words(self):
        for n in range(1,6):
            for x in words(n):
                for q in (Q(0),Q(1,3),Q(1,2),Q(4,5),Q(1)):
                    law=deletion_law(x,q)
                    self.assertEqual(sum(law.values()),1)
                    self.assertEqual(sum((p for t,p in law.items()
                                          if len(t)==0),Q()),q**n)
                    self.assertTrue(all(len(t)<=n and p>=0
                                        for t,p in law.items()))

    def test_decks_equal_up_to_two_and_not_three(self):
        a,b="0110","1001"
        self.assertTrue(deck_equal_through(a,b,2))
        self.assertNotEqual(deletion_deck(a,3),deletion_deck(b,3))
        self.assertFalse(deck_equal_through(a,b,3))

    def test_deck_tail_bound_on_all_small_collisions(self):
        for n in (2,3,4,5):
            for x,y in itertools.combinations(words(n),2):
                for k in range(n):
                    if deck_equal_through(x,y,k):
                        q=Q(4,5)
                        tv=total_variation(deletion_law(x,q),deletion_law(y,q))
                        self.assertLessEqual(tv,retained_length_tail(n,q,k))

    def test_single_and_product_trace_tv_coupling(self):
        a,b="0110","1001"
        for q in (Q(1,3),Q(1,2),Q(4,5)):
            law_a,law_b=deletion_law(a,q),deletion_law(b,q)
            delta=total_variation(law_a,law_b)
            self.assertGreater(delta,0)
            for m in range(4):
                observed=total_variation(product_law(law_a,m),
                                         product_law(law_b,m))
                self.assertLessEqual(observed,1-(1-delta)**m)
                self.assertLessEqual((1+observed)/2,
                                     hard_pair_success_ceiling(delta,m))

    def test_hardest_pair_search_agrees_exhaustive(self):
        n,q=4,Q(1,2)
        found=hard_pair(n,q)
        delta=Q(found["one_trace_tv"])
        brute=min(total_variation(deletion_law(x,q),deletion_law(y,q))
                  for x,y in itertools.combinations(words(n),2))
        self.assertEqual(delta,brute)

    def test_all_terminal_finite_state_success_ceiling(self):
        for n in (2,3,4):
            for q in (Q(1,2),Q(4,5)):
                for B in (1,2,3,4):
                    for m in (0,1,2,3):
                        laws=streaming_state_laws(n,q,m,B)
                        success=optimal_terminal_success(laws)
                        self.assertLessEqual(success,
                                             terminal_state_ceiling(n,B,m))
                        self.assertGreaterEqual(success,Q(1,1<<n))

    def test_finite_storage_saturation_with_traces(self):
        n,q=4,Q(4,5)
        for m in (0,1,2):
            upper=optimal_full_trace_success(n,q,m)
            for B in (1,2,3,4):
                bounded=optimal_terminal_success(
                    streaming_state_laws(n,q,m,B))
                self.assertLessEqual(bounded,upper)

    def test_zero_and_clock_terminal_cap(self):
        self.assertEqual(terminal_state_ceiling(8,0,100),Q(1,256))
        self.assertEqual(terminal_state_ceiling(8,4,3),Q(1,16))
        self.assertEqual(terminal_state_ceiling(8,4,3,clock=True),Q(1,4))
        self.assertEqual(terminal_state_ceiling(8,8,7),1)
        self.assertEqual(hard_pair_success_ceiling(Q(0),10),Q(1,2))

    def test_trace_bit_volume_gate(self):
        result=universal_information_ceiling(8,Q(1,4),3,2,clock=True)
        self.assertEqual(result["expected_retained_trace_bits"],"6")
        self.assertEqual(result["terminal_alphabet"],4*4)
        self.assertEqual(result["source_entropy_bits"],8)

    def test_exact_binomial_markov_thinning(self):
        for m in range(0,45):
            for r in (Q(0),Q(1,10),Q(1,5),Q(1,2),Q(1)):
                k=thinning_markov_cap(m,r)
                self.assertGreaterEqual(k,6*m*r)
                self.assertLessEqual(binomial_tail(m,r,k),Q(1,6))

    def test_gaussian_cap_bound_3d_sphere(self):
        for eps in (.05,.1,.3,.5,1.0,1.4):
            exact=(1-math.cos(eps))/2
            self.assertLessEqual(exact,gaussian_cap_upper(3,eps))
        self.assertLess(gaussian_terminal_ceiling(16,.1,8),.001)
        self.assertEqual(gaussian_terminal_ceiling(2,math.pi/2,4),1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): deletion_law("021",Q(1,2))
        with self.assertRaises(ValueError): deletion_law("01",Q(3,2))
        with self.assertRaises(ValueError): product_law({"":Q(1)},5)
        with self.assertRaises(ValueError): streaming_state_laws(4,Q(1,2),2,0)
        with self.assertRaises(ValueError): gaussian_cap_upper(1,.1)
        with self.assertRaises(ValueError): binomial_tail(1,Q(2),0)
        with self.assertRaises(ValueError): retained_length_tail(4,Q(1,2),5)

    def test_one_bit_monitor_recovers_compressible_prior(self):
        self.assertEqual(structured_two_word_success(8,0,Q(4,5)),Q(1,2))
        self.assertEqual(structured_two_word_success(8,1,Q(4,5)),
                         1-Q(1,2)*Q(4,5)**8)
        self.assertGreater(structured_two_word_success(8,1,Q(4,5)),Q(9,10))
        self.assertGreater(structured_two_word_success(8,3,Q(4,5)),
                           structured_two_word_success(8,1,Q(4,5)))

    def test_complete_fixed_receipt(self):
        r=control_report()
        self.assertEqual(r["status"],"EXACT_FINITE_CONTROL")
        self.assertEqual(r["deck_witness"]["equal_through"],2)
        self.assertEqual(r["word_length"],4)
        self.assertFalse(r["trained_neural_checkpoint_evaluated"])
        self.assertFalse(r["source_122_superpolynomial_hard_pair_reproduced"])
        self.assertFalse(r["source_140_gaussian_lower_bound_reproduced"])


if __name__=="__main__":
    unittest.main()
