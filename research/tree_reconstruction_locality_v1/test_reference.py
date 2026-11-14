"""KSR-01: independent finite broadcast, martingale and FIID-boundary checks."""
from __future__ import annotations

import itertools
import unittest
from fractions import Fraction as Q

from reference import (
    bayes_floor, bit_budget, covariance_oracle, dropout_moment_factor,
    exact_broadcast, leaf_pair_covariance_factor, moment_closed,
    moment_factor, reference_report, supercritical_limit, transition,
    validate,
)


class ExactBroadcastTreeTests(unittest.TestCase):
    def test_channel_rows_and_eigenspace(self):
        for q in (2,3,4,5):
            for lam in (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)):
                P=transition(q,lam)
                self.assertEqual(len(P),q)
                for row in P:
                    self.assertEqual(sum(row),1)
                    self.assertTrue(all(0<=v<=1 for v in row))
                # Symmetric state contrast g_j(s)=q*1{s=j}-1.
                for root in range(q):
                    for j in range(q):
                        expected=q*(root==j)-1
                        actual=sum(P[root][s]*(q*(s==j)-1)
                                   for s in range(q))
                        self.assertEqual(actual,lam*expected)
            # Negative eigenvalues are valid down to -1/(q-1).
            lam=-Q(1,q-1)
            self.assertTrue(all(v>=0 for row in transition(q,lam)
                                for v in row))

    def test_rational_second_moment_closed_vs_recursion(self):
        for b in (2,3,4,5):
            for lam in (Q(1,4),Q(1,2),Q(4,5),Q(1)):
                for L in range(12):
                    self.assertEqual(moment_factor(b,lam,L),
                                     moment_closed(b,lam,L))
                    self.assertGreaterEqual(moment_factor(b,lam,L),1)

    def test_leaf_LCA_covariance_independent_oracle(self):
        for b in (2,3,4):
            for lam in (Q(1,4),Q(1,2),Q(3,4),Q(-1,4)):
                for L in range(4):
                    self.assertEqual(covariance_oracle(b,lam,L),
                                     moment_factor(b,lam,L))
        self.assertEqual(leaf_pair_covariance_factor(2,2,0,0,Q(1,2)),1)
        self.assertEqual(leaf_pair_covariance_factor(2,2,0,1,Q(1,2)),Q(1,4))
        self.assertEqual(leaf_pair_covariance_factor(2,2,0,3,Q(1,2)),Q(1,16))

    def test_three_state_full_bayes_and_count_bayes_bounds(self):
        for lam in (Q(1,2),Q(4,5)):
            for depth in (1,2,3):
                result=exact_broadcast(3,2,lam,depth)
                self.assertGreaterEqual(Q(result["count_MAP"]),
                                        Q(result["count_lower_bound"]))
                self.assertGreaterEqual(Q(result["full_leaf_MAP"]),
                                        Q(result["count_MAP"]))
                self.assertLessEqual(Q(result["full_leaf_MAP"]),1)
                self.assertEqual(Q(result["score_root_covariance"]),2)
                self.assertEqual(Q(result["score_second_moment"]),
                                 2*Q(result["moment_factor"]))

    def test_two_state_exact_full_and_count_bayes_bounds(self):
        for lam in (Q(1,2),Q(4,5),Q(1)):
            for depth in (1,2,3):
                x=exact_broadcast(2,2,lam,depth)
                self.assertLessEqual(Q(x["count_lower_bound"]),Q(x["count_MAP"]))
                self.assertLessEqual(Q(x["count_MAP"]),Q(x["full_leaf_MAP"]))
        self.assertEqual(Q(exact_broadcast(2,2,1,3)["count_MAP"]),1)

    def test_noisy_erasure_moment_and_bayes(self):
        for q,b,lam,L,r in [
            (2,2,Q(4,5),3,Q(1,2)),
            (3,2,Q(4,5),2,Q(1,2)),
            (3,2,Q(4,5),2,Q(1,4)),
            (2,4,Q(1,2),1,Q(1,3)),
        ]:
            actual=exact_broadcast(q,b,lam,L,r)
            self.assertEqual(Q(actual["moment_factor"]),
                             dropout_moment_factor(b,lam,L,r))
            self.assertGreaterEqual(Q(actual["count_MAP"]),
                                    Q(actual["count_lower_bound"]))
            self.assertEqual(Q(actual["score_root_covariance"]),q-1)
            self.assertEqual(
                Q(actual["score_second_moment"]),
                (q-1)*dropout_moment_factor(b,lam,L,r))
            self.assertLessEqual(Q(actual["full_leaf_MAP"]),1)

    def test_dropout_penalty_exact_formula(self):
        b,lam,L=2,Q(4,5),4
        t=b*lam*lam
        for r in (Q(1,4),Q(1,2),Q(3,4),Q(1)):
            V=dropout_moment_factor(b,lam,L,r)
            self.assertEqual(V-moment_factor(b,lam,L),
                             (1-r)/(r*t**L))
            self.assertGreaterEqual(V,moment_factor(b,lam,L))
        self.assertEqual(dropout_moment_factor(4,Q(1,2),8,Q(1,2)),
                         7+1)

    def test_supercritical_long_depth_advantage_and_fiid_gate(self):
        a=supercritical_limit(2,2,Q(4,5))
        b=supercritical_limit(3,2,Q(4,5))
        self.assertEqual(a["limiting_recovery_lower_bound"],"23/32")
        self.assertEqual(b["limiting_recovery_lower_bound"],"5/8")
        self.assertEqual(a["t"],"32/25")
        self.assertEqual(b["limiting_variance_factor"],"16/7")
        # The infinite 3-regular Ising result uses b=d-1=2.
        # Source #236 is imported, not reproved by this local check.
        for depth in (1,2,4,8,16,32):
            self.assertGreater(bayes_floor(3,2,Q(4,5),depth),Q(5,8))
        with self.assertRaises(ValueError):
            supercritical_limit(3,4,Q(1,2))

    def test_critical_variance_linear_growth_and_memory(self):
        self.assertEqual(moment_factor(4,Q(1,2),8),7)
        self.assertEqual(bayes_floor(3,4,Q(1,2),8),Q(3,7))
        for depth in (0,1,2,3,10):
            self.assertEqual(moment_factor(4,Q(1,2),depth),
                             1+Q(3,4)*depth)
            bits=bit_budget(3,4,depth)
            self.assertEqual(bits["persistent_bit_upper_bound"],
                             2*(4**depth).bit_length())
            self.assertEqual(bits["leaf_symbols"],4**depth)

    def test_counter_memory_linearly_scales_in_depth(self):
        for depth in range(1,20):
            N=2**depth
            budget=bit_budget(3,2,depth,1)
            self.assertEqual(budget["persistent_bit_upper_bound"],
                             2*(depth+1))
            self.assertEqual(budget["raw_alphabet_symbols"],N)
            erased=bit_budget(3,2,depth,Q(1,2))
            self.assertEqual(erased["persistent_bit_upper_bound"],
                             3*(depth+1))

    def test_full_reference_receipt_has_source_boundaries(self):
        r=reference_report()
        self.assertEqual(r["schema"],"ksr01.exact_broadcast.v1")
        self.assertEqual(r["status"],"EXACT_FINITE_TREE_CONTROL")
        self.assertEqual(r["source_result_families"],[229,236])
        self.assertEqual(len(r["cases"]),13)
        self.assertFalse(r["trains_neural_checkpoint"])
        self.assertFalse(r["reproduces_source_236_infinite_factor_construction"])
        self.assertFalse(r["reproduces_source_229_full_infinite_tree_proof"])

    def test_negative_lambda_changes_sign_not_variance(self):
        x=exact_broadcast(3,2,-Q(1,2),2)
        self.assertEqual(Q(x["score_second_moment"]),2*Q(x["moment_factor"]))
        self.assertEqual(x["moment_factor"],
                         str(moment_factor(2,Q(1,2),2)))

    def test_invalid_probabilities_and_sizes(self):
        for args in [(1,2,Q(1,2)),(3,1,Q(1,2)),(3,2,Q(-3,4)),
                     (3,2,Q(2))]:
            with self.assertRaises(ValueError):
                validate(*args)
        with self.assertRaises(ValueError):
            exact_broadcast(3,2,Q(4,5),3,Q(0))
        with self.assertRaises(ValueError):
            exact_broadcast(4,2,Q(4,5),3,Q(1,2))
        with self.assertRaises(ValueError):
            moment_factor(2,0,2)


if __name__=="__main__":
    unittest.main()
