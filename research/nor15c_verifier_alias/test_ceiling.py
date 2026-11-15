"""NOR-15C exact fixed-check ceiling tests independent of source generation."""
from __future__ import annotations
import itertools
import math
import unittest
from research.nor15c_verifier_alias import ceiling as n


class NOR15CFixedRankingTest(unittest.TestCase):
    def test_four_check_choice_stops_at_first_public_accepted_template(self):
        A=[(1<<0)|(1<<1), 1<<2, (1<<1)|(1<<2)]
        C=[1<<1, 1<<2, 1<<2]
        x=n.score_order_masks(A,C,(0,1,2,3))
        self.assertEqual(x["future_correct"],1)
        self.assertEqual(x["false_accept"],2)
        self.assertEqual(x["accepted"],3)
        self.assertEqual(x["paid_checks"],1+3+2)
        good=n.score_order_masks(A,C,(2,1,0,3))
        self.assertEqual(good["future_correct"],3)
        self.assertEqual(good["false_accept"],0)
        self.assertEqual(good["paid_checks"],2+1+1)

    def test_exhaustive_toy_two_check_order_oracle_is_exact(self):
        A=[1<<0,1<<1,(1<<0)|(1<<1)]
        C=[1<<0,1<<1,1<<1]
        x=n.enumerate_fixed_rankings(A,C,candidates=4,budget=2)
        self.assertEqual(x["distinct_fixed_orders_exhaustively_checked"],12)
        self.assertEqual(x["hindsight_optimal_metrics"]["future_correct"],3)
        self.assertEqual(x["hindsight_optimal_metrics"]["false_accept"],0)
        self.assertEqual(x["hindsight_optimal_order"],[1,0])
        self.assertFalse(x["inference_feasible_untrained_oracle"])
        self.assertTrue(x["posterior_test_future_used_to_select_oracle"])

    def test_four_out_of_sixteen_has_exact_43680_policy_space(self):
        self.assertEqual(len(list(itertools.permutations(range(16),4))),43680)
        A=[1<<i for i in range(4)]
        C=list(A)
        x=n.enumerate_fixed_rankings(A,C)
        self.assertEqual(x["distinct_fixed_orders_exhaustively_checked"],43680)
        self.assertEqual(x["hindsight_optimal_metrics"]["future_correct"],4)
        self.assertEqual(x["hindsight_optimal_metrics"]["false_accept"],0)
        self.assertEqual(x["min_false_accept_for_each_future_correct"][-1]["min_false_accept"],0)

    def test_future_mask_cannot_exceed_public_acceptance_mask(self):
        with self.assertRaises(ValueError):
            n.enumerate_fixed_rankings([1<<0],[1<<1],candidates=4,budget=2)
        with self.assertRaises(AssertionError):
            n.score_order_masks([1<<0],[1<<1],[0,1])
        with self.assertRaises(ValueError):
            n.score_order_masks([1<<0],[1<<0],[0,0])
        with self.assertRaises(ValueError):
            n.enumerate_fixed_rankings([1],[1],candidates=3,budget=4)

    def test_all_rejected_cases_charge_full_budget_and_return_zero(self):
        x=n.score_order_masks([1<<5,1<<6],[1<<5,1<<6],[0,1,2,3])
        self.assertEqual(x["future_correct"],0)
        self.assertEqual(x["false_accept"],0)
        self.assertEqual(x["paid_checks"],8)


if __name__=="__main__":
    unittest.main()
