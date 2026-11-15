"""NOR-15C exhaustive four-check mask-policy algebra and oracle split tests."""
from __future__ import annotations

from itertools import permutations
import unittest

from research.nor15_late_source_ceiling import ceiling as c


def case(public,correct):
    return {"public_accepted_mask_16":public,
            "withheld_future_correct_mask_16":correct}


class NOR15CCeilingTest(unittest.TestCase):
    def test_fixed_verifier_stops_at_first_public_accepted(self):
        x=c.rank_one((1<<2)|(1<<5),1<<5,[0,2,5,7])
        self.assertEqual(x["false_accept"],1)
        self.assertEqual(x["future_correct"],0)
        self.assertEqual(x["paid_checks"],2)
        self.assertEqual(x["accepted_template"],2)
        y=c.rank_one((1<<2)|(1<<5),1<<5,[0,5,2,7])
        self.assertEqual(y["future_correct"],1)
        self.assertEqual(y["paid_checks"],2)
        self.assertEqual(y["accepted_template"],5)
        none=c.rank_one(0,0,[0,2,5,7])
        self.assertEqual(none["abstained"],1)
        self.assertEqual(none["paid_checks"],4)
        with self.assertRaises(AssertionError):
            c.rank_one(3,1,[0,0,1,2])

    def test_exhaustively_identifies_maximum_and_reproduces_mask_scores(self):
        rows=[
            case(1<<1,1<<1),
            case(1<<3,1<<3),
            case((1<<1)|(1<<3),1<<3),
            case((1<<1)|(1<<3),1<<3),
            case(1<<5,1<<5),
            case(0,0),
        ]
        result=c.optimize_order(rows)
        self.assertEqual(result["orders_exhaustively_checked"],43680)
        self.assertEqual(result["future_correct"],5)
        self.assertEqual(result["abstained"],1)
        self.assertEqual(len(result["ordered_top4"]),4)
        actual=c.score_order(rows,result["ordered_top4"])
        compressed=c.score_unique_signatures(rows,result["ordered_top4"])
        self.assertEqual(compressed["future_correct"],actual["future_correct"])
        self.assertEqual(compressed["false_accept"],actual["false_accept"])
        self.assertEqual(compressed["paid_checks"],actual["paid_checks"])
        self.assertEqual(result["future_correct"],actual["future_correct"])

    def test_full_public_mask_refined_oracle_bounds_fixed_ranking(self):
        rows=[
            case((1<<0)|(1<<3),1<<3),
            case((1<<0)|(1<<3),1<<0),
            case((1<<0)|(1<<3),1<<3),
            case(1<<7,1<<7),
            case(0,0),
        ]
        oracle=c.mask_conditional_hindsight(rows)
        self.assertEqual(oracle["public_mask_equivalence_groups"],3)
        self.assertEqual(oracle["hindsight_correct_with_one_template_per_public_mask"],3)
        self.assertEqual(oracle["per_case_any_of_sixteen_future_correct_upper_bound"],4)
        self.assertTrue(oracle["uses_withheld_future_grades_for_diagnostic_best_choice"])

    def test_guard_source_budgets_and_four_tuples(self):
        with self.assertRaises(ValueError):
            c.late_source_new(set(),161,c.TEST_OFFSET,"test")
        with self.assertRaises(ValueError):
            c.late_source_new(set(),8,0,"invalid")
        with self.assertRaises(ValueError):
            c.run(79,160)
        with self.assertRaises(ValueError):
            c.run(80,159)
        self.assertEqual(sum(1 for _ in permutations(range(16),4)),43680)


if __name__=="__main__":
    unittest.main()
