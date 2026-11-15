"""NOR-15D exact public-acceptance-mask ceiling arithmetic tests."""
from __future__ import annotations
import itertools
import unittest

from research.nor15d_mask_capacity import mask_ceiling as n


class NOR15DMaskCapacity(unittest.TestCase):
    def row(self,accept,correct,key):
        return {"public_accepted_bitmask":accept,
                "future_correct_bitmask_hindsight":correct,
                "public_visible_sha256":key}

    def test_identical_public_mask_capacity_is_best_constant_template(self):
        a=(1<<0)|(1<<1)
        group=[self.row(a,1<<0,"x"),self.row(a,1<<1,"y"),
               self.row(a,1<<1,"z"),self.row(a,1<<0,"w")]
        x=n.group_capacity(group)
        self.assertEqual(x["cases"],4)
        self.assertEqual(x["oracle_future_correct"],2)
        self.assertEqual(x["unavoidable_future_incorrect_under_same_mask_choice"],2)
        self.assertEqual(x["oracle_selected_template"],0)
        for choice in [0,1]:
            count=sum(bool(r["future_correct_bitmask_hindsight"]>>choice & 1)
                      for r in group)
            self.assertLessEqual(count,x["oracle_future_correct"])

    def test_reject_mixed_or_zero_masks(self):
        with self.assertRaises(ValueError):n.group_capacity([])
        with self.assertRaises(ValueError):
            n.group_capacity([self.row(0,0,"x")])
        with self.assertRaises(ValueError):
            n.group_capacity([self.row(1,1,"x"),self.row(2,2,"y")])

    def test_source_locked_exact_fourteen_mask_capacity(self):
        out=n.run()
        self.assertEqual(out["distinct_complete_public_masks"],14)
        self.assertEqual(out["total_public_mask_cases"],40)
        self.assertEqual(out["best_hindsight_full_mask_future_correct"],25)
        self.assertEqual(out["best_hindsight_fixed_four_future_correct"],15)
        self.assertEqual(out["original_fixed_four_future_correct"],8)
        self.assertEqual(out["unavoidable_lost_future_correct_under_complete_mask_only"],15)
        self.assertEqual(out["all_candidates_publicly_accepted_group"]["cases"],13)
        self.assertEqual(out["all_candidates_publicly_accepted_group"]["oracle_future_correct"],3)
        self.assertEqual(len(out["capacity_integer_certificate"]),14)
        self.assertEqual(sum(out["capacity_integer_certificate"]),25)
        self.assertFalse(out["scope"]["not_deployable_model_evaluation"] is False)


if __name__=="__main__":
    unittest.main()
