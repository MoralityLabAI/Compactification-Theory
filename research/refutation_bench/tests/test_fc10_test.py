"""FC-10 test source isolation and real Lean trace extraction checks."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest
from research.refutation_bench.fc10 import core,test_data,train_data


class FC10TestSourceTests(unittest.TestCase):
    def test_counts_and_families(self):
        rows=test_data.test_rows()
        self.assertEqual(len(rows),48)
        self.assertEqual(Counter(r["family"] for r in rows),
                         {family:12 for family in core.TEST_FAMILIES})
        self.assertTrue(set(core.TEST_FAMILIES).isdisjoint(core.TRAIN_FAMILIES))
        self.assertTrue(set(core.TEST_FAMILIES).isdisjoint(core.DEV_FAMILIES))

    def test_signatures_are_unique_across_all_splits(self):
        rows=train_data.source_rows()+test_data.test_rows()
        fingerprints=[r["signature_sha256"] for r in rows]
        self.assertEqual(len(fingerprints),204)
        self.assertEqual(len(set(fingerprints)),204)
        self.assertTrue(all(r["reference_tactic"] in core.TACTICS for r in rows))

    def test_all_test_goals_have_trusted_reference_tactic(self):
        for row in test_data.test_rows():
            src=core.render_theorem(row)
            self.assertIn("trace_state",src)
            self.assertIn("FC10_BEGIN:"+row["id"],src)
            self.assertIn("FC10_END:"+row["id"],src)
        self.assertTrue(all(r["split"]=="test" for r in test_data.test_rows()))

    def test_candidate_replacement_never_repairs_illegal_action(self):
        task=test_data.test_rows()[0]
        with self.assertRaises(ValueError):
            core.render_theorem(task,tactic="sorry")
        with self.assertRaises(ValueError):
            core.render_theorem(task,tactic="admit")

    def test_no_reference_label_in_actual_goal_payload(self):
        row=test_data.test_rows()[0]
        raw=(f'FC10_BEGIN:{row["id"]}\n'
             'a b : Nat\nh : a ≤ b\n⊢ a ≤ b\n'
             f'FC10_END:{row["id"]}')
        parsed=core.parse_actual_goal_states(raw,[row])
        self.assertNotIn("reference_tactic",parsed[0]["goal_state"])
        self.assertNotIn("signature_sha256",parsed[0]["goal_state"])


if __name__=="__main__":
    unittest.main()
