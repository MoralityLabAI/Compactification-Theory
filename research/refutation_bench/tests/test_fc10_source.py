"""FC-10: actual Lean tactic-state source isolation, no label leakage."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench.fc10 import core,train_data


class FC10SourceTests(unittest.TestCase):
    def test_source_counts_and_family_disjointness(self):
        rows=train_data.source_rows()
        self.assertEqual(len(rows),156)
        self.assertEqual(Counter(x["split"] for x in rows),{"train":132,"dev":24})
        self.assertEqual(len({x["signature_sha256"] for x in rows}),156)
        self.assertTrue(set(core.TRAIN_FAMILIES).isdisjoint(core.DEV_FAMILIES))
        self.assertEqual(len(core.TACTICS),12)

    def test_reference_tactics_in_registered_action_set(self):
        for task in train_data.source_rows():
            self.assertIn(task["reference_tactic"],core.TACTICS)
            self.assertIn("aux_le",task["signature"])
            self.assertEqual(task["lean_version"],"4.19.0")
            source=core.render_theorem(task)
            self.assertIn('trace_state',source)
            self.assertIn("FC10_BEGIN:"+task["id"],source)
            self.assertIn("FC10_END:"+task["id"],source)

    def test_reject_unregistered_candidate_and_invalid_id(self):
        task=train_data.source_rows()[0]
        with self.assertRaises(ValueError):
            core.render_theorem(task,tactic="sorry")
        bad=deepcopy(task)
        bad["id"]="not_a_lean_theorem"
        with self.assertRaises(ValueError):
            core.render_theorem(bad)

    def test_parser_extracts_only_actual_trace_text(self):
        task=train_data.source_rows()[0]
        raw="FC10_BEGIN:"+task["id"]+"\nA : Prop\nh : A\n⊢ A\nFC10_END:"+task["id"]
        parsed=core.parse_actual_goal_states(raw,[task])
        self.assertEqual(parsed[0]["goal_state"],"A : Prop\nh : A\n⊢ A")
        self.assertEqual(parsed[0]["reference_tactic"],task["reference_tactic"])
        self.assertNotIn("signature",parsed[0])

    def test_parser_rejects_missing_and_malformed_state(self):
        task=train_data.source_rows()[0]
        with self.assertRaises(AssertionError):
            core.parse_actual_goal_states("nothing",[task])
        with self.assertRaises(AssertionError):
            core.parse_actual_goal_states(
                "FC10_BEGIN:"+task["id"]+"\nno lean goal\nFC10_END:"+task["id"],[task])
        with self.assertRaises(AssertionError):
            core.parse_actual_goal_states(
                "FC10_BEGIN:"+task["id"]+"\n⊢ True\nFC10_END:fc10_test_9999",[task])

    def test_deterministic_source_and_name_renumbering(self):
        a=train_data.source_rows()
        b=train_data.source_rows()
        self.assertEqual(a,b)
        self.assertEqual(a[0]["id"],"fc10_train_0000")
        self.assertEqual(b[-1]["id"],"fc10_dev_0155")


if __name__=="__main__":
    unittest.main()
