"""FC-07 train-only graph sanity, exact certificates and goal-ancestry control."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench.fc07 import train_data as f
from research.refutation_bench.fc06.posthoc_backward_slice import priority_trace,goal_ancestors
from research.refutation_bench.fc05.proofs import verify_trace
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class TrainCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=f.generate_train_dev()

    def test_preregistered_counts_and_disjoint_families(self):
        report=f.audit_train_dev(self.rows)
        self.assertEqual((report["train_count"],report["dev_count"]),(768,192))
        self.assertEqual((report["positive"],report["negative"]),(480,480))
        self.assertEqual(report["proof_action_budget"],5)
        self.assertEqual(report["previous_overlap"],0)
        self.assertEqual(report["fingerprints_distinct"],960)
        self.assertFalse(set(f.TRAIN_FAMILIES)&set(f.DEV_FAMILIES))

    def test_true_goal_has_short_proof_but_many_ancestors(self):
        for row in self.rows:
            self.assertEqual(len(row["atoms"]),12)
            self.assertLessEqual(len(row["rules"]),20)
            if row["label"]!="established":
                continue
            self.assertEqual(row["min_steps"],row["depth"])
            self.assertLessEqual(row["min_steps"],5)
            self.assertEqual(len(row["reachable"])-len(row["facts"]),10)
            self.assertGreaterEqual(len(goal_ancestors(row)),11)
            self.assertTrue(verify_trace(row,row["shortest_trace"],budget=5)["verified"])

    def test_all_invalid_have_exact_refuting_assignments(self):
        for row in self.rows:
            if row["label"]!="refuted":
                continue
            p=row["formal"]
            self.assertIsNotNone(find_countermodel(p["premises"],p["conclusion"]))
            self.assertTrue(verify_countermodel(
                p["premises"],p["conclusion"],
                row["countermodel"]["assignment"])["verified"])
            self.assertIsNone(row["shortest_trace"])

    def test_naive_ancestry_is_not_a_perfect_training_policy(self):
        positives=[r for r in self.rows if r["split"]=="train" and r["label"]=="established"]
        successes=sum(verify_trace(r,priority_trace(r,budget=5),budget=5)["verified"]
                      for r in positives)
        self.assertGreater(successes,0)
        self.assertLess(successes,len(positives))
        # Training-side control, never used to accept or reject test cases.
        print(f"TRAIN_BACKWARD_SLICE_SUCCESS {successes}/{len(positives)}",flush=True)

    def test_original_source_excludes_gold_from_model_inputs(self):
        for t in self.rows:
            x={key:t[key] for key in ("atoms","facts","rules","goal")}
            self.assertFalse(set(x)&{"label","shortest_trace","min_steps","split",
                                    "reachable","countermodel","family_id"})

    def test_wl_or_annotation_corruption_rejected(self):
        rows=deepcopy(self.rows)
        rows[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            f.audit_train_dev(rows)
        rows=deepcopy(self.rows)
        rows[0]["fingerprint"]=self.rows[1]["fingerprint"]
        with self.assertRaises(AssertionError):
            f.audit_train_dev(rows)


if __name__=="__main__":
    unittest.main()
