"""FC-08 held-out conjunctive Horn exact labels and frozen symbolic controls."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.fc08 import train_data,test_data,controls
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.fc07.posthoc_unary import unary_shortest_trace
from research.refutation_bench.fc06.posthoc_backward_slice import goal_ancestors
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC08TestCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=train_data.generate_train_dev()
        cls.records=test_data.generate_tests()

    def test_new_families_counts_and_WL_source_isolation(self):
        summary=test_data.audit_tests(self.records,training=self.training)
        self.assertEqual(summary["count"],384)
        self.assertEqual((summary["test_same"],summary["test_deep"]),(192,192))
        self.assertEqual((summary["positive"],summary["negative"]),(192,192))
        self.assertEqual(summary["prior_WL_overlap"],0)
        past={x["fingerprint"] for x in self.training}
        past.update(train_data.previous_fingerprints())
        self.assertTrue(past.isdisjoint(x["fingerprint"] for x in self.records))
        for split in test_data.TEST_SPECS:
            for family in test_data.TEST_FAMILIES:
                items=[x for x in self.records
                       if x["split"]==split and x["family_id"]==family]
                self.assertEqual(Counter(x["label"] for x in items),
                                 {"established":24,"refuted":24})

    def test_no_unary_goal_path_and_exact_short_positive_proof(self):
        for task in self.records:
            self.assertEqual(len(task["atoms"]),12)
            self.assertLessEqual(len(task["rules"]),20)
            self.assertFalse(unary_shortest_trace(task,budget=5))
            self.assertTrue(all(len(r["premises"])>=2
                        for r in task["rules"] if r["head"]==task["goal"]))
            if task["label"]!="established":
                continue
            self.assertEqual(task["depth"],task["min_steps"])
            self.assertEqual(len(task["reachable"])-len(task["facts"]),10)
            self.assertGreaterEqual(len(goal_ancestors(task)),11)
            self.assertTrue(proofs.verify_trace(
                task,task["shortest_trace"],budget=5)["verified"])
            self.assertIn("theorem "+task["case_id"].lower(),fc02.lean_theorem(task))

    def test_exhaustive_negative_countermodels(self):
        for task in self.records:
            f=task["formal"]
            counter=find_countermodel(f["premises"],f["conclusion"])
            if task["label"]=="refuted":
                self.assertIsNotNone(counter)
                self.assertTrue(verify_countermodel(
                    f["premises"],f["conclusion"],
                    task["countermodel"]["assignment"])["verified"])
                self.assertIsNone(task["shortest_trace"])
            else:
                self.assertIsNone(counter)

    def test_preregistered_baselines_are_source_only_and_limit_search(self):
        info=test_data.audit_tests(self.records,training=self.training)
        for split in test_data.TEST_SPECS:
            for policy,values in info["first_observed_source_only_controls"][split].items():
                self.assertEqual(values["eligible_valid"],96)
                self.assertTrue(0<=values["certified_valid"]<=96)
                if policy=="unary_bfs":
                    self.assertEqual(values["certified_valid"],0)
                if policy=="bounded_best_first":
                    self.assertTrue(0<=values["states_popped"]<=24*96)
        positive=next(t for t in self.records if t["label"]=="established")
        for policy in ("random_legal","goal_first","backward_slice",
                       "unary_bfs","bounded_best_first"):
            x=controls.baseline_trace(positive,policy)
            self.assertLessEqual(len(x["trace"]),5)
            self.assertLessEqual(x["states_popped"],24)

    def test_no_gold_or_reference_trace_in_graph_projection(self):
        for task in self.records:
            public={key:task[key] for key in ("atoms","facts","rules","goal")}
            self.assertFalse(set(public)&{"label","split","family_id",
               "shortest_trace","countermodel","min_steps","reachable","formal"})

    def test_tampering_fails_before_use(self):
        items=deepcopy(self.records)
        items[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            test_data.audit_tests(items,training=self.training)
        items=deepcopy(self.records)
        items[0]["fingerprint"]=self.training[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            test_data.audit_tests(items,training=self.training)

    def test_deterministic_generation(self):
        repeats=test_data.generate_tests()
        digest=lambda rows:hashlib.sha256(b"".join(
            fc02.canonical_bytes(x)+b"\n" for x in rows)).hexdigest()
        self.assertEqual(digest(self.records),digest(repeats))


if __name__=="__main__":
    unittest.main()
