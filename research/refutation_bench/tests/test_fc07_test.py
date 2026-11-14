"""FC-07 held-out ancestry alternatives: proof oracle, strict budget, leakage."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.fc07 import train_data,test_data
from research.refutation_bench.fc06.posthoc_backward_slice import goal_ancestors,priority_trace
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=train_data.generate_train_dev()
        cls.tests=test_data.generate_tests()

    def test_frozen_family_counts_and_disjoint_WL(self):
        report=test_data.audit_tests(self.tests,train_records=self.training)
        self.assertEqual(report["cases"],384)
        self.assertEqual(report["splits"],{"test_same":192,"test_deep":192})
        self.assertEqual((report["positive"],report["negative"]),(192,192))
        self.assertEqual(report["prior_overlap"],0)
        self.assertEqual(report["distinct_WL"],384)
        prior={x["fingerprint"] for x in self.training}
        prior|=set(train_data.previous_fingerprints())
        self.assertTrue(prior.isdisjoint(x["fingerprint"] for x in self.tests))
        for split in test_data.TEST_SPECS:
            for family in test_data.TEST_FAMILIES:
                items=[x for x in self.tests if x["split"]==split and x["family_id"]==family]
                self.assertEqual(Counter(x["label"] for x in items),
                                 {"established":24,"refuted":24})

    def test_positive_shortest_proofs_and_reachable_ancestral_decoys(self):
        for task in self.tests:
            self.assertEqual(len(task["atoms"]),12)
            self.assertLessEqual(len(task["rules"]),20)
            if task["label"]!="established":
                continue
            self.assertEqual(task["min_steps"],task["depth"])
            self.assertLessEqual(task["min_steps"],5)
            self.assertEqual(len(task["reachable"])-len(task["facts"]),10)
            self.assertTrue(set(task["reachable"])<=goal_ancestors(task))
            self.assertTrue(proofs.verify_trace(task,task["shortest_trace"],budget=5)["verified"])
            self.assertIn("theorem "+task["case_id"].lower(),fc02.lean_theorem(task))

    def test_exact_negative_countermodels(self):
        for task in self.tests:
            p=task["formal"]
            counter=find_countermodel(p["premises"],p["conclusion"])
            if task["label"]=="established":
                self.assertIsNone(counter)
            else:
                self.assertIsNotNone(counter)
                self.assertIsNone(task["shortest_trace"])
                self.assertTrue(verify_countermodel(
                    p["premises"],p["conclusion"],
                    task["countermodel"]["assignment"])["verified"])

    def test_backward_slice_baseline_recorded_without_test_case_filtering(self):
        report=test_data.audit_tests(self.tests,train_records=self.training)
        for split in ("test_same","test_deep"):
            relevant=[x for x in self.tests if x["split"]==split and x["label"]=="established"]
            successes=sum(proofs.verify_trace(x,priority_trace(x,budget=5),budget=5)["verified"]
                          for x in relevant)
            recorded=report["backward_slice_first_observed"][split]
            self.assertEqual(recorded,{"valid":96,"backward_slice_success":successes})

    def test_no_model_input_may_include_reference_trace_or_gold(self):
        for t in self.tests:
            public={name:t[name] for name in ("atoms","facts","rules","goal")}
            self.assertFalse(set(public)&{
                "label","min_steps","shortest_trace","family_id",
                "split","reachable","countermodel","formal"})

    def test_source_tampering_fails_audit(self):
        rows=deepcopy(self.tests)
        rows[0]["shortest_trace"]=[]
        with self.assertRaises(AssertionError):
            test_data.audit_tests(rows,train_records=self.training)
        rows=deepcopy(self.tests)
        rows[0]["fingerprint"]=self.training[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            test_data.audit_tests(rows,train_records=self.training)

    def test_generator_determinism(self):
        other=test_data.generate_tests()
        digest=lambda rows:hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n"
                                            for x in rows)).hexdigest()
        self.assertEqual(digest(self.tests),digest(other))


if __name__=="__main__":
    unittest.main()
