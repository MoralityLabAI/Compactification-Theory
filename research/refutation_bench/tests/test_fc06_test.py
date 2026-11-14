"""FC-06 virgin family-held-out strict proof budget source and certificate tests."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.fc06 import train_data as train
from research.refutation_bench.fc06 import test_data as test
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.certificates import find_countermodel,verify_countermodel
from research.refutation_bench.formal_curriculum import generator as fc02


class FC06HeldOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=train.generate_train_dev()
        cls.tests=test.generate_tests()

    def test_frozen_family_splits_and_no_overlap(self):
        report=test.audit_tests(self.tests,train_rows=self.train)
        self.assertEqual(report["case_count"],384)
        self.assertEqual((report["test_same"],report["test_deep"]),(192,192))
        self.assertEqual((report["positives"],report["negatives"]),(192,192))
        self.assertEqual(report["overlap_count"],0)
        self.assertEqual(report["max_proof_actions"],5)
        self.assertTrue({r["family_id"] for r in self.train}.isdisjoint(test.TEST_FAMILIES))
        prior={r["fingerprint"] for r in self.train}|set(train.older_fingerprints())
        self.assertTrue(prior.isdisjoint(r["fingerprint"] for r in self.tests))
        for split in test.TEST_SPECS:
            for family in test.TEST_FAMILIES:
                subset=[t for t in self.tests if t["split"]==split and t["family_id"]==family]
                self.assertEqual(Counter(t["label"] for t in subset),
                                 {"established":24,"refuted":24})

    def test_proof_budget_is_binding_and_minimal(self):
        for t in self.tests:
            self.assertEqual(len(t["atoms"]),12)
            self.assertLessEqual(len(t["rules"]),20)
            if t["label"]!="established":
                continue
            self.assertEqual(t["min_steps"],t["depth"])
            self.assertLessEqual(t["depth"],5)
            self.assertEqual(len(t["reachable"])-len(t["facts"]),10)
            self.assertTrue(proofs.verify_trace(t,t["shortest_trace"],budget=5)["verified"])

    def test_every_negative_has_independently_checked_countermodel(self):
        for t in self.tests:
            formula=t["formal"]
            counter=find_countermodel(formula["premises"],formula["conclusion"])
            if t["label"]=="refuted":
                self.assertIsNotNone(counter)
                self.assertTrue(verify_countermodel(
                    formula["premises"],formula["conclusion"],
                    t["countermodel"]["assignment"])["verified"])
                self.assertIsNone(t["shortest_trace"])
            else:
                self.assertIsNone(counter)
                self.assertIn("theorem "+t["case_id"].lower(),fc02.lean_theorem(t))

    def test_first_random_and_goal_first_baselines_are_recorded_not_filtered(self):
        result=test.audit_tests(self.tests,train_rows=self.train)
        for method in ("first_observed_random_control","first_observed_goal_first_control"):
            for split in ("test_same","test_deep"):
                entry=result[method][split]
                self.assertEqual(entry["total"],96)
                self.assertGreaterEqual(entry["correct"],0)
                self.assertLessEqual(entry["correct"],96)
                self.assertAlmostEqual(entry["positive_recall"],
                    round(entry["correct"]/96,6))

    def test_heldout_features_have_no_gold_keys(self):
        for item in self.tests:
            public={k:item[k] for k in ("atoms","rules","facts","goal")}
            self.assertFalse({"label","min_steps","shortest_trace","family_id",
                              "split","countermodel","reachable"}&public.keys())

    def test_tampering_rejected(self):
        tampered=deepcopy(self.tests)
        tampered[0]["shortest_trace"]=[0]
        with self.assertRaises(AssertionError):
            test.audit_tests(tampered,train_rows=self.train)
        tampered=deepcopy(self.tests)
        tampered[0]["fingerprint"]=self.train[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            test.audit_tests(tampered,train_rows=self.train)

    def test_generator_is_stable(self):
        second=test.generate_tests()
        digest=lambda rows:hashlib.sha256(b"".join(
            train.canonical_bytes(x)+b"\n" for x in rows)).hexdigest()
        self.assertEqual(digest(self.tests),digest(second))


if __name__=="__main__":
    unittest.main()
