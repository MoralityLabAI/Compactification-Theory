"""FC-08 train-only exact Horn/source certificate regression."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench.fc08 import train_data as f
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.fc06.posthoc_backward_slice import goal_ancestors
from research.refutation_bench.fc07.posthoc_unary import unary_shortest_trace
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC08TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=f.generate_train_dev()

    def test_frozen_train_dev_counts_and_WL_disjointness(self):
        report=f.audit_train_dev(self.rows)
        self.assertEqual((report["train"],report["dev"]),(768,192))
        self.assertEqual((report["positive"],report["negative"]),(480,480))
        self.assertEqual(report["past_overlap"],0)
        self.assertEqual(report["unique_fingerprints"],960)
        self.assertEqual(report["rule_action_budget"],5)
        for split in f.SPLITS:
            _,_,__,___,families=f.SPLITS[split]
            for family in families:
                cases=[x for x in self.rows if x["split"]==split
                       and x["family_id"]==family]
                self.assertEqual(Counter(x["label"] for x in cases),
                    {"established":len(cases)//2,"refuted":len(cases)//2})

    def test_every_valid_requires_conjunction_and_has_minimal_bounded_proof(self):
        for task in self.rows:
            self.assertEqual(len(task["atoms"]),12)
            self.assertLessEqual(len(task["rules"]),20)
            self.assertTrue(all(len(r["premises"])>=2 for r in task["rules"]
                                if r["head"]==task["goal"]))
            if task["label"]!="established":
                continue
            self.assertEqual(task["min_steps"],task["depth"])
            self.assertLessEqual(task["depth"],5)
            self.assertEqual(len(task["reachable"])-len(task["facts"]),10)
            self.assertGreaterEqual(len(goal_ancestors(task)),11)
            self.assertFalse(unary_shortest_trace(task,budget=5))
            self.assertTrue(proofs.verify_trace(
                task,task["shortest_trace"],budget=5)["verified"])

    def test_all_false_entailments_have_independent_boolean_witnesses(self):
        for task in self.rows:
            if task["label"]!="refuted":
                continue
            p=task["formal"]
            self.assertIsNotNone(find_countermodel(p["premises"],p["conclusion"]))
            self.assertTrue(verify_countermodel(
                p["premises"],p["conclusion"],
                task["countermodel"]["assignment"])["verified"])
            self.assertIsNone(task["shortest_trace"])

    def test_forward_training_syntax_excludes_answer_keys(self):
        for task in self.rows:
            public={key:task[key] for key in ("atoms","facts","rules","goal")}
            self.assertFalse(set(public)&{"label","family_id","split","min_steps",
                "countermodel","shortest_trace","reachable","formal","fingerprint"})

    def test_corrupted_record_rejected(self):
        original=deepcopy(self.rows)
        original[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            f.audit_train_dev(original,prior=set(f.previous_fingerprints()))
        original=deepcopy(self.rows)
        original[0]["fingerprint"]=self.rows[1]["fingerprint"]
        with self.assertRaises(AssertionError):
            f.audit_train_dev(original,prior=set(f.previous_fingerprints()))

    def test_one_shorter_proof_action_is_insufficient(self):
        task=next(x for x in self.rows if x["label"]=="established" and x["depth"]==4)
        self.assertFalse(proofs.verify_trace(
            task,task["shortest_trace"][:3],budget=5)["verified"])


if __name__=="__main__":
    unittest.main()
