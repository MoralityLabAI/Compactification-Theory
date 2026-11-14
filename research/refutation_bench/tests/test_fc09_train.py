"""FC-09 train-only AND/OR source exactness, family isolation and proof budget."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench.fc09 import train_data as fc09
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.fc07.posthoc_unary import unary_shortest_trace
from research.refutation_bench.fc06.posthoc_backward_slice import goal_ancestors
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC09SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fc09.generate_train_dev()

    def test_source_counts_and_family_isolation(self):
        report=fc09.audit_train_dev(self.data)
        self.assertEqual(report["n"],960)
        self.assertEqual((report["train_count"],report["dev_count"]),(768,192))
        self.assertEqual((report["positives"],report["negatives"]),(480,480))
        self.assertEqual(report["prior_fingerprint_overlap"],0)
        self.assertEqual(report["five_action_cap"],5)
        self.assertFalse(set(fc09.TRAIN_FAMILIES)&set(fc09.DEV_FAMILIES))
        for split,(_,_,__,___,families) in fc09.SPLITS.items():
            for family in families:
                cases=[x for x in self.data if x["split"]==split and
                       x["family_id"]==family]
                self.assertEqual(Counter(x["label"] for x in cases),
                                 {"established":len(cases)//2,"refuted":len(cases)//2})

    def test_every_positive_requires_and_has_minimal_checked_proof(self):
        for task in self.data:
            self.assertEqual(len(task["atoms"]),12)
            self.assertLessEqual(len(task["rules"]),20)
            self.assertTrue(all(len(x["premises"])>=2
                for x in task["rules"] if x["head"]==task["goal"]))
            self.assertEqual(unary_shortest_trace(task,budget=5),[])
            if task["label"]!="established":
                continue
            self.assertEqual(task["min_steps"],task["depth"])
            self.assertIn(task["depth"],(3,4))
            self.assertEqual(len(task["reachable"])-len(task["facts"]),10)
            self.assertGreaterEqual(len(goal_ancestors(task)),11)
            self.assertTrue(proofs.verify_trace(
                task,task["shortest_trace"],budget=5)["verified"])

    def test_every_invalid_has_exact_finite_countermodel(self):
        for task in self.data:
            if task["label"]!="refuted":
                continue
            p=task["formal"]
            self.assertIsNotNone(find_countermodel(p["premises"],p["conclusion"]))
            self.assertTrue(verify_countermodel(p["premises"],p["conclusion"],
                task["countermodel"]["assignment"])["verified"])
            self.assertIsNone(task["shortest_trace"])

    def test_graph_only_inputs_exclude_gold_and_reference_trace(self):
        for task in self.data:
            user_data={k:task[k] for k in ("atoms","facts","rules","goal")}
            self.assertFalse(set(user_data)&{"label","min_steps","shortest_trace",
                "family_id","split","countermodel","reachable","formal"})

    def test_bad_annotation_fails_without_repair(self):
        rows=deepcopy(self.data)
        rows[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            fc09.audit_train_dev(rows,prior=set(fc09.previous_fingerprints()))
        rows=deepcopy(self.data)
        rows[0]["fingerprint"]=rows[1]["fingerprint"]
        with self.assertRaises(AssertionError):
            fc09.audit_train_dev(rows,prior=set(fc09.previous_fingerprints()))

    def test_registered_seeds_and_parent_grammar_deterministic(self):
        self.assertEqual(fc09.ROOT_SEED,20261113)
        self.assertEqual(fc09.BUDGET,5)
        for family in fc09.TRAIN_FAMILIES:
            a=fc09.construct(123456,4,True,family,fc09._parents)
            b=fc09.construct(123456,4,True,family,fc09._parents)
            self.assertEqual(a,b)


if __name__=="__main__":
    unittest.main()
