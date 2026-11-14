"""Posthoc FC-06 zero-parameter source-slicing control; added after first model CI."""
from __future__ import annotations
from copy import deepcopy
import unittest

from research.refutation_bench.fc06 import test_data
from research.refutation_bench.fc06 import posthoc_backward_slice as bs
from research.refutation_bench.fc05 import proofs


class BackSliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks=test_data.generate_tests()

    def test_exact_certification_of_all_source_positive_graphs(self):
        report=bs.evaluate_records(self.tasks)
        self.assertEqual(report["test_positive"],192)
        self.assertEqual(report["test_negative"],192)
        self.assertEqual(report["certified_positive"],192)
        self.assertEqual(report["missed_positive"],0)
        self.assertEqual(report["false_certified_positive"],0)
        self.assertEqual(report["model_parameters"],0)
        self.assertEqual(report["model_recurrent_cell_calls"],0)
        self.assertEqual(report["counts_by_split"]["test_same"]["certified"],96)
        self.assertEqual(report["counts_by_split"]["test_deep"]["certified"],96)

    def test_every_positive_trace_is_valid_and_minimal(self):
        for task in self.tasks:
            result=bs.priority_trace(task,budget=5)
            check=proofs.verify_trace(task,result,budget=5)
            if task["label"]=="established":
                self.assertTrue(check["verified"])
                self.assertEqual(len(result),task["min_steps"])
            else:
                self.assertFalse(check["verified"])

    def test_input_only_backward_slice_ignores_gold_keys(self):
        task=next(t for t in self.tasks if t["label"]=="established")
        alternate=deepcopy(task)
        alternate["label"]="refuted"
        alternate["shortest_trace"]=[]
        alternate["reachable"]=[]
        alternate["countermodel"]={"poison":True}
        alternate["min_steps"]=999
        alternate["family_id"]="different"
        self.assertEqual(bs.goal_ancestors(task),bs.goal_ancestors(alternate))
        self.assertEqual(bs.priority_trace(task),bs.priority_trace(alternate))

    def test_strict_budget_still_rejects_deep_shortened_trace(self):
        sample=next(t for t in self.tasks
                    if t["label"]=="established" and t["depth"]==5)
        truncated=bs.priority_trace(sample,budget=4)
        self.assertEqual(len(truncated),4)
        self.assertFalse(proofs.verify_trace(sample,truncated,budget=4)["verified"])


if __name__=="__main__":
    unittest.main()
