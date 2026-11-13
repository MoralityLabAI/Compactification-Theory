"""Post-hoc stronger FC-07 unary-path control. No test-label supervision."""
from __future__ import annotations
from copy import deepcopy
import unittest
from research.refutation_bench.fc07 import test_data,posthoc_unary
from research.refutation_bench.fc05 import proofs


class UnaryControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=test_data.generate_tests()

    def test_complete_positive_certification(self):
        result=posthoc_unary.evaluate(self.items)
        self.assertEqual(result["n"],384)
        self.assertEqual(result["source_positive"],192)
        self.assertEqual(result["certified_positive"],192)
        self.assertEqual(result["wrong_positive_certificates"],0)
        self.assertEqual(result["trainable_parameters"],0)
        self.assertEqual(result["neural_recurrent_calls"],0)
        for split in ("test_same","test_deep"):
            self.assertEqual(result["by_split"][split],
                {"valid":96,"certified":96,"missed":0})

    def test_original_rule_indices_and_five_action_budget(self):
        for task in self.items:
            proposed=posthoc_unary.unary_shortest_trace(task)
            result=proofs.verify_trace(task,proposed,budget=5)
            if task["label"]=="established":
                self.assertTrue(result["verified"])
                self.assertEqual(len(proposed),task["min_steps"])
            else:
                self.assertFalse(result["verified"])

    def test_source_answer_and_metadata_poison_do_not_affect_proposal(self):
        task=self.items[0]
        edited=deepcopy(task)
        edited["label"]="refuted" if task["label"]=="established" else "established"
        edited["shortest_trace"]=[999]
        edited["min_steps"]=999
        edited["countermodel"]={"do_not_use":True}
        edited["family_id"]="fake"
        edited["reachable"]=[]
        edited["split"]="fake"
        self.assertEqual(posthoc_unary.unary_shortest_trace(task),
                         posthoc_unary.unary_shortest_trace(edited))

    def test_one_missing_rule_disables_chain_certificate(self):
        task=next(t for t in self.items
                  if t["label"]=="established" and t["min_steps"]==5)
        trace=posthoc_unary.unary_shortest_trace(task)
        self.assertEqual(len(trace),5)
        broken=deepcopy(task)
        idx=trace[2]
        # Remove a backbone unary rule and ask the restricted path
        # policy to find a proof without repairing the graph.
        broken["rules"].pop(idx)
        # Other unary shortest paths cannot prove the goal in this source
        # construction: conjunctive alternative producers are omitted.
        self.assertFalse(proofs.verify_trace(
            broken,posthoc_unary.unary_shortest_trace(broken),budget=5)["verified"])


if __name__=="__main__":
    unittest.main()
