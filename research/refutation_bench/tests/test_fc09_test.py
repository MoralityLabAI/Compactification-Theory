"""FC-09 held-out exact AND/OR source, structural and timing controls."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench.fc09 import train_data,test_data,controls
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.fc07.posthoc_unary import unary_shortest_trace
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC09HeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=train_data.generate_train_dev()
        cls.records=test_data.generate_tests()

    def test_source_counts_and_nonoverlap(self):
        report=test_data.audit_tests(self.records,self.training)
        self.assertEqual(report["case_count"],384)
        self.assertEqual((report["test_same"],report["test_deep"]),(192,192))
        self.assertEqual((report["positives"],report["negatives"]),(192,192))
        self.assertEqual(report["earlier_WL_overlap"],0)
        self.assertFalse({x["family_id"] for x in self.training}&set(test_data.FAMILIES))
        for split in test_data.SPLITS:
            for fam in test_data.FAMILIES:
                cases=[x for x in self.records if x["split"]==split and x["family_id"]==fam]
                self.assertEqual(Counter(x["label"] for x in cases),
                                 {"established":24,"refuted":24})

    def test_all_positives_need_and_and_verify_under_five_actions(self):
        for x in self.records:
            self.assertEqual(len(x["atoms"]),12)
            self.assertLessEqual(len(x["rules"]),20)
            self.assertFalse(unary_shortest_trace(x,budget=5))
            self.assertTrue(all(len(r["premises"])>=2 for r in x["rules"] if r["head"]==x["goal"]))
            if x["label"]=="established":
                self.assertEqual(x["min_steps"],x["depth"])
                self.assertEqual(len(x["reachable"])-len(x["facts"]),10)
                self.assertTrue(proofs.verify_trace(x,x["shortest_trace"],budget=5)["verified"])

    def test_negative_exact_countermodels(self):
        for x in self.records:
            p=x["formal"]
            witness=find_countermodel(p["premises"],p["conclusion"])
            if x["label"]=="refuted":
                self.assertIsNotNone(witness)
                self.assertTrue(verify_countermodel(
                    p["premises"],p["conclusion"],x["countermodel"]["assignment"])["verified"])
            else:
                self.assertIsNone(witness)

    def test_hmax_lower_bound_is_admissible_on_valid_records(self):
        for x in self.records:
            if x["label"]!="established":
                continue
            n,known,goal,rules=controls._compile(x)
            cost=controls.hmax_lower_bound(n,known,goal,rules)
            self.assertLessEqual(cost,x["min_steps"])
            self.assertGreaterEqual(cost,0)

    def test_source_control_scores_frozen_before_model(self):
        report=test_data.audit_tests(self.records,self.training)
        for split in test_data.SPLITS:
            for name,score in report["pre_model_symbolic_controls"][split].items():
                self.assertEqual(score["valid_cases"],96)
                self.assertTrue(0<=score["certified_positive"]<=96)
                if name=="hmax_24":
                    self.assertLessEqual(score["search_states_popped"],24*96)
                if name=="unary_bfs":
                    self.assertEqual(score["certified_positive"],0)

    def test_per_case_zero_time_budget_enforced_and_logged(self):
        x=next(z for z in self.records if z["label"]=="established")
        result=controls.hmax_search(x,max_runtime_ms=0.)
        self.assertLessEqual(result["states_popped"],1)
        self.assertEqual(result["allowed_ms"],0.)
        self.assertGreaterEqual(result["budget_overrun_ms"],0)
        fixed=controls.hmax_search(x,max_popped=24)
        self.assertLessEqual(fixed["states_popped"],24)
        self.assertLessEqual(len(fixed["trace"]),5)

    def test_input_only_source_control_invariant_to_gold_poison(self):
        x=next(z for z in self.records if z["label"]=="established")
        y=deepcopy(x)
        y.update(label="refuted",shortest_trace=[-3],min_steps=999,
                 reachable=[],countermodel={"poison":True},family_id="changed")
        a=controls.hmax_search(x,max_popped=12)
        b=controls.hmax_search(y,max_popped=12)
        self.assertEqual(a["trace"],b["trace"])
        self.assertEqual(a["found_goal"],b["found_goal"])

    def test_source_tampering_fails(self):
        changed=deepcopy(self.records)
        changed[0]["fingerprint"]=self.training[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            test_data.audit_tests(changed,self.training)


if __name__=="__main__":
    unittest.main()
