"""FC-06 preregistered train-only proof budget, proof families and exact witnesses."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import random
import unittest

from research.refutation_bench.fc06 import train_data as f
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import verify_countermodel,find_countermodel


class TrainSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=f.generate_train_dev()

    def test_preregistered_splits_and_source_graphs(self):
        r=f.audit_train_dev(self.rows)
        self.assertEqual((r["train_count"],r["dev_count"]),(768,192))
        self.assertEqual((r["positive"],r["negative"]),(480,480))
        self.assertEqual(r["distinct_WL_graphs"],960)
        self.assertEqual(r["budget"],5)
        self.assertEqual(r["positive_novel_closure"],10)

    def test_binding_budget_and_reference_legal_proofs(self):
        for x in self.rows:
            self.assertEqual(len(x["atoms"]),12)
            self.assertLessEqual(len(x["rules"]),20)
            if x["label"]!="established":
                continue
            self.assertEqual(x["min_steps"],x["depth"])
            self.assertLessEqual(x["min_steps"],5)
            self.assertEqual(len(x["reachable"])-len(x["facts"]),10)
            self.assertTrue(proofs.verify_trace(x,x["shortest_trace"],budget=5)["verified"])
            self.assertIn("theorem "+x["case_id"].lower(),fc02.lean_theorem(x))

    def test_all_negatives_have_exact_refuting_assignments(self):
        for x in self.rows:
            if x["label"]!="refuted":
                continue
            p=x["formal"]
            self.assertIsNotNone(find_countermodel(p["premises"],p["conclusion"]))
            self.assertTrue(verify_countermodel(
                p["premises"],p["conclusion"],x["countermodel"]["assignment"])["verified"])
            self.assertIsNone(x["shortest_trace"])

    def test_source_input_whitelists_no_certificate_gold(self):
        for x in self.rows:
            public={k:x[k] for k in ("atoms","rules","facts","goal")}
            self.assertFalse({"shortest_trace","label","min_steps","split",
                              "family_id","reachable","countermodel"}&public.keys())

    def test_exact_checker_rejects_budget_truncation(self):
        task=next(x for x in self.rows
                  if x["label"]=="established" and x["min_steps"]==4)
        true_trace=task["shortest_trace"]
        self.assertFalse(proofs.verify_trace(task,true_trace[:3],budget=5)["verified"])
        with self.assertRaises(proofs.ProofError):
            proofs.verify_trace(task,true_trace+[true_trace[0],true_trace[0]],budget=5)

    def test_random_legal_policy_is_nontrivially_underpowered_on_train(self):
        positives=[x for x in self.rows if x["label"]=="established" and x["split"]=="train"]
        rng=random.Random(17013)
        success=0
        for task in positives:
            known=set(task["facts"])
            for _ in range(5):
                if task["goal"] in known:
                    break
                candidates=[rule for rule in task["rules"]
                            if rule["head"] not in known and set(rule["premises"])<=known]
                if not candidates:
                    break
                selected=rng.choice(candidates)
                known.add(selected["head"])
            success+=int(task["goal"] in known)
        self.assertLess(success/len(positives),.90)

    def test_deterministic_task_and_bad_metadata_fail_closed(self):
        raw1=f._make(12345,4,True,"fork_decoys")
        raw2=f._make(12345,4,True,"fork_decoys")
        self.assertEqual(f.canonical_bytes(raw1),f.canonical_bytes(raw2))
        changed=deepcopy(self.rows)
        changed[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            f.audit_train_dev(changed,prior=set(f.older_fingerprints()))


if __name__=="__main__":
    unittest.main()
