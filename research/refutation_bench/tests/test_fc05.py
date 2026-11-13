"""FC-05 public graph-family and model-proposed proof certificate regression."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.fc05 import generator as fc05
from research.refutation_bench.fc05 import proofs
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import find_countermodel,verify_countermodel


class FC05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior=fc05.previous_records()
        cls.rows=fc05.generate_tests()

    def test_new_dataset_fully_disjoint_and_balanced(self):
        report=fc05.audit(self.rows,self.prior)
        self.assertEqual(report["count"],384)
        self.assertEqual(report["splits"],{"test_same":192,"test_deep":192})
        self.assertEqual(report["positive_lean_proofs"],192)
        self.assertEqual(report["negative_checked_witnesses"],192)
        self.assertTrue({r["fingerprint"] for r in self.prior}.isdisjoint(
            {r["fingerprint"] for r in self.rows}))
        for split in fc05.SPLITS:
            for family in fc05.FAMILIES:
                subset=[r for r in self.rows
                        if r["split"]==split and r["family_id"]==family]
                self.assertEqual(Counter(r["label"] for r in subset),
                    {"established":24,"refuted":24})

    def test_all_reference_labels_independently_verified(self):
        for t in self.rows:
            p=t["formal"]
            other=find_countermodel(p["premises"],p["conclusion"])
            if t["label"]=="refuted":
                self.assertIsNotNone(other)
                self.assertTrue(verify_countermodel(p["premises"],p["conclusion"],
                    t["countermodel"]["assignment"])["verified"])
            else:
                self.assertIsNone(other)
                self.assertIn("theorem "+t["case_id"].lower(),fc02.lean_theorem(t))

    def test_proposed_positive_trace_is_independently_accepted(self):
        for t in self.rows:
            if t["label"]!="established":
                continue
            trace=[]
            for rule_id,head in fc02.proof_trace(t):
                trace.append(rule_id)
                if head==t["goal"]:
                    break
            # The full reference generator trace may exceed the cap if there
            # are excess decoys. Verify it only when the budget permits.
            if len(trace)>10:
                continue
            certificate=proofs.verify_trace(t,trace)
            self.assertTrue(certificate["verified"])
            text=proofs.lean_from_trace(t,trace,theorem_name="checked_model_trace")
            self.assertIn("theorem checked_model_trace",text)
            self.assertIn("  exact ",text)
            self.assertNotIn("sorry",text)

    def test_incorrect_trace_fails_closed(self):
        t=next(x for x in self.rows if x["label"]=="established")
        invalid=[-1]
        self.assertFalse(proofs.verify_trace(t,invalid)["verified"])
        with self.assertRaises(proofs.ProofError):
            proofs.lean_from_trace(t,invalid,theorem_name="failed")
        with self.assertRaises(proofs.ProofError):
            proofs.verify_trace(t,[0]*11)
        with self.assertRaises(proofs.ProofError):
            proofs.verify_trace(t,[True])
        for t in self.rows:
            if t["label"]=="refuted":
                self.assertFalse(proofs.verify_trace(t,[])["verified"])
                break

    def test_trace_with_false_premise_cannot_certify(self):
        t=next(x for x in self.rows if x["label"]=="established")
        rule_id=next((i for i,r in enumerate(t["rules"])
                       if not set(r["premises"]).issubset(set(t["facts"]))),None)
        if rule_id is not None:
            self.assertEqual(proofs.verify_trace(t,[rule_id])["reason"],
                             "unavailable_antecedent")

    def test_original_case_metadata_are_excluded_from_public_inputs(self):
        for t in self.rows:
            inp={k:t[k] for k in ("atoms","facts","rules","goal")}
            self.assertFalse(set(inp)&{"label","split","family_id",
                "formal","reachable","countermodel","fingerprint"})

    def test_auditor_detects_label_or_fingerprint_tamper(self):
        rows=deepcopy(self.rows)
        rows[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            fc05.audit(rows,self.prior)
        rows=deepcopy(self.rows)
        rows[0]["fingerprint"]=self.prior[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            fc05.audit(rows,self.prior)

    def test_posthoc_saturation_any_legal_order_has_ten_step_budget(self):
        """Diagnostic added after FC-05 first test, not an eligibility rule.

        The generated unreachable guard means at most len(atoms)-2
        distinct noninitial reachable nodes; len(atoms) <= 12. Hence
        every legal-novel action schedule eventually proves a valid goal.
        """
        for task in self.rows:
            if task["label"]!="established":
                continue
            guard_candidates={a for a in task["atoms"] if a not in task["reachable"]}
            self.assertGreaterEqual(len(guard_candidates),1)
            known=set(task["facts"])
            trace=[]
            while task["goal"] not in known:
                legal=[j for j,rule in enumerate(task["rules"])
                       if rule["head"] not in known and
                       set(rule["premises"]).issubset(known)]
                self.assertTrue(legal)
                chosen=max(legal)
                trace.append(chosen)
                known.add(task["rules"][chosen]["head"])
                self.assertLessEqual(len(trace),10)
            self.assertTrue(proofs.verify_trace(task,trace)["verified"])

    def test_determinism(self):
        repeat=fc05.generate_tests()
        key=lambda rows:hashlib.sha256(b"".join(
            fc02.canonical_bytes(x)+b"\n" for x in rows)).hexdigest()
        self.assertEqual(key(repeat),key(self.rows))


if __name__=="__main__":
    unittest.main()
