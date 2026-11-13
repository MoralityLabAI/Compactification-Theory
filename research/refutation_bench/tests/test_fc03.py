"""Preregistered FC-03 corpus: exact truth tables, positive Lean terms, leak guard."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest
from research.refutation_bench.fc03 import generator as fc03
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import verify_countermodel, find_countermodel


class FC03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=fc02.generate()
        cls.rows=fc03.generate_tests()

    def test_balanced_fixed_family_splits(self):
        a=fc03.audit_tests(self.rows,self.old)
        self.assertEqual(a["splits"],{"test_same":192,"test_deep":192})
        self.assertEqual(a["distinct_fingerprints"],384)
        self.assertEqual(a["fc02_fingerprint_overlap"],0)
        self.assertEqual(a["positive_lean_theorems"],192)
        self.assertEqual(a["exact_countermodels"],192)
        self.assertEqual(len(a["families"]),4)
        for split in ("test_same","test_deep"):
            for family in fc03.FAMILIES:
                rows=[t for t in self.rows if t["split"]==split and t["family_id"]==family]
                self.assertEqual(len(rows),48)
                self.assertEqual(Counter(t["label"] for t in rows),
                                 {"established":24,"refuted":24})

    def test_withheld_family_separation(self):
        train_dev={t["family_id"] for t in self.old if t["split"] in ("train","dev")}
        self.assertTrue(train_dev.isdisjoint(fc03.FAMILIES))
        self.assertTrue(set(t["fingerprint"] for t in self.old).isdisjoint(
            t["fingerprint"] for t in self.rows))

    def test_certificates_against_independent_checker(self):
        for t in self.rows:
            formal=t["formal"]
            other=find_countermodel(formal["premises"],formal["conclusion"])
            if t["label"]=="refuted":
                self.assertIsNotNone(other)
                self.assertTrue(verify_countermodel(
                    formal["premises"],formal["conclusion"],
                    t["countermodel"]["assignment"])["verified"])
            else:
                self.assertIsNone(other)
                self.assertIsNone(t["countermodel"])
                self.assertIn("theorem "+t["case_id"].lower(),fc02.lean_theorem(t))

    def test_no_proof_fields_in_student_public_features(self):
        # Student interface supplies facts, edges, graph nodes and goal only.
        for t in self.rows:
            public={k:t[k] for k in ("atoms","facts","rules","goal")}
            self.assertFalse(set(public)&{
                "split","family_id","fingerprint","reachable",
                "countermodel","label","formal","depth"})

    def test_audit_detects_corrupted_gold_and_overlap(self):
        one=deepcopy(self.rows)
        one[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            fc03.audit_tests(one,self.old)
        one=deepcopy(self.rows)
        one[0]["fingerprint"]=self.old[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            fc03.audit_tests(one,self.old)

    def test_repeatable_bytes(self):
        second=fc03.generate_tests()
        self.assertEqual(
            hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in self.rows)).hexdigest(),
            hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in second)).hexdigest()
        )


if __name__=="__main__":
    unittest.main()
