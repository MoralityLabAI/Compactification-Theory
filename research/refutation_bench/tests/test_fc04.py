"""FC-04: exact source gate and new family isolation."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.fc04 import generator as fc04
from research.refutation_bench.fc03 import generator as fc03
from research.refutation_bench.formal_curriculum import generator as fc02
from research.refutation_bench.certificates import find_countermodel, verify_countermodel


class FC04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=fc02.generate()+fc03.generate_tests()
        cls.rows=fc04.generate_tests()

    def test_counts_and_preregistered_family_balance(self):
        self.assertEqual(fc04.SEED,20261113)
        self.assertEqual(fc04.PRE_REGISTRATION,
                         "ce7190129617d8387a7acf4ceed3e8d0570cfce4")
        result=fc04.audit_tests(self.rows,self.old)
        self.assertEqual(result["count"],384)
        self.assertEqual(result["split_counts"],{"test_same":192,"test_deep":192})
        self.assertEqual(result["positive_lean_theorems"],192)
        self.assertEqual(result["negative_checked_countermodels"],192)
        for split in fc04.SPLITS:
            for family in fc04.FAMILIES:
                items=[r for r in self.rows if r["family_id"]==family and r["split"]==split]
                self.assertEqual(Counter(x["label"] for x in items),
                                 {"established":24,"refuted":24})

    def test_isolation_from_both_predecessor_corpora(self):
        source_families={x["family_id"] for x in self.old}
        self.assertTrue(source_families.isdisjoint(fc04.FAMILIES))
        self.assertTrue({x["fingerprint"] for x in self.old}.isdisjoint(
            x["fingerprint"] for x in self.rows))

    def test_all_counterexamples_exact(self):
        for x in self.rows:
            p=x["formal"]
            counter=find_countermodel(p["premises"],p["conclusion"])
            if x["label"]=="refuted":
                self.assertIsNotNone(counter)
                self.assertTrue(verify_countermodel(
                    p["premises"],p["conclusion"],
                    x["countermodel"]["assignment"])["verified"])
            else:
                self.assertIsNone(counter)
                self.assertIn("theorem "+x["case_id"].lower(),fc02.lean_theorem(x))

    def test_no_answer_fields_in_model_input(self):
        for x in self.rows:
            inp={k:x[k] for k in ("atoms","facts","rules","goal")}
            self.assertFalse({"label","reachable","split","family_id",
                              "countermodel","fingerprint","formal"}&inp.keys())

    def test_sentinel_fail_closed(self):
        modified=deepcopy(self.rows)
        modified[0]["reachable"]=[]
        with self.assertRaises(AssertionError):
            fc04.audit_tests(modified,self.old)
        modified=deepcopy(self.rows)
        modified[0]["fingerprint"]=self.old[0]["fingerprint"]
        with self.assertRaises(AssertionError):
            fc04.audit_tests(modified,self.old)

    def test_reproducible_generator(self):
        repeat=fc04.generate_tests()
        self.assertEqual(
          hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in self.rows)).digest(),
          hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in repeat)).digest())


if __name__=="__main__":
    unittest.main()
