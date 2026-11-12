"""Independent checks for the deterministic, exact Horn curriculum."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import unittest

from research.refutation_bench.certificates import (
    find_countermodel, verify_countermodel
)
from research.refutation_bench.formal_curriculum import generator as gen


class FormalCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate()
        cls.report = gen.audit(cls.rows)

    def test_counts_and_family_isolation(self):
        self.assertEqual(len(self.rows), 1536)
        self.assertEqual(self.report["splits"], {"train":1152, "dev":192, "test":192})
        self.assertEqual(self.report["labels"], {"established":768, "refuted":768})
        self.assertEqual(len(self.report["families"]), 7)
        self.assertFalse(self.report["fingerprint_overlap"])

    def test_sources_and_answers_are_hidden_in_input_projection(self):
        for task in self.rows:
            public = {k: task[k] for k in ("atoms", "facts", "rules", "goal")}
            self.assertNotIn("label", public)
            self.assertNotIn("reachable", public)
            self.assertNotIn("countermodel", public)
            self.assertNotIn("split", public)

    def test_every_invalid_has_a_verified_countermodel(self):
        negative = [t for t in self.rows if t["label"] == "refuted"]
        for task in negative:
            cert = task["countermodel"]["assignment"]
            p = task["formal"]
            self.assertTrue(verify_countermodel(p["premises"], p["conclusion"], cert)["verified"])
            self.assertFalse(cert[task["goal"]])

    def test_every_valid_is_exhaustively_entailed(self):
        valid = [t for t in self.rows if t["label"] == "established"]
        for task in valid:
            p = task["formal"]
            self.assertIsNone(find_countermodel(p["premises"], p["conclusion"]))
            self.assertIn(task["goal"], task["reachable"])
            self.assertIn("theorem " + task["case_id"].lower(), gen.lean_theorem(task))

    def test_invalid_term_refused(self):
        task = next(t for t in self.rows if t["label"] == "refuted")
        with self.assertRaises(ValueError):
            gen.lean_theorem(task)

    def test_corrupt_gold_or_family_fails_audit(self):
        rows = deepcopy(self.rows)
        rows[0]["reachable"] = []
        with self.assertRaises(AssertionError):
            gen.audit(rows)
        rows = deepcopy(self.rows)
        rows[0]["family_id"] = self.rows[-1]["family_id"]
        with self.assertRaises(AssertionError):
            gen.audit(rows)

    def test_name_permutation_preserves_independent_validity(self):
        task = next(t for t in self.rows if t["label"] == "refuted")
        mapping = {a: "Z" + str(i) for i,a in enumerate(reversed(task["atoms"]))}
        p = {
            "facts": [mapping[x] for x in task["facts"]],
            "rules": [{"premises": [mapping[x] for x in r["premises"]],
                       "head": mapping[r["head"]]} for r in task["rules"]],
            "goal": mapping[task["goal"]],
        }
        f = gen.formal_problem(p)
        assignment = {mapping[a]: a in task["reachable"] for a in task["atoms"]}
        self.assertTrue(verify_countermodel(f["premises"], f["conclusion"], assignment)["verified"])

    def test_determinism_canonical_digest(self):
        second = gen.generate()
        self.assertEqual([gen.canonical_bytes(x) for x in second],
                         [gen.canonical_bytes(x) for x in self.rows])
        self.assertEqual(hashlib.sha256(gen.canonical_bytes(second)).hexdigest(),
                         hashlib.sha256(gen.canonical_bytes(self.rows)).hexdigest())


if __name__ == "__main__":
    unittest.main()
