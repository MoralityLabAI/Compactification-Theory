"""Regression tests for sourced historical proof-audit expansion."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest

from research.refutation_bench import bench
from research.refutation_bench.expansion import suite


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.gold, cls.sources = suite.load_combined()

    def test_combined_suite_validates_and_separates_sources(self):
        report = bench.validate_dataset(self.cases, self.gold, self.sources)
        self.assertEqual(report["case_count"], 25)
        self.assertEqual(report["family_count"], 11)
        self.assertEqual(report["splits"], {"dev": 19, "train": 6})
        family_split = {}
        for case in self.cases:
            family_split.setdefault(case["family_id"], case["split"])
            self.assertEqual(family_split[case["family_id"]], case["split"])

    def test_expansion_verdicts_have_valid_balanced_controls(self):
        pack = self.cases[15:]
        answers = {g["case_id"]: g for g in self.gold}
        self.assertEqual(len(pack), 10)
        labels = Counter(answers[c["case_id"]]["verdict"] for c in pack)
        self.assertEqual(labels, Counter(refuted=6, established=3, conditional=1))
        for case in pack:
            self.assertEqual(case["split"], "dev")
            self.assertTrue(all(e["locator"] and e["text"] for e in case["evidence"]))

    def test_reference_answers_validate_and_score(self):
        references = [g["reference_response"] for g in self.gold]
        scores = bench.evaluate(self.cases, self.gold, references)
        self.assertEqual(scores["metrics"]["verdict_accuracy"], 1.0)
        self.assertEqual(scores["metrics"]["supported_refutation_rate"], 1.0)

    def test_first_collatz_counterexample_is_exact(self):
        witness = suite.ansari_witness()
        self.assertEqual(witness["witness"], 11)
        self.assertIn(11, witness["lhs"])
        self.assertNotIn(11, witness["rhs"])
        self.assertEqual(witness["lhs"], [3, 7, 11, 15, 19, 23, 27, 31])
        self.assertEqual(witness["rhs"], [3, 7, 15, 19])

    def test_source_family_leak_fails_closed(self):
        cases = deepcopy(self.cases)
        one = next(c for c in cases if c["case_id"] == "RB-0016")
        one["split"] = "train"
        with self.assertRaises(bench.DataError):
            bench.validate_dataset(cases, self.gold, self.sources)

    def test_export_excludes_key_fields(self):
        records = [bench.prompt_record(c) for c in self.cases]
        sample = next(r for r in records if r["case_id"] == "RB-0024")
        public = sample["messages"][1]["content"]
        self.assertNotIn("reference_response", public)
        self.assertNotIn("rationale", public)
        self.assertNotIn("family_id", public)
        self.assertIn("11", public)

    def test_always_refuted_falsely_rejects_controls(self):
        answers = {g["case_id"]: g for g in self.gold}
        pack = self.cases[15:]
        controls = [c for c in pack if answers[c["case_id"]]["verdict"] != "refuted"]
        self.assertEqual(len(controls), 4)
        responses = [dict(case_id=c["case_id"], verdict="refuted", confidence=1.0,
                          findings=[], assertions=[]) for c in pack]
        score = bench.evaluate(pack, [answers[c["case_id"]] for c in pack], responses)
        self.assertEqual(score["metrics"]["supported_refutation_rate"], 0.0)
        self.assertEqual(score["metrics"]["nonrefuted_false_refutation_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
