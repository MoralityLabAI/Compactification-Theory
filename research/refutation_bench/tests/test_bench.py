"""Behavioral checks for leakage, shortcut scoring and unsupported allegations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.refutation_bench import bench


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.gold, cls.sources = bench.load_dataset()

    def setUp(self):
        self.responses = [deepcopy(g["reference_response"]) for g in self.gold]

    def score(self, responses):
        return bench.evaluate(self.cases, self.gold, responses)

    def test_pinned_sources_and_family_split_integrity(self):
        report = bench.validate_dataset(self.cases, self.gold, self.sources)
        self.assertFalse(report["family_split_overlap"])
        real = [c for c in self.cases if c["family_id"] == "nielsen_semita_rh_20261112"]
        self.assertGreater(len(real), 1)
        self.assertEqual({c["split"] for c in real}, {"dev"})
        self.assertNotIn("test", report["splits"])

    def test_cross_split_derivative_leak_rejected(self):
        cases = deepcopy(self.cases)
        family = cases[0]["family_id"]
        self.assertGreater(sum(c["family_id"] == family for c in cases), 1)
        cases[0]["split"] = "train"
        with self.assertRaisesRegex(bench.DataError, "leaks across splits"):
            bench.validate_dataset(cases, self.gold, self.sources)

    def test_prompt_export_whitelist_excludes_answer_material(self):
        case = deepcopy(self.cases[0])
        marker = "PRIVATE_GOLD_CANARY_92cda"
        case.update(gold=marker, rationale=marker, reference_response=marker)
        case["family_id"] = marker
        prompt = bench.prompt_record(case)
        serialized = json.dumps(prompt)
        self.assertNotIn(marker, serialized)
        content = json.loads(prompt["messages"][1]["content"])
        self.assertEqual(set(content), {"case_id", "task", "claim", "evidence"} |
                         ({"formal"} if "formal" in case else set()))

    def test_gold_fixture_has_full_annotation_score(self):
        report = self.score(self.responses)
        self.assertEqual(report["metrics"]["verdict_accuracy"], 1)
        self.assertEqual(report["metrics"]["verdict_macro_f1"], 1)
        self.assertEqual(report["metrics"]["finding_precision"], 1)
        self.assertEqual(report["metrics"]["supported_refutation_rate"], 1)
        self.assertEqual(report["metrics"]["prohibited_assertion_count"], 0)
        self.assertTrue(report["human_semantic_review_required"])

    def test_missing_responses_stay_in_denominator(self):
        report = self.score([])
        self.assertEqual(report["metrics"]["verdict_accuracy"], 0)
        self.assertEqual(report["metrics"]["response_coverage"], 0)
        self.assertEqual(len(report["cases"]), len(self.cases))

    def test_duplicate_and_unknown_ids_fail_closed(self):
        with self.assertRaisesRegex(bench.DataError, "Duplicate"):
            self.score(self.responses + [deepcopy(self.responses[0])])
        self.responses[0]["case_id"] = "DOES_NOT_EXIST"
        with self.assertRaisesRegex(bench.DataError, "unknown or out-of-split"):
            self.score(self.responses)

    def test_correct_verdict_without_finding_is_unsupported(self):
        for response in self.responses:
            response["findings"] = []
        metrics = self.score(self.responses)["metrics"]
        self.assertEqual(metrics["verdict_accuracy"], 1)
        self.assertEqual(metrics["supported_refutation_rate"], 0)

    def test_repeating_finding_cannot_gain_credit(self):
        row = next(r for r in self.responses if r["findings"])
        row["findings"].append(deepcopy(row["findings"][0]))
        report = self.score(self.responses)
        result = next(r for r in report["cases"] if r["case_id"] == row["case_id"])
        self.assertEqual(result["unmatched_finding_count"], 1)
        self.assertFalse(result["supported_refutation"])
        self.assertLess(report["metrics"]["finding_precision"], 1)

    def test_shotgun_evidence_does_not_localize_flaw(self):
        response = self.responses[0]
        response["findings"] = [deepcopy(response["findings"][0])]
        response["findings"][0]["evidence_ids"] = [e["evidence_id"] for e in self.cases[0]["evidence"]]
        row = self.score(self.responses)["cases"][0]
        self.assertEqual(row["matched_finding_ids"], [])
        self.assertFalse(row["supported_refutation"])

    def test_unsupported_stronger_claim_removes_supported_credit(self):
        self.responses[0]["assertions"] = ["rh_false", "kernel_axioms_printed"]
        row = self.score(self.responses)["cases"][0]
        self.assertTrue(row["verdict_correct"])
        self.assertFalse(row["supported_refutation"])
        self.assertEqual(len(row["prohibited_assertions"]), 2)

    def test_unknown_evidence_is_schema_error(self):
        self.responses[0]["findings"][0]["evidence_ids"] = ["GHOST"]
        row = self.score(self.responses)["cases"][0]
        self.assertFalse(row["schema_valid"])
        self.assertFalse(row["verdict_correct"])

    def test_confidence_bool_and_nan_are_rejected(self):
        for value in (True, float("nan"), -1, 1.01, 10 ** 1000):
            response = deepcopy(self.responses[0])
            response["confidence"] = value
            self.assertTrue(bench.validate_response(response, self.cases[0]))

    def test_conditional_controls_penalize_unconditional_refutation(self):
        rows = [r for r in self.responses if r["verdict"] in ("established", "conditional")]
        self.assertGreater(len(rows), 0)
        for row in rows:
            row["verdict"] = "refuted"
        self.assertEqual(self.score(self.responses)["metrics"]["valid_control_false_refutation_rate"], 1)

    def test_invalid_countermodel_cannot_support_refutation(self):
        response = next(r for r in self.responses if r.get("certificate", {}).get("kind") == "countermodel")
        response["certificate"]["assignment"] = {}
        row = next(r for r in self.score(self.responses)["cases"] if r["case_id"] == response["case_id"])
        self.assertFalse(row["formal_certificate"]["verified"])
        self.assertFalse(row["supported_refutation"])

    def test_actual_utf8_bytes_not_character_or_token_estimate(self):
        response = self.responses[0]
        response["findings"][0]["explanation"] = "π ≠ 证明"
        row = self.score(self.responses)["cases"][0]
        self.assertEqual(row["response_utf8_bytes"], len(bench.canonical_bytes(response)))
        self.assertGreater(row["response_utf8_bytes"], len(bench.canonical_bytes(response).decode("utf-8")))

    def test_empty_and_duplicate_json_objects_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text('{"case_id":"a","case_id":"b"}\n')
            with self.assertRaisesRegex(bench.DataError, "Duplicate JSON key"):
                bench.read_jsonl(path)
            path.write_text('[]\n')
            with self.assertRaisesRegex(bench.DataError, "must be an object"):
                bench.read_jsonl(path)

    def test_source_hash_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(b"source")
            manifest = {"sources": [{"source_id": "x", "filename": "source.txt",
                                      "sha256": hashlib.sha256(b"source").hexdigest()}]}
            self.assertTrue(bench.verify_sources(manifest, Path(directory))["all_matched"])
            path.write_bytes(b"changed")
            self.assertFalse(bench.verify_sources(manifest, Path(directory))["all_matched"])

    def test_source_paths_cannot_escape(self):
        manifest = {"sources": [{"source_id": "x", "filename": "../outside", "sha256": "0" * 64}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(bench.DataError, "escapes"):
                bench.verify_sources(manifest, Path(directory))

    def test_minimal_fixture_preserves_decisive_credit_and_costs_less(self):
        receipt = bench.run_baselines(self.cases, self.gold)
        full = receipt["methods"]["reference_fixture"]["metrics"]
        short = receipt["methods"]["minimal_reference_fixture"]["metrics"]
        self.assertEqual(short["supported_refutation_rate"], 1)
        self.assertLess(short["mean_response_utf8_bytes"], full["mean_response_utf8_bytes"])
        self.assertLess(short["finding_recall"], full["finding_recall"])
        self.assertFalse(receipt["model_training_performed"])

    def test_macro_f1_uses_classes_present_in_selected_split(self):
        cases = [c for c in self.cases if c["split"] == "train"]
        ids = {c["case_id"] for c in cases}
        gold = [g for g in self.gold if g["case_id"] in ids]
        result = bench.evaluate(cases, gold, [g["reference_response"] for g in gold])
        self.assertEqual(result["metrics"]["verdict_macro_f1"], 1)


if __name__ == "__main__":
    unittest.main()
