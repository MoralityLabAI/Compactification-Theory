"""Unit tests for a deliberately limited structural-comparison prototype."""
import copy
import json
from pathlib import Path
import unittest

from sch01 import AXES, compare, coarse_match, faithful_match, has_triad, typed_agreement, validate


class Sch01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(Path(__file__).with_name("fixtures.json").read_text(encoding="utf-8"))["records"]
        cls.lookup = {r["id"]: r for r in cls.records}

    def test_fixtures_valid(self):
        validate(self.records)

    def test_triad_counts(self):
        self.assertEqual(sum(map(has_triad, self.records)), 7)
        self.assertFalse(has_triad(self.lookup["STEADY"]))

    def test_forgotten_labels_produce_false_friend(self):
        self.assertTrue(coarse_match(self.lookup["JW"], self.lookup["KURZWEIL"]))
        self.assertFalse(faithful_match(self.lookup["JW"], self.lookup["KURZWEIL"]))
        self.assertLess(typed_agreement(self.lookup["JW"], self.lookup["KURZWEIL"]), 0.5)

    def test_synthetic_migration_same_coarse_path(self):
        self.assertTrue(coarse_match(self.lookup["JW"], self.lookup["MIGRATION"]))
        self.assertFalse(faithful_match(self.lookup["JW"], self.lookup["MIGRATION"]))

    def test_ontological_return_distinct_from_future(self):
        self.assertEqual(self.lookup["PLOTINUS"]["axes"]["time"], "ontological_reversion")
        self.assertNotEqual(self.lookup["PLOTINUS"]["axes"]["time"],
                            self.lookup["KURZWEIL"]["axes"]["time"])

    def test_nonduality_distinct_from_creator(self):
        self.assertNotEqual(self.lookup["ADVAITA"]["axes"]["ultimate"],
                            self.lookup["QURAN"]["axes"]["ultimate"])

    def test_symmetry(self):
        a, b = self.lookup["JW"], self.lookup["QURAN"]
        self.assertEqual(typed_agreement(a, b), typed_agreement(b, a))

    def test_identical_copy_has_faithful_isomorphism(self):
        a = self.lookup["JW"]
        self.assertTrue(faithful_match(a, copy.deepcopy(a)))
        self.assertEqual(typed_agreement(a, copy.deepcopy(a)), 1)

    def test_axis_change_breaks_faithful_isomorphism(self):
        a, b = self.lookup["JW"], copy.deepcopy(self.lookup["JW"])
        b["axes"]["agency"] = "technology"
        self.assertTrue(coarse_match(a, b))
        self.assertFalse(faithful_match(a, b))
        self.assertEqual(typed_agreement(a, b), (len(AXES) - 1) / len(AXES))

    def test_stable_state_is_no_pair(self):
        self.assertFalse(coarse_match(self.lookup["STEADY"], self.lookup["STEADY"]))
        self.assertIsNone(typed_agreement(self.lookup["STEADY"], self.lookup["JW"]))

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate([self.records[0], copy.deepcopy(self.records[0])])

    def test_reject_unanchored_historical_record(self):
        bad = copy.deepcopy(self.records)
        bad[0]["source_url"] = None
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            validate(bad)

    def test_reject_synthetic_disguised_as_primary(self):
        bad = copy.deepcopy(self.records)
        bad[6]["source_kind"] = "primary_doctrine"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate(bad)

    def test_deterministic_pair_count_and_no_claimed_evidence(self):
        report = compare(self.records)
        self.assertEqual(report["pair_count"], 28)
        self.assertEqual(report["coarse_positive_pairs"], 21)
        self.assertEqual(report["faithful_positive_pairs"], 0)
        self.assertIn("Bayesian evidence for metaphysics", report["inference_excluded"])
        self.assertEqual(report, compare(self.records))


if __name__ == "__main__":
    unittest.main()
