"""Offline registration checks; never access a model or GPU."""
import copy
import json
import unittest
from pathlib import Path
from research.algernon.validate_register import REGISTER, validate


class RegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(Path(REGISTER).read_text(encoding="utf-8"))

    def test_registration_passes(self):
        self.assertEqual(validate(copy.deepcopy(self.data)), [])

    def test_cannot_claim_results_before_execution(self):
        d = copy.deepcopy(self.data)
        d["observed_results"] = [{"accuracy": 1}]
        self.assertTrue(any("cannot claim executed" in s for s in validate(d)))

    def test_sources_require_full_sha_or_unpinned(self):
        d = copy.deepcopy(self.data)
        d["sources"][0]["revision"] = "not-a-git-sha"
        self.assertTrue(any("invalid source SHA" in s for s in validate(d)))

    def test_controls_required(self):
        d = copy.deepcopy(self.data)
        d["hypotheses"][0]["required_controls"] = ["base"]
        self.assertTrue(any("distinct controls" in s for s in validate(d)))

    def test_hypothesis_ids_unique(self):
        d = copy.deepcopy(self.data)
        d["hypotheses"][3]["id"] = "H-C"
        self.assertTrue(any("hypothesis IDs" in s for s in validate(d)))


if __name__ == "__main__":
    unittest.main()
