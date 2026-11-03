import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.algernon.inventory.audit_inventory import audit_inventory


class InventoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.digest = "0" * 64
        self.inventory = {
            "count": 2,
            "records": [
                {"experiment": f"run{i}", "file": "state.pt", "bytes": 100,
                 "sha256": self.digest,
                 "structure": {"tensors": {"block.weight": [2, 4]}}}
                for i in range(2)
            ],
        }

    def test_dedup_and_no_unverified_promotion(self):
        result = audit_inventory(self.inventory)
        self.assertEqual(result["inventory_files"], 2)
        self.assertEqual(result["distinct_checkpoints"], 1)
        self.assertEqual(result["duplicate_aliases"], 1)
        self.assertEqual(result["source_verified_graft_candidates"], 0)
        self.assertEqual(result["checkpoint_groups"][0]["parameter_count"], 8)

    def test_bad_shape_fails(self):
        self.inventory["records"][0]["structure"]["tensors"]["block.weight"] = [2, 0]
        with self.assertRaises(ValueError):
            audit_inventory(self.inventory)

    def test_inconsistent_same_sha_fails(self):
        self.inventory["records"][1]["bytes"] = 101
        with self.assertRaises(ValueError):
            audit_inventory(self.inventory)

    def test_source_verified_receipt(self):
        with tempfile.TemporaryDirectory() as p:
            root = Path(p)
            (root / "experiment.log").write_bytes(b"measured on heldout")
            receipt = {
                "checkpoint_sha256": self.digest, "source_log_path": "experiment.log",
                "source_log_sha256": hashlib.sha256(b"measured on heldout").hexdigest(),
                "code_revision": "a" * 40, "dataset_id": "synthetic",
                "dataset_sha256": "1" * 64, "split_sha256": "2" * 64,
                "optimizer_recipe": {"name": "Adam", "lr": 0.001},
                "recurrence_schedule": {"steps": [1, 2, 4, 8]},
                "evaluation": {"split": "held_out", "n": 100, "metric": "accuracy", "value": 0.5},
            }
            result = audit_inventory(self.inventory, [receipt], root)
            self.assertEqual(result["source_verified_graft_candidates"], 1)
            receipt["source_log_sha256"] = "f" * 64
            self.assertEqual(audit_inventory(self.inventory, [receipt], root)["source_verified_graft_candidates"], 0)

    def test_escape_root_rejected(self):
        receipt = {"checkpoint_sha256": self.digest}
        self.assertEqual(audit_inventory(self.inventory, [receipt])["source_verified_graft_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
