"""CB02 protocol tests, independent of the Lean compiler."""
import tempfile
from pathlib import Path
import unittest

from transfer_bench_v1 import (ROOT, git_blob_sha, manifest, metrics,
                               verify_blob, verify_frozen_files)
from runner import classes, shortest_witness


class CB02TransferProtocolTests(unittest.TestCase):
    def test_git_blob_integrity_and_mutation(self):
        cfg = manifest()
        measured = verify_frozen_files(cfg)
        self.assertEqual(len(measured), 5)
        with tempfile.TemporaryDirectory() as folder:
            source = ROOT / "CB01/Transfer/Statements.lean"
            modified = Path(folder) / "Statements.lean"
            modified.write_bytes(source.read_bytes() + b" ")
            self.assertNotEqual(git_blob_sha(modified.read_bytes()),
                                cfg["frozen_git_blobs"]["CB01/Transfer/Statements.lean"])
            with self.assertRaises(ValueError):
                verify_blob(modified, cfg["frozen_git_blobs"]["CB01/Transfer/Statements.lean"])

    def test_manifest_references_the_exact_evalmath_commit(self):
        cfg = manifest()
        self.assertEqual(cfg["evalmath"]["ref"],
                         "2616092a2c73c083378e508abfd966fdf5e84814")
        self.assertEqual(len(cfg["targets"]), 3)
        self.assertEqual([t["id"] for t in cfg["targets"]],
                         ["T01_MENU", "T02_CODE", "T03_STEP"])

    def test_source_proof_metrics_and_dependency_separation(self):
        data = metrics()
        self.assertEqual(data["baseline_proof_count"], 3)
        self.assertEqual(data["candidate_proof_count"], 3)
        self.assertEqual(data["canonical_helper_theorem_count"], 4)
        self.assertEqual(data["statement_identity_checks"], 6)
        self.assertLess(data["candidate_proof_body_bytes"],
                        data["baseline_proof_body_bytes"])
        self.assertNotIn("CB01.Transfer.Baseline",
                         data["candidate_import_closure"])

    def test_example_timed_distinguishing_trace(self):
        self.assertEqual(classes(3), [[0, 2], [1, 3], [4], [5]])
        self.assertIsNone(shortest_witness(0, 4, 2))
        self.assertEqual(shortest_witness(0, 4, 3), ["switch", "read"])

    def test_holdout_is_phase_marked(self):
        data = metrics()
        holds = ROOT / "CB01/Transfer/Heldout.lean"
        self.assertEqual(data["heldout_theorem_count"], 4 if holds.exists() else 0)
        if holds.exists():
            cfg = manifest()
            self.assertEqual(verify_blob(holds, cfg["heldout_git_blob"]),
                             cfg["heldout_git_blob"])


if __name__ == "__main__":
    unittest.main()
