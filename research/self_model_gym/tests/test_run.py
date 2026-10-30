import json
import tempfile
import unittest
from pathlib import Path

from research.self_model_gym.backends import ScriptedBackend
from research.self_model_gym.protocol import CONDITIONS
from research.self_model_gym.run import evaluate
from research.self_model_gym.verify import verify


class TestRun(unittest.TestCase):
    def test_receipt_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            receipt = evaluate(ScriptedBackend(), replicates=2, root_seed=23,
                               output=out, save_traces=True)
            self.assertEqual(receipt["episodes"], 2 * len(CONDITIONS))
            self.assertEqual(receipt["paired_n"], 2)
            self.assertAlmostEqual(receipt["primary_paired_difference"], 0.0)
            self.assertEqual(verify(out)["activation_metrics_recomputed"], True)
            with self.assertRaises(FileExistsError):
                evaluate(ScriptedBackend(), replicates=2, root_seed=23, output=out)

    def test_run_without_trace_and_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            receipt = evaluate(ScriptedBackend(), replicates=1, root_seed=3, output=out)
            self.assertEqual(verify(out)["activation_metrics_recomputed"], False)
            artifact = out / "episodes.jsonl"
            artifact.write_text(artifact.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify(out)

    def test_bounded_trial_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluate(ScriptedBackend(), replicates=0, root_seed=5, output=Path(tmp)/"run")


if __name__ == "__main__":
    unittest.main()
