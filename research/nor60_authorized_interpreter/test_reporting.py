"""Source-free checks for first-failure preservation and reporter provenance."""
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from research.nor60_authorized_interpreter import benchmark_cpu as benchmark


class FirstEvidenceReportingTests(unittest.TestCase):
    def test_journal_chain_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            journal = benchmark.Journal(path)
            journal.write("source_started", source_ordinal=1)
            journal.write("timed_arm_completed", elapsed_ms=3.5)
            journal.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            previous = "0" * 64
            for ordinal, record in enumerate(records):
                digest = record.pop("event_sha256")
                self.assertEqual(record["ordinal"], ordinal)
                self.assertEqual(record["previous_sha256"], previous)
                self.assertEqual(sha256(benchmark.canonical(record)).hexdigest(), digest)
                previous = digest
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                benchmark.Journal(path)
            self.assertEqual(path.read_bytes(), before)

    def test_fatal_report_keeps_partial_evidence_and_blocks_same_path_retry(self):
        def toy_failure(public_file, journal, repeats):
            journal.write("timed_arm_completed", source_ordinal=1,
                          repetition=0, arm=benchmark.ARMS[0], elapsed_ms=3.5)
            raise RuntimeError("deliberate source-free reporting probe")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "toy.json"
            argv = ["benchmark", "--public-file", str(Path(directory) / "unused.json"), "--out", str(output)]
            with patch("sys.argv", argv), patch.object(benchmark, "run", side_effect=toy_failure) as run:
                with self.assertRaisesRegex(RuntimeError, "source-free"):
                    benchmark.main()
                self.assertEqual(run.call_count, 1)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    benchmark.main()
                self.assertEqual(run.call_count, 1)
            journal = output.with_suffix(".progress.jsonl")
            failure = json.loads(output.with_suffix(".failure.json").read_text())
            self.assertFalse(output.exists())
            self.assertFalse(failure["first_original_engineering_pass"])
            self.assertEqual(failure["progress_journal_sha256"], sha256(journal.read_bytes()).hexdigest())
            self.assertTrue(failure["rerun_is_not_original_first_attempt"])
            self.assertEqual([json.loads(line)["event"] for line in journal.read_text().splitlines()],
                             ["timed_arm_completed", "experiment_fatal_failure"])


if __name__ == "__main__":
    unittest.main()
