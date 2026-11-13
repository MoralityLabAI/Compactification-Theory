"""The Phase 5 roll-up (results/phase4/metrics.json) is reproducible from the committed Phase 4 receipts."""
import io
import unittest
from contextlib import redirect_stdout

from research.algernon.circuit_mirror.lotus_bridge import phase4_summary


class Phase4SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = phase4_summary.build()

    def test_committed_metrics_match_the_receipts(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(phase4_summary.main(["--check"]), 0)

    def test_every_run_is_scored_on_the_same_records(self):
        for run in self.summary["runs"]:
            self.assertEqual(run["records"]["count"], 5472, run["run"])
            for arm in "PCIN":
                self.assertEqual(run["records"]["per_arm_condition"][arm],
                                 {"clean": 72, "primary": 864, "half": 216, "double": 216}, run["run"])

    def test_grid_cells_and_their_decisions(self):
        grid = self.summary["grid_noise_by_loss"]
        self.assertEqual({c: g["met"] for c, g in grid.items()},
                         {"sigma16_local": [], "sigma16_answer": [], "sigma1_local": ["R4"], "sigma1_answer": ["R1", "R2", "R3"]})
        settings = {r["run"]: r["settings"] for r in self.summary["runs"]}
        for cell, g in grid.items():
            sigma, loss = cell.split("_")
            self.assertEqual(settings[g["run"]]["sigma_primary"], float(sigma.removeprefix("sigma")))
            self.assertEqual(settings[g["run"]]["loss"], loss)
            self.assertEqual(settings[g["run"]]["noise_ref"], "fixed")

    def test_plain_arm_is_the_same_within_each_noise_level(self):
        acc = {r["run"]: r["accuracy"]["P"] for r in self.summary["runs"]}
        self.assertEqual(acc["algernon-03d"], acc["algernon-03g"])
        self.assertEqual(acc["algernon-03e"], acc["algernon-03f"])


if __name__ == "__main__":
    unittest.main()
