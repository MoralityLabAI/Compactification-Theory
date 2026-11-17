"""Posthoc analyzer tests use only the byte-archived NOR38 prior grader."""
import unittest
from pathlib import Path
from research.nor39_clean_global_uniform import posthoc_analysis as a


class HistoricalReceiptPosthocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        f=root/"research/nor38_global_uniform/results/nor38_original_first_hosted.zip"
        cls.original=a.load(f)
        cls.analysis=a.describe(cls.original)

    def test_historical_rates_and_grouping(self):
        self.assertEqual(cls_len:=self.analysis["source_units"],48)
        self.assertEqual(self.analysis["independent_replicate_blocks"],8)
        self.assertEqual(len(self.analysis["strata"]["family"]),3)
        self.assertEqual(len(self.analysis["strata"]["degree"]),2)
        self.assertTrue(self.analysis["registered_primary_pass_as_recorded"])
        for p in a.POLICIES:
            vals=self.analysis["policy_rates"][p]
            self.assertGreater(vals["self"],0)
            self.assertGreater(vals["worst"],0)
            self.assertGreater(vals["top"],0)
        self.assertGreater(
            self.analysis["policy_rates"]["uniform16"]["worst"],
            self.analysis["policy_rates"]["global7_mix"]["worst"])

    def test_reconstruct_original_registered_source_means(self):
        analysis=self.analysis
        reported=self.original["source_means"]
        for key,_ in a.CONTRASTS:
            self.assertAlmostEqual(analysis["contrasts"][key]["mean"],
                                   reported[key],places=12)
        self.assertEqual(analysis["contrasts"]["global7_self_minus_deterministic_self"]["source_positive"],48)
        self.assertEqual(analysis["contrasts"]["global7_top_minus_atom_random"]["source_positive"],44)
        self.assertEqual(
            sum(v["count"] for v in analysis["strata"]["family"].values()),48)

    def test_markdown_explicit_exploratory_scope(self):
        rendered=a.as_markdown(self.analysis,"NOR-38 historical control")
        self.assertIn("exploratory source-level audit",rendered)
        self.assertIn("replicate-block",rendered)
        self.assertIn("uniform16",rendered)
        self.assertNotIn("NOR-39 original scientific PASS",rendered)


if __name__=="__main__":
    unittest.main()
