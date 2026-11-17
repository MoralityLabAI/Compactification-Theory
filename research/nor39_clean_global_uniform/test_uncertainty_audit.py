"""Synthetic source-block math tests; NEVER constructs a NOR39 source program."""
from __future__ import annotations
from statistics import mean
import unittest

from research.nor39_clean_global_uniform import uncertainty_audit as a


class NOR39UncertaintyPreDataTests(unittest.TestCase):
    def fake_grade(self):
        rows=[]
        families=("wide_dense_small","wide_sparse_small","wide_rational_short")
        for rep in a.REPS:
            for d in (512,1024):
                for fam in families:
                    rows.append({
                        "replicate":rep,"degree":d,"family":fam,
                        "public_sha256":f"synthetic-sentinel-{rep}-{d}-{fam}",
                        "global7_self_minus_deterministic_self":.018+rep*1e-5,
                        "global7_worst_minus_original_worst":-.005+rep*1e-5,
                        "global7_top_minus_atom_random":.037+rep*1e-5,
                        "deterministic_top_minus_atom_random":.15})
        means={
            key:mean(row[key] for row in rows) for key in (
                "global7_self_minus_deterministic_self",
                "global7_worst_minus_original_worst",
                "global7_top_minus_atom_random",
                "deterministic_top_minus_atom_random")}
        return {
            "schema":"nor39.clean_global_uniform_policyaware.v1",
            "fresh_source_units":48,
            "source_level_primary":rows,
            "source_means":means,
            "retained_ratio_global7_to_deterministic_top_gain":
                means["global7_top_minus_atom_random"]/
                means["deterministic_top_minus_atom_random"],
            "registered_five_gates":{"synthetic_test_only":True},
            "source_direction_wins":{"synthetic_test_only":48}
        }

    def test_exact_eight_block_signflip_resolution(self):
        self.assertEqual(a.exact_signflip_p_greater([2]*8,0),1/256)
        self.assertEqual(a.exact_signflip_p_greater([0]*8,0),1)

    def test_percentile_and_leave_one_block_out(self):
        values=[1,2,3,4,5,6,7,8]
        self.assertEqual(a.percentile(values,0),1)
        self.assertEqual(a.percentile(values,1),8)
        report=a.block_stats(values,0,202611390091)
        self.assertEqual(len(report["leave_one_block_out_means"]),8)
        self.assertEqual(report["blocks_positive_to_threshold"],8)
        self.assertGreaterEqual(report["block_bootstrap_95_percentile_interval"][0],1)
        self.assertLessEqual(report["block_bootstrap_95_percentile_interval"][1],8)

    def test_synthetic_48_rows_no_fresh_source_execution(self):
        outcome=a.analyze(self.fake_grade())
        self.assertEqual(outcome["source_units"],48)
        self.assertEqual(outcome["source_replicate_blocks"],8)
        self.assertEqual(outcome["within_block_degree_family_cells"],6)
        self.assertEqual(len(outcome["comparisons"]),3)
        self.assertAlmostEqual(
            outcome["block_mean_ratios"]["global7_retained_fraction_top"],
            self.fake_grade()["retained_ratio_global7_to_deterministic_top_gain"])

    def test_incomplete_synthetic_replicate_is_rejected(self):
        grade=self.fake_grade()
        grade["source_level_primary"].pop()
        with self.assertRaises(AssertionError):
            a.analyze(grade)


if __name__=="__main__":
    unittest.main()
