"""NOR-40 read-only report fixtures, synthetic only, no new source generation."""
from __future__ import annotations
import unittest
from fractions import Fraction
from research.nor40_strong_adversaries import experiment as x
from research.nor40_strong_adversaries import posthoc_analysis as a


class NOR40PosthocTests(unittest.TestCase):
    def synthetic(self):
        rows=[]
        for rep in range(76,84):
            for degree in (512,1024):
                for family in x.FAMILIES:
                    cross={}
                    for attacker in list(x.POLICIES)+[
                        "baseline__"+p for p in x.POLICIES]+list(x.CONTROLS):
                        cross[attacker]={}
                        for defender in x.POLICIES:
                            cross[attacker][defender]=round(4096*256*(
                                .035+(.004 if attacker.startswith("baseline__") else 0)
                                +(.001 if defender=="original_mix" else 0)))
                    cert={p:{
                        "baseline_eight_full_cal_sum":200000,
                        "strong_eight_full_cal_sum":170000,
                        "global_singleton_lower_bound":2000,
                        "global_best_found_upper_bound":15000,
                        "global_optimality_gap_upper_minus_lower":13000,
                        "restricted_exact_value":16000,
                        "restricted_sets_enumerated":1820,
                        "strong_unique_count":8
                    } for p in x.POLICIES}
                    tops={p:[28,100] for p in x.POLICIES}
                    tops["atom_random_h50"]=[18,100]
                    tops["deterministic_mix"]=[45,100]
                    tops["global7_mix"]=[27,100]
                    rows.append({
                        "replicate":rep,"degree":degree,"family":family,
                        "public_sha256":f"synthetic-{rep}-{degree}-{family}",
                        "crossplay_weighted_numerators":cross,
                        "crossplay_denominator":4096*256,
                        "exact_top_support_detection":tops,
                        "attack_certificates":cert})
        derived=x.measures(rows)
        means={k:sum(v[k] for v in derived)/48 for k in (
            "global7_baseline_minus_strong_selfplay",
            "global7_strong_self_minus_deterministic_strong",
            "global7_strong_worst_minus_original_strong_worst",
            "global7_top_minus_atom_random",
            "deterministic_top_minus_atom_random")}
        return {
            "schema":"nor40.strong_calibration_policyaware.v1",
            "registration_first_commit":x.REGISTERED,
            "source_records":rows,"fresh_source_units":48,
            "auditor_policies":list(x.POLICIES),
            "blind_control_families":list(x.CONTROLS),
            "source_means":means,
            "registered_primary_pass":False,
            "registered_five_gates":{"synthetic_data":False}
        }

    def test_reports_complete_crossplay_and_eight_blocks(self):
        report=a.describe(self.synthetic())
        self.assertEqual(report["schema"],"nor40.original_posthoc_replicate_block_analysis.v1")
        self.assertEqual(report["per_source_units"],48)
        self.assertEqual(report["replicate_blocks"],8)
        self.assertEqual(len(report["calibrated_20x7_crossplay"]),20)
        self.assertTrue(all(len(x)==7 for x in
                            report["calibrated_20x7_crossplay"].values()))
        self.assertEqual(len(report["policy_means"]),7)
        self.assertIn("global7_strong_self_minus_deterministic_strong",
                      report["replicate_block_descriptors"])
        self.assertIn("Exact 20×7",a.markdown(report))

    def test_rejects_missing_synthetic_identity(self):
        bad=self.synthetic()
        bad["source_records"][0]["public_sha256"]=bad["source_records"][1]["public_sha256"]
        with self.assertRaises(AssertionError):
            a.describe(bad)


if __name__=="__main__":
    unittest.main()
