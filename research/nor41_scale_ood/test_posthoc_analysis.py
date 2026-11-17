"""NOR-41 posthoc audit tests from synthetic receipts, never new source rows."""
from __future__ import annotations
import unittest
from research.nor41_scale_ood import posthoc_analysis as a

class PosthocSyntheticTests(unittest.TestCase):
    def synthetic(self):
        rows=[];recs=[]
        for rep in range(84,100):
            for d in (512,1024):
                for fam in ("wide_dense_small","wide_regime_blocks"):
                    sha=f"SYNTHETIC-{rep}-{d}-{fam}"
                    contrast=.004 if d==512 else .002
                    rows.append({
                        "replicate":rep,"degree":d,"family":fam,
                        "public_sha256":sha,
                        "original_native_mask_cardinality":1,
                        "global7_256_minus_4096_eval":contrast,
                        "global7_strong_self_minus_deterministic":.02,
                        "global7_strong_worst_minus_original":-.005,
                        "global7_top_minus_atom_random":.04,
                        "deterministic_top_minus_atom_random":.16,
                        "global7_budget_self":{
                            "256":.04,"512":.039,"2048":.038,"4096":.036},
                    })
                    cross={}
                    for att in [f"cal{n}__{p}" for n in a.BUDGETS
                                for p in a.POLICIES]+[
                                "top_curvature","bottom_curvature",
                                "eight_atom_matched","full_uniform",
                                "low_cal_exposure","zero_quota_stress"]:
                        cross[att]={p:256*8*1024*.03 for p in a.POLICIES}
                    cert={
                        str(n):{p:{
                            "global_optimality_gap":100,
                            "selected_attack_uniques":7
                        } for p in a.POLICIES} for n in a.BUDGETS}
                    recs.append({
                        "public_sha256":sha,
                        "crossplay_denominator":256*8*1024,
                        "crossplay_weighted_numerators":cross,
                        "attack_certificates":cert
                    })
        means={
            "global7_budget_transfer":.003,
            "degree512_minus_degree1024_transfer":.002,
            "ood_top_gain":.04,
            "global7_strong_self_minus_deterministic":.02,
            "global7_strong_worst_minus_original":-.005
        }
        return {
            "schema":"nor41.input_ood_nested_cal_transfer.v1",
            "registration_first_commit":a.REGISTERED,
            "predata_amendment_01_commit":a.AMENDMENT,
            "source_records":recs,"source_level_primary":rows,
            "fresh_source_units":64,"source_means":means,
            "registered_primary_pass":False,
            "registered_five_gates":{"synthetic_test_only":False},
            "source_direction_wins":{"synthetic_test_only":32},
            "retained_ood_top_gain_ratio":.25,
            "blind_control_families":[
                "top_curvature","bottom_curvature","eight_atom_matched",
                "full_uniform","low_cal_exposure","zero_quota_stress"],
        }

    def test_64_source_16_block_budget_crossplay(self):
        result=a.analyze(self.synthetic(),with_signflips=False)
        self.assertEqual(result["source_units"],64)
        self.assertEqual(result["replicate_blocks"],16)
        self.assertEqual(len(result["crossplay_22x4"]),22)
        self.assertEqual(len(result["budget_policy_means"]),4)
        self.assertEqual(result["degree_scale_block_contrast"]["positive_blocks"],16)
        self.assertAlmostEqual(result["degree_scale_block_contrast"]["mean"],.002)
        self.assertIn("Complete calibrated 22×4 crossplay",a.markdown(result))

    def test_invalid_source_duplicate_rejected(self):
        grade=self.synthetic()
        grade["source_records"][1]["public_sha256"]=(
            grade["source_records"][0]["public_sha256"])
        # Duplicate source rows remain their own observed records; disjoint-source
        # assertion belongs to the independent replay and source-first fixture.
        with self.assertRaises(AssertionError):
            a.analyze(grade,with_signflips=False)

    def test_symmetric_exact_signflips(self):
        self.assertEqual(a.signflip_two_sided([1]*16),2/(2**16))
        self.assertEqual(a.signflip_two_sided([0]*16),1)


if __name__=="__main__":
    unittest.main()
