"""NOR42 posthoc audit tests on synthetic receipts only; no new source IDs."""
from __future__ import annotations
import unittest
from research.nor42_paired_program_ood.prospective import posthoc_analysis as audit

class NOR42PosthocSynthetic(unittest.TestCase):
    def data(self):
        rec=[];rows=[]
        den=256*1024
        for rep in range(100,116):
            for degree in (512,1024):
                for family in ("wide_dense_small","wide_regime_blocks"):
                    name=f"SYNTHETIC-{rep}-{degree}-{family}"
                    rows.append({
                        "public_sha256":name,
                        "replicate":rep,"degree":degree,"family":family,
                        "global7_detection":.70,"uniform_detection":.80,
                        "global7_minus_uniform":-.10,
                        "global7_minus_exact_floor":.30,
                        "absolute_uniform_calibration_error":.01,
                        "exact_uniform_probability":.79,
                        "difference_support_size":200,
                        "hidden_true_world_witnessfree_false_accept":True,
                        "clean_true_world_authentic_accept":True,
                        "paired_public_prediction_accuracy":[1,2]
                    })
                    rec.append({
                        "public_sha256":name,
                        "replicate":rep,"degree":degree,"family":family,
                        "difference_support_size":200,
                        "mixture_detection_denominator":den,
                        "mixture_detection_numerators":{
                            p:int(round(den*v)) for p,v in {
                                "uniform16":.80,"original_mix":.40,
                                "deterministic_mix":.65,"global7_mix":.70,
                                "atom_random_h50":.45}.items()},
                        "paired_public_demos_equal":True,
                        "paired_worlds":2,
                        "public_predictor_correct_worlds":1
                    })
        return {
            "schema":"nor42.paired_program_ood_authenticated_witness.v1",
            "registration_first_commit":audit.REGISTERED,
            "fresh_public_source_pairs":64,
            "two_world_targets":128,
            "excluded_historical_sources":388,
            "auditor_policies":list(audit.POLICIES),
            "diagnostic_auditor":audit.DIAG,
            "audit_budget":16,
            "heldout_schedules_per_source":1024,
            "source_records":rec,"source_level_primary":rows,
            "source_means":{
                "absolute_uniform_calibration_error":.01,
                "global7_hidden_detection":.70,
                "global7_minus_exact_floor":.30
            },
            "source_direction_wins":{"synthetic":64},
            "registered_five_gates":{
                "paired_prefix_identity_and_future_underdetermination":True,
                "authenticated_uniform_exact_support_calibration":True,
                "authenticated_hidden_detection_over_witnessfree_consistency":True,
                "global7_competitive_offgrammar_new_family":False,
                "source_first_hidden_witness_and_independent_replay_contract":True
            },
            "registered_primary_pass":False
        }

    def test_source_pair_block_and_oracle_analysis(self):
        report=audit.audit(self.data(),with_signflip=False)
        self.assertEqual(report["independent_pair_units"],64)
        self.assertEqual(report["replicate_blocks"],16)
        self.assertEqual(report["true_target_worlds"],128)
        self.assertEqual(len(report["subgroup_means"]["family"]),2)
        self.assertEqual(report["absolute_detection_means"]["global7_mix"],
                         0.70 if 0.70*256*1024%1==0 else
                         int(round(.70*256*1024))/(256*1024))
        self.assertIn("hidden-world",audit.markdown(report))

    def test_duplicate_source_pair_is_rejected(self):
        grade=self.data()
        grade["source_records"][1]["public_sha256"]=grade["source_records"][0]["public_sha256"]
        with self.assertRaises(AssertionError):
            audit.audit(grade,with_signflip=False)


if __name__=="__main__":
    unittest.main()
