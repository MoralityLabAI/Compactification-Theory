"""NOR-12B independent heldout replay and model-level basin-input ablation gates."""
import unittest
from dataclasses import replace
from fractions import Fraction

from research.nor12_task_readout import independent_confirmation as c
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as task


class FrozenReadoutConfirmation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.ck,cls.probe,cls.negative,cls.fitted,cls.old,
         cls.design,cls.roster)=c.rebuild_frozen_models()

    def test_rebuild_is_exact_original_96_training_cases(self):
        self.assertEqual(len(self.old["train"]),96)
        self.assertEqual(len(self.old["dev"]),36)
        self.assertEqual(len(self.old["test"]),72)
        self.assertEqual(self.design["fixed_development_scales"]["basin"],.25)
        self.assertEqual(self.design["global_development_choice"]["kind"],"random_feature")
        self.assertEqual(set(self.fitted),set(c.FIT_HASH_KEYS))
        for name in self.fitted:
            self.assertEqual(
                c.weight_digest(self.fitted[name]["weights"]),
                self.design["fitted_weight_sha256_by_kind"][name])

    def test_fresh_seed_masks_every_train_dev_and_original_test_case(self):
        new=c.fresh_cohorts(self.roster,replicates=1,per_family=2)
        self.assertEqual(len(new),1)
        cases=new[0]
        self.assertEqual(len(cases),12)
        original_features={digest for group in self.roster.values()
                           for case,head,digest in group}
        new_features={digest for case,head,digest in cases
                      if case.family!="late_source"}
        self.assertFalse(new_features & original_features)
        self.assertEqual(len(new_features),10)
        late=[d for case,head,d in cases if case.family=="late_source"]
        self.assertEqual(len(late),2)
        self.assertEqual(len(set(late)),1)

    def test_original_first_split_basin_feature_ablation(self):
        a=c.ablate_frozen_basin_feature(self.old["test"],self.fitted,self.design)
        self.assertEqual(a["frozen_basin_readout_scale"],.25)
        x=a["variants"]
        self.assertEqual(x["original_basin_feature"]["future_correct"],44)
        self.assertEqual(x["original_basin_feature"]["paired_baseline"]["newly_correct"],2)
        self.assertEqual(x["original_basin_feature"]["paired_baseline"]["lost_correct"],0)
        self.assertEqual(x["original_basin_feature"]["feature_groups"]["unique_model_inputs"],61)
        self.assertEqual(set(x),{
          "original_basin_feature","set_basin_feature_zero",
          "replace_with_shuffled_probe","permuted_unique_feature_groups"})

    def test_exact_sign_test_on_paired_discordance(self):
        self.assertEqual(c.exact_sign_p(0,0),Fraction(1))
        self.assertEqual(c.exact_sign_p(2,0),Fraction(1,2))
        self.assertEqual(c.exact_sign_p(3,0),Fraction(1,4))
        self.assertEqual(c.exact_sign_p(1,1),Fraction(1))

    def test_confirm_short_is_sealed_and_no_test_tuning(self):
        # This deliberately re-fits the original 96 training rows and
        # freezes the exact weight hashes before generating the new cases.
        r=c.evaluate_confirm(replicates=1,per_family=2)
        self.assertTrue(r["scope"]["all_weight_hashes_verified_before_new_cases_scored"])
        self.assertTrue(r["scope"]["all_scales_pinned_before_independent_test"])
        self.assertFalse(r["scope"]["heldout_future_labels_used_for_training"])
        self.assertEqual(r["independent_holdout_aggregate"]["original"]["cases"],12)
        self.assertEqual(r["independent_holdout_replicates"][0]["cases"],12)
        for model in ("original",)+c.FIT_HASH_KEYS:
            x=r["independent_holdout_aggregate"][model]
            self.assertEqual(x["future_correct"]-r["independent_holdout_aggregate"]["original"]["future_correct"],
                             x["net_change_from_original"])
            self.assertLessEqual(x["paid_checks"],4*x["cases"])
            self.assertGreaterEqual(x["unique_test_input_groups"],1)


if __name__=="__main__":
    unittest.main()
