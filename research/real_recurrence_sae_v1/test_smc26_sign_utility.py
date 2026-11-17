"""SMC-26 CPU source-independent admissions and native decision statistics tests."""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc26_train import TEACHERS,NEW_SEEDS,protocol_sha,prior_component_key
from smc26_predict import policy,fresh_rosters,FAMILIES,SIZE,THRESHOLD
from smc26_score import (target_class,decision_utility,case_summary,
                         COST,CONTROLS)
from audit_smc26 import independent_summary
from smc25_candidate_effects import weighted_median,candidate_predictions
from smc13_family import new_full_solution,dig_unique,count_solutions,rectangle_count
from upstream_arch import TinyRecursiveModel


class SMC26Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registration_and_independent_teacher_roster(self):
        self.assertEqual(len(protocol_sha()),64)
        self.assertEqual(TEACHERS,("B501","B701","B811","B919"))
        self.assertEqual(NEW_SEEDS,(811,919))
        self.assertEqual(FAMILIES["smc26_novel30"],(202636030000,30))
        self.assertEqual(FAMILIES["smc26_novel34"],(202636040000,34))
        self.assertEqual(SIZE,128)
        self.assertEqual(sum(p.numel() for p in TinyRecursiveModel(48,4).parameters()),44001)

    def test_original_SMC13_B_teacher_SAE_aliases_are_source_exact(self):
        self.assertEqual(prior_component_key("B501"),"newB_501_SAE_71")
        self.assertEqual(prior_component_key("B701"),"newB_701_SAE_71")
        with self.assertRaises(ValueError):
            prior_component_key("B811")

    def test_predeclared_ternary_decision_and_cost(self):
        self.assertEqual(THRESHOLD,.15)
        self.assertEqual(policy(.15),0)
        self.assertEqual(policy(-.15),0)
        self.assertEqual(policy(.150001),1)
        self.assertEqual(policy(-.150001),-1)
        self.assertAlmostEqual(COST,.1)
        self.assertAlmostEqual(decision_utility(1,1),.9)
        self.assertAlmostEqual(decision_utility(-1,1),-1.1)
        self.assertAlmostEqual(decision_utility(0,1),0.)
        self.assertAlmostEqual(decision_utility(0,0),1.)
        self.assertAlmostEqual(decision_utility(-1,0),-1.1)
        self.assertEqual(target_class(0.1),0)
        self.assertEqual(target_class(0.4),1)

    def test_classwise_median_is_label_blind(self):
        clean=torch.linspace(-1,1,10,dtype=torch.float64)
        tangent=torch.tensor([0.,1.,-.1,.2,.3,-.4,.5,-.6,.7,.1],dtype=torch.float64)
        p=candidate_predictions(clean,tangent,temperature=2.)
        self.assertEqual(len(p["classwise_effects"]),9)
        self.assertEqual(len(p["calibrated_digit_distribution"]),9)
        self.assertAlmostEqual(sum(p["calibrated_digit_distribution"]),1.,places=10)
        self.assertAlmostEqual(p["predictions"]["calibrated_median"],
                               weighted_median(p["classwise_effects"],
                                               p["calibrated_digit_distribution"]))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            policy(float("nan"))

    @staticmethod
    def fake_rows(active=True):
        rows=[]
        for pi in range(128):
            goals=[]
            if active:
                patterns=[(1,{"calibrated_median":1,"uniform_median":-1,
                              "shifted_median":-1,"top_digit_effect":-1,
                              "always_abstain":0}),
                          (0,{"calibrated_median":0,"uniform_median":1,
                              "shifted_median":1,"top_digit_effect":1,
                              "always_abstain":0})]
                for truth,acts in patterns:
                    goals.append({"status":"eligible",
                                  "actions":acts,
                                  "true_effect_class":truth,
                                  "observed":{"true_D":1. if truth else 0.0,
                                              "clean_correct":True}})
            goals.extend({"status":"dose_cap"} for _ in range(8-len(goals)))
            rows.append({"puzzle_index":pi,"teacher_capability_matched":True,
                         "goals":goals})
        return rows

    def test_registered_puzzle_utility_pass_on_synthetic_positive_control(self):
        rows=self.fake_rows()
        r=case_summary(rows,"B501","smc26_novel30",draws=5000)
        self.assertEqual(r["eligible_goals"],256)
        self.assertEqual(r["eligible_puzzles"],128)
        self.assertEqual(r["bootstrap_draws"],5000)
        self.assertEqual(r["calibrated_action_rate"],.5)
        self.assertTrue(r["registered_pass"])
        self.assertGreater(r["puzzle_mean_gain_vs_primary_control"]["uniform_median"],.5)
        self.assertGreater(r["puzzle_mean_gain_vs_primary_control"]["always_abstain"],.1)
        aud=independent_summary(rows,"B501","smc26_novel30")
        for c in CONTROLS:
            self.assertAlmostEqual(
                aud["puzzle_mean_gain_vs_primary_control"][c],
                r["puzzle_mean_gain_vs_primary_control"][c],places=12)
            self.assertEqual(aud["bootstrap99_6875_gain"][c],
                             r["bootstrap99_6875_gain"][c])

    def test_all_abstain_and_ineligible_denominators(self):
        r=case_summary(self.fake_rows(active=False),
                       "B501","smc26_novel30",draws=100)
        self.assertEqual(r["status"],"underpowered")
        self.assertEqual(r["eligible_goals"],0)
        self.assertFalse(r["registered_pass"])

    def test_capability_mismatch_blocks_positive_synthetic_gate(self):
        rows=self.fake_rows()
        for r in rows:r["teacher_capability_matched"]=False
        s=case_summary(rows,"B919","smc26_novel34",draws=100)
        self.assertFalse(s["registered_pass"])
        self.assertEqual(s["status"],"capability_mismatch")

    def test_new_sudoku_source_uses_prospective_noncanonical_family(self):
        grid=new_full_solution(202636030000)
        x=dig_unique(grid,202636030000+1000000,30)
        self.assertGreater(rectangle_count(grid),0)
        self.assertEqual(x.count(0),30)
        self.assertEqual(count_solutions(x,2),1)


if __name__=="__main__":
    unittest.main()
