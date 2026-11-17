"""SMC-27 source-independent CPU regression tests for old-only policy selection.

No test target, checkpoint training, or SMC26 observed test effect is needed.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc27_calibrate import (
    OLD_BUNDLE_SHA,OLD_CAL_SHA,GRID,protocol_sha,policy,
    select_old_threshold,evaluate_old_threshold
)
from smc27_predict import FAMILIES,SIZE,POLICIES,actions_for_effects
from smc27_score import summary,decision_utility,CONTROLS
from audit_smc27 import independently_resample,_close,independent_test_rosters
from smc26_train import TEACHERS,prior_component_key
from smc13_family import new_full_solution,dig_unique,count_solutions,rectangle_count


class SMC27Tests(unittest.TestCase):
    def test_protocol_registered_and_original_bundle_pinned(self):
        self.assertEqual(len(protocol_sha()),64)
        self.assertEqual(len(OLD_BUNDLE_SHA),64)
        self.assertEqual(OLD_BUNDLE_SHA[:8],"408348ee")
        self.assertEqual(OLD_CAL_SHA[:8],"145312c0")
        self.assertEqual(TEACHERS,("B501","B701","B811","B919"))
        self.assertEqual(prior_component_key("B501"),"newB_501_SAE_71")
        self.assertEqual(FAMILIES["smc27_novel30"],(202637030000,30))
        self.assertEqual(FAMILIES["smc27_novel34"],(202637040000,34))
        self.assertEqual(SIZE,128)

    def test_exact_original_action_cost_and_abstention_semantics(self):
        self.assertEqual(policy(.15,.15),0)
        self.assertEqual(policy(-.15,.15),0)
        self.assertEqual(policy(.150001,.15),1)
        self.assertEqual(policy(-.150001,.15),-1)
        self.assertAlmostEqual(decision_utility(0,0),1.)
        self.assertAlmostEqual(decision_utility(1,0),-1.1)
        self.assertAlmostEqual(decision_utility(1,1),.9)
        self.assertAlmostEqual(decision_utility(-1,1),-1.1)
        self.assertAlmostEqual(decision_utility(0,1),0.)
        self.assertEqual(len(GRID),9)

    @staticmethod
    def old_cal_fixture():
        cases=[]
        for i in range(64):
            eligible=[
                {"status":"eligible","calibrated_median":.5,"true_class":1},
                {"status":"eligible","calibrated_median":.25,"true_class":0},
            ]
            cases.append({"puzzle_index":i,"goals":eligible+
                          [{"status":"dose_cap"} for _ in range(6)]})
        return cases

    def test_calibration_threshold_selected_before_new_source(self):
        rows=self.old_cal_fixture()
        res=select_old_threshold(rows)
        self.assertEqual(res["status"],"CALIBRATED")
        self.assertEqual(res["selected_threshold"],.3)
        self.assertAlmostEqual(res["selected_threshold_old_action_rate"],.5)
        self.assertTrue(res["admissible"])
        results=res["threshold_grid_scores"]
        self.assertEqual(len(results),9)
        self.assertGreater(evaluate_old_threshold(rows,.3)["mean_utility_all_original_slots"],
                           evaluate_old_threshold(rows,.15)["mean_utility_all_original_slots"])
        self.assertTrue(results[1]["in_range"])

    def test_calibration_abstention_only_is_marked_inadmissible(self):
        rows=[{"goals":[{"status":"eligible","calibrated_median":0.,
                          "true_class":0}]*8} for _ in range(64)]
        choice=select_old_threshold(rows)
        self.assertFalse(choice["admissible"])
        self.assertEqual(choice["selected_threshold"],5.)
        self.assertEqual(choice["status"],"CALIBRATION_INADMISSIBLE")

    @staticmethod
    def synthetic_scored_rows():
        rows=[]
        for i in range(128):
            one={"status":"eligible","observed":{"true_D":1.},
                 "actions":{"calibrated_selected":1,"uniform_selected":-1,
                            "calibrated_fixed015":1,"always_abstain":0,
                            "shifted_selected":-1,"top_digit_selected":1}}
            zero={"status":"eligible","observed":{"true_D":0.},
                  "actions":{"calibrated_selected":0,"uniform_selected":1,
                             "calibrated_fixed015":1,"always_abstain":0,
                             "shifted_selected":1,"top_digit_selected":0}}
            rows.append({"puzzle_index":i,"goals":[
                {**g,"utility":{p:decision_utility(g["actions"][p],1 if g["observed"]["true_D"] else 0)
                                for p in POLICIES}} for g in (one,zero)
            ]+[{"status":"dose_cap"} for _ in range(6)]})
        return rows

    def test_registered_128_puzzle_bootstrap_passes_positive_synthetic_fixture(self):
        rows=self.synthetic_scored_rows()
        computed=summary(rows,"B811","smc27_novel30",True,True)
        self.assertEqual(computed["eligible_goals"],256)
        self.assertEqual(computed["eligible_puzzles"],128)
        self.assertEqual(computed["calibrated_selected_test_action_rate"],.5)
        self.assertEqual(computed["bootstrap_draws"],5000)
        self.assertTrue(computed["registered_pass"])
        self.assertTrue(all(v[0]>.02 for v in computed["bootstrap99_79167_gain"].values()))
        independent=independently_resample(rows,"B811","smc27_novel30",True,True)
        _close(computed,independent)
        for c in CONTROLS:
            self.assertAlmostEqual(computed["mean_primary_utility_gain"][c],
                                   independent["mean_primary_utility_gain"][c])

    def test_coverage_failure_and_calibration_failure_prevent_pass(self):
        rows=self.synthetic_scored_rows()
        for p in rows[50:]:
            p["goals"]=[{"status":"dose_cap"}]*8
        under=summary(rows,"B919","smc27_novel34",True,True,draws=200)
        self.assertFalse(under["registered_pass"])
        self.assertEqual(under["status"],"UNDERPOWERED")
        no_cal=summary(self.synthetic_scored_rows(),"B919","smc27_novel34",
                       False,True,draws=200)
        self.assertFalse(no_cal["registered_pass"])
        self.assertEqual(no_cal["status"],"CALIBRATION_INADMISSIBLE")

    def test_equal_computation_policy_arms(self):
        d={"calibrated_median":.3,"uniform_median":-.4,
           "shifted_median":.1,"top_digit_effect":-.3}
        actions=actions_for_effects(d,.35)
        self.assertEqual(actions["calibrated_selected"],0)
        self.assertEqual(actions["uniform_selected"],-1)
        self.assertEqual(actions["calibrated_fixed015"],1)
        self.assertEqual(actions["always_abstain"],0)
        self.assertEqual(len(actions),len(POLICIES))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            policy(float("nan"),.3)

    def test_new_source_seed_unique_sudoku_without_old_test_access(self):
        answer=new_full_solution(202637030000)
        puzzle=dig_unique(answer,202637030000+1000000,30)
        self.assertGreater(rectangle_count(answer),0)
        self.assertEqual(puzzle.count(0),30)
        self.assertEqual(count_solutions(puzzle,2),1)


if __name__=="__main__":
    unittest.main()
