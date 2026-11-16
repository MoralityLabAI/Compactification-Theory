"""Mutation tests for independent SMC-17 source receipt replay.

Check that the auditor recomputes all-96 SAE disturbances rather than simply
trusting the runner's summary or its best-sham label.
"""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import audit_smc17 as audit
import smc17_inactive_control as geometry
from test_smc17_inactive_control import InactiveControlTests


class IndependentAuditTests(unittest.TestCase):
    def eligible_toy(self):
        toy=InactiveControlTests()
        toy.setUp()
        h,sae,mean,scale,code,cov,kill,baseline=toy.toy()
        _,details=geometry.best_inactive_matched_sham(
            h,code,0,sae,mean,scale,cov,kill,baseline,0,"toy",1,0)
        return {"status":"eligible","candidate_geometry":details}

    def test_replay_recomputes_full_96d_disturbance(self):
        receipt=self.eligible_toy()
        audit.verify_candidate(receipt)
        corrupted=copy.deepcopy(receipt)
        index=corrupted["candidate_geometry"]["inactive_ids"][0]
        corrupted["candidate_geometry"]["all96_best_preactivation_delta"][index]+=1
        with self.assertRaises(ValueError):
            audit.verify_candidate(corrupted)

    def test_changing_winner_epsilon_is_caught(self):
        receipt=self.eligible_toy()
        corrupted=copy.deepcopy(receipt)
        corrupted["candidate_geometry"]["epsilon_best"]+=.5
        with self.assertRaises(ValueError):
            audit.verify_candidate(corrupted)

    def test_violating_equal_physical_dose_is_caught(self):
        receipt=self.eligible_toy()
        corrupted=copy.deepcopy(receipt)
        corrupted["candidate_geometry"]["physical_doses"][2]*=1.1
        with self.assertRaises(ValueError):
            audit.verify_candidate(corrupted)

    def test_shuffled_random_candidate_roster_is_caught(self):
        receipt=self.eligible_toy()
        corrupted=copy.deepcopy(receipt)
        candidates=corrupted["candidate_geometry"]["candidate_receipts"]
        candidates[4]["kind"]="random_999"
        with self.assertRaises(ValueError):
            audit.verify_candidate(corrupted)

    def test_preregistration_hash_cannot_be_forged(self):
        bad={"schema":"SMC-17/source-locked-inactive-matching-sham-v1",
             "preregistration_commit":"0000000",
             "candidate_random_budget":128}
        with self.assertRaises(ValueError):
            audit.audit(bad)

    def test_recomputed_primary_score_catches_forged_interval(self):
        # All 64 puzzle clusters contain one eligible goal.
        raw=[]
        for idx in range(64):
            raw.append({
                "eligible_mask":[True]+[False]*7,
                "kill_margin_drop":[2.0]+[None]*7,
                "sham_margin_drop":[0.0]+[None]*7,
                "baseline_sham_margin_drop":[1.0]+[None]*7,
                "clean_correct":[True]*8,
                "kill_correct_to_wrong":[True]+[None]*7,
                "sham_correct_to_wrong":[False]+[None]*7,
                "baseline_sham_correct_to_wrong":[False]+[None]*7,
                "edit_diagnostics":[{
                    "status":"eligible","candidate_geometry":{
                      "epsilon_baseline":2.0,"epsilon_best":1.0,
                      "inactive_residual_floor":0.0,"chosen_candidate":"ls_positive"
                    }
                }]+[{"status":"matched_inactive"} for _ in range(7)]
            })
        result=audit.replay_stats(raw,seed=202617130)
        self.assertTrue(result["registered_success"])
        self.assertEqual(result["inactive_matching_ratio"],.5)
        self.assertEqual(result["best_sham_causal_score"]["puzzle_cluster_bootstrap95"],[2.0,2.0])
        forged=copy.deepcopy(result["best_sham_causal_score"])
        forged["puzzle_cluster_bootstrap95"][0]=-10
        with self.assertRaises(ValueError):
            audit.compare_stats(result["best_sham_causal_score"],forged,"forged")


if __name__=="__main__":
    unittest.main()
