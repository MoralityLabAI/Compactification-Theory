"""SMC-22 independent standard-library mutation tests."""
import copy
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc22 as a
from test_smc22_lad_transfer import LADTests


class AuditTests(unittest.TestCase):
    def setup_calibration(self):
        toy=LADTests()
        raw=toy.source()
        from smc22_lad_calibration import weighted_median_fit
        old={'coefficient':.45}
        fit=weighted_median_fit(raw,'source_finite_residual',old)
        return raw,old,fit

    def test_independent_sorted_median_reconstructs_all_source_records(self):
        rows,old,fit=self.setup_calibration()
        a.replay_one_median(rows,'source_finite_residual',old,fit)

    def test_changed_median_is_detected(self):
        rows,old,fit=self.setup_calibration()
        corrupted=copy.deepcopy(fit)
        corrupted['coefficient']+=1.
        with self.assertRaisesRegex(ValueError,'weighted median'):
            a.replay_one_median(rows,'source_finite_residual',old,corrupted)

    def test_forged_sorted_puzzle_goal_tie_break_is_detected(self):
        rows,old,fit=self.setup_calibration()
        corrupted=copy.deepcopy(fit)
        corrupted['sorted_nonzero_ratio_weight_records'][0]['goal_slot']=7
        with self.assertRaises(ValueError):
            a.replay_one_median(rows,'source_finite_residual',old,corrupted)

    def test_original_ridge_baseline_replay(self):
        rows,old,fit=self.setup_calibration()
        corrupted=copy.deepcopy(fit)
        corrupted['calibration_absolute_error_original_ridge']+=1.
        with self.assertRaises(ValueError):
            a.replay_one_median(rows,'source_finite_residual',old,corrupted)

    def test_independent_three_control_puzzle_bootstrap(self):
        t=LADTests()
        result=a.independent_score(t.predictions(),t.fits(),202622130)
        self.assertTrue(result['registered_success'])
        self.assertEqual(result['bootstrap95_gain_vs_direct'],[1.,1.])
        self.assertEqual(result['bootstrap95_gain_vs_original_ridge'],[2.,2.])
        self.assertEqual(result['bootstrap95_gain_vs_other_LAD'],[2.5,2.5])

    def test_independent_original_ridge_beat_lad_is_scientific_failure(self):
        t=LADTests();rows=t.predictions()
        for row in rows:
            if row['metrics'][0]:
                row['metrics'][0]['ridge_source_prediction']=3.
                row['metrics'][0]['source_calibrated_prediction']=2.5
        self.assertFalse(a.independent_score(rows,t.fits(),202622130)['registered_success'])

    def test_new_cohort_size_gate(self):
        with self.assertRaises(ValueError):
            a.audit_rosters({'puzzles_per_family':10,'fresh_rosters':{},
                            'test_seed_bases':{}})

    def test_mismatch_original_frozen_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError,'schema'):
            a.audit_calibration_lad({}, {'schema':'fake'},Path('.'))


if __name__=='__main__':unittest.main()
