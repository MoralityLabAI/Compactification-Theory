"""SMC-21 no-checkpoint CPU tests for calibration-only ridge and frozen gates."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc21_residual_transfer as s


class ResidualTests(unittest.TestCase):
    def calibration(self):
        out=[]
        for p in range(64):
            x={'source_finite_residual':1.,
               'target_finite_residual':.5,
               'other_finite_residual':-1.}
            out.append({'metrics':[dict(x) if p<30 else None]+[None]*7})
        return out

    def scored_rows(self):
        out=[]
        for p in range(64):
            good=p<30
            item={'target_native_drop':2.0,
                  'direct_prediction':1.,
                  'source_calibrated_prediction':2.,
                  'other_calibrated_prediction':0.}
            out.append({'eligible_mask':[good]+[False]*7,
                        'metrics':[dict(item) if good else None]+[None]*7})
        return out

    def test_shrunk_univariate_ridge_no_target_filter(self):
        r=s.fit_ridge(self.calibration(),'source_finite_residual')
        self.assertEqual(r['eligible_goals'],30)
        self.assertEqual(r['eligible_puzzles'],30)
        self.assertTrue(r['identified'])
        self.assertAlmostEqual(r['coefficient'],.5/1.05)

    def test_equal_parameter_other_active_calibration(self):
        r=s.fit_ridge(self.calibration(),'other_finite_residual')
        self.assertTrue(r['identified'])
        self.assertAlmostEqual(r['coefficient'],-.5/1.05)

    def test_zero_variance_abstains_on_frozen_source(self):
        rows=self.calibration()
        for p in rows:
            if p['metrics'][0]:
                p['metrics'][0]['source_finite_residual']=0.
        r=s.fit_ridge(rows,'source_finite_residual')
        self.assertTrue(r['degenerate'])
        self.assertFalse(r['identified'])
        self.assertEqual(r['coefficient'],0.)

    def test_frozen_ridge_clips_to_two(self):
        rows=self.calibration()
        for p in rows:
            if p['metrics'][0]:
                p['metrics'][0]['target_finite_residual']=10.
        self.assertEqual(s.fit_ridge(rows,'source_finite_residual')['coefficient'],2.)

    def test_puzzle_clustered_direct_and_equal_budget_null_gate(self):
        b={'beta_source':{'identified':True},
           'beta_other':{'identified':True}}
        r=s.score_family(self.scored_rows(),b,202621130)
        self.assertTrue(r['registered_success'])
        self.assertEqual(r['eligible_goals'],30)
        self.assertEqual(r['eligible_puzzles'],30)
        self.assertEqual(r['puzzle_cluster_bootstrap95_gain_vs_direct'],[1.,1.])
        self.assertEqual(r['puzzle_cluster_bootstrap95_gain_vs_other'],[2.,2.])
        self.assertAlmostEqual(r['source_vs_direct_mae_ratio'],0.)
        self.assertEqual(r['valid_bootstrap_draws'],2000)

    def test_no_calibration_interventions_fails_gate(self):
        b={'beta_source':{'identified':False},
           'beta_other':{'identified':True}}
        self.assertFalse(s.score_family(self.scored_rows(),b,202621130)['registered_success'])

    def test_strong_direct_gradient_wins_preserve_failure(self):
        rows=self.scored_rows()
        for r in rows:
            if r['metrics'][0] is not None:
                r['metrics'][0]['direct_prediction']=2.
                r['metrics'][0]['source_calibrated_prediction']=1.
        b={'beta_source':{'identified':True},'beta_other':{'identified':True}}
        r=s.score_family(rows,b,202621130)
        self.assertFalse(r['registered_success'])
        self.assertFalse(r['registered_both_positive_gain_gate'])

    def test_other_active_surrogate_wins_preserve_failure(self):
        rows=self.scored_rows()
        for r in rows:
            if r['metrics'][0] is not None:
                r['metrics'][0]['source_calibrated_prediction']=1.5
                r['metrics'][0]['other_calibrated_prediction']=2.
        b={'beta_source':{'identified':True},'beta_other':{'identified':True}}
        r=s.score_family(rows,b,202621130)
        self.assertFalse(r['registered_success'])
        self.assertLess(r['puzzle_cluster_bootstrap95_gain_vs_other'][0],0)

    def test_pilot_limit_and_sign_thresholds(self):
        with self.assertRaises(ValueError):
            s.fresh_rosters({},12)
        self.assertEqual(s.sign_eps(0.),0)
        self.assertEqual(s.sign_eps(1e-8),0)
        self.assertEqual(s.sign_eps(-1.0001e-8),-1)


if __name__=='__main__':unittest.main()
