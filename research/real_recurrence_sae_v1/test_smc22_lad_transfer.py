"""SMC-22 CPU tests of deterministic LAD fit and strict three-control gates."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc22_lad_calibration as fit
import smc22_lad_evaluate as run


class LADTests(unittest.TestCase):
    def source(self):
        rows=[]
        for i in range(64):
            good=i<30
            item={'source_finite_residual':1.,
                  'other_finite_residual':-1.,
                  'target_finite_residual':.5}
            rows.append({'puzzle_index':i,'seed':i,
                         'metrics':[dict(item) if good else None]+[None]*7})
        return rows

    def predictions(self):
        rows=[]
        for i in range(64):
            good=i<30
            item={'target_native_drop':3.,
                  'direct_prediction':2.,
                  'ridge_source_prediction':1.,
                  'source_calibrated_prediction':3.,
                  'other_calibrated_prediction':.5}
            rows.append({'puzzle_index':i,'eligible_mask':[good]+[False]*7,
                         'metrics':[dict(item) if good else None]+[None]*7})
        return rows

    def fits(self):
        return {'lad_source':{'identified':True},
                'lad_other':{'identified':True},
                'original_ridge_source':{'identified':True}}

    def test_source_weighted_median_exact(self):
        result=fit.weighted_median_fit(self.source(),'source_finite_residual',
                                       {'coefficient':.45})
        self.assertTrue(result['identified'])
        self.assertEqual(result['calibration_eligible_goals'],30)
        self.assertEqual(result['calibration_eligible_puzzles'],30)
        self.assertAlmostEqual(result['coefficient'],.5)
        self.assertEqual(result['winning_sorted_index'],14)

    def test_equal_budget_other_active_lad(self):
        result=fit.weighted_median_fit(self.source(),'other_finite_residual',
                                       {'coefficient':-.45})
        self.assertTrue(result['identified'])
        self.assertAlmostEqual(result['coefficient'],-.5)

    def test_weighted_median_uses_absolute_source_magnitude(self):
        rows=self.source()
        rows[0]['metrics'][0]['source_finite_residual']=100.
        rows[0]['metrics'][0]['target_finite_residual']=100.
        result=fit.weighted_median_fit(rows,'source_finite_residual',
                                       {'coefficient':1.})
        self.assertEqual(result['coefficient'],1.)

    def test_median_source_zero_degeneracy_fails_closed(self):
        rows=self.source()
        for r in rows:
            if r['metrics'][0] is not None:
                r['metrics'][0]['source_finite_residual']=0.
        result=fit.weighted_median_fit(rows,'source_finite_residual',
                                       {'coefficient':0.})
        self.assertFalse(result['identified'])
        self.assertTrue(result['degenerate'])

    def test_weighted_median_clip_is_frozen(self):
        rows=self.source()
        for r in rows:
            if r['metrics'][0] is not None:
                r['metrics'][0]['target_finite_residual']=50.
        self.assertEqual(fit.weighted_median_fit(rows,'source_finite_residual',
                          {'coefficient':2.})['coefficient'],2.)

    def test_all_three_paired_controls_have_positive_gate(self):
        x=run.score_family(self.predictions(),self.fits(),202622130)
        self.assertTrue(x['registered_success'])
        self.assertEqual(x['eligible_goals'],30)
        self.assertEqual(x['eligible_puzzles'],30)
        self.assertEqual(x['bootstrap95_gain_vs_direct'],[1.,1.])
        self.assertEqual(x['bootstrap95_gain_vs_original_ridge'],[2.,2.])
        self.assertEqual(x['bootstrap95_gain_vs_other_LAD'],[2.5,2.5])
        self.assertEqual(x['source_LAD_over_direct_MAE_ratio'],0.)

    def test_original_ridge_superiority_fails_gate(self):
        rows=self.predictions()
        for row in rows:
            if row['metrics'][0]:
                row['metrics'][0]['ridge_source_prediction']=3.
                row['metrics'][0]['source_calibrated_prediction']=2.5
        r=run.score_family(rows,self.fits(),202622130)
        self.assertFalse(r['registered_success'])
        self.assertFalse(r['registered_three_positive_paired_CI_gate'])

    def test_other_lad_superiority_fails_gate(self):
        rows=self.predictions()
        for row in rows:
            if row['metrics'][0]:
                row['metrics'][0]['other_calibrated_prediction']=3.
                row['metrics'][0]['source_calibrated_prediction']=2.5
        r=run.score_family(rows,self.fits(),202622130)
        self.assertFalse(r['registered_three_positive_paired_CI_gate'])

    def test_lad_calibration_not_identified_fails(self):
        fits=self.fits();fits['lad_source']={'identified':False}
        self.assertFalse(run.score_family(self.predictions(),fits,202622130)['registered_success'])

    def test_new_cohort_count_and_sign_convention(self):
        with self.assertRaises(ValueError):run.fresh_rosters({},limit=6)
        self.assertEqual(run.sign_eps(1e-8),0)
        self.assertEqual(run.sign_eps(-1e-8),0)
        self.assertEqual(run.sign_eps(-1.001e-8),-1)


if __name__=='__main__':unittest.main()
