"""SMC-20 CPU preregistered anchored-gradient and frozen-bootstrap tests."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc20_anchored_refit as s


class SMC20AnchoredGradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def toy_rows(self):
        rows=[]
        for p in range(64):
            present=p<30
            record={
              'observed_target_drop':2.,
              'observed_reference_drop':1.,
              'gradient_anchored_prediction':2.,
              'reference_only_prediction':1.,
              'other_active_surrogate_prediction':0.
            }
            rows.append({
                'eligible_mask':[present]+[False]*7,
                'metrics':[dict(record) if present else None]+[None]*7
            })
        return rows

    def test_exact_clean_linear_anchor_same_teacher(self):
        g=torch.zeros(48,dtype=torch.float64);g[0]=3.;g[1]=4.
        a=torch.zeros(48,dtype=torch.float64);a[0]=1.
        b=torch.zeros(48,dtype=torch.float64);b[1]=2.
        # Under linear m(h)=3h_0+4h_1, source effect is 3 and target 8.
        self.assertEqual(s.anchored_predict(3.,g,a,b),8.)

    def test_identical_edits_leave_source_effect_unchanged(self):
        a=torch.arange(48,dtype=torch.float64)/97.
        g=torch.flip(a,dims=[0])
        self.assertEqual(s.anchored_predict(.123,g,a,a),.123)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            s.anchored_predict(1.,torch.zeros(47,dtype=torch.float64),
                               torch.zeros(48,dtype=torch.float64),
                               torch.zeros(48,dtype=torch.float64))

    def test_fixed_64_puzzle_bootstrap_and_all_primary_gates(self):
        r=s.score_family(self.toy_rows(),202620130)
        self.assertTrue(r['registered_success'])
        self.assertEqual(r['eligible_goals'],30)
        self.assertEqual(r['eligible_puzzles'],30)
        self.assertEqual(r['valid_bootstrap_resamples'],2000)
        self.assertEqual(r['puzzle_cluster_bootstrap95_gain_over_reference'],[1.,1.])
        self.assertEqual(r['puzzle_cluster_bootstrap95_gain_over_other'],[2.,2.])
        self.assertEqual(r['reference_only_sign_accuracy'],1.)
        self.assertEqual(r['gradient_anchored_sign_accuracy'],1.)

    def test_negative_gradient_improvement_preserved_as_failure(self):
        rows=self.toy_rows()
        for row in rows:
            if row['metrics'][0] is not None:
                row['metrics'][0]['gradient_anchored_prediction']=-2.
        result=s.score_family(rows,202620130)
        self.assertFalse(result['registered_success'])
        self.assertFalse(result['registered_sign_nondegradation_gate'])
        self.assertFalse(result['registered_relative_mae_gate'])
        self.assertFalse(result['registered_both_bootstrap_improvement_gate'])

    def test_positive_heldout_effect_but_failed_other_active_null(self):
        rows=self.toy_rows()
        for row in rows:
            if row['metrics'][0] is not None:
                row['metrics'][0]['other_active_surrogate_prediction']=2.
        result=s.score_family(rows,202620130)
        self.assertFalse(result['registered_success'])
        self.assertLess(result['puzzle_cluster_bootstrap95_gain_over_other'][0],0)

    def test_registered_full_or_pilot_roster_sizes_only(self):
        with self.assertRaises(ValueError):
            s.fresh_rosters({},limit=12)

    def test_sign_zero_convention(self):
        self.assertEqual(s.sign_eps(0.),0)
        self.assertEqual(s.sign_eps(1e-8),0)
        self.assertEqual(s.sign_eps(-1e-8),0)
        self.assertEqual(s.sign_eps(-1.00000001e-8),-1)
        self.assertEqual(s.sign_eps(1.00000001e-8),1)


if __name__=='__main__':unittest.main()
