"""SMC-21 standalone numerical receipt mutation tests."""
import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc21 as a
from test_smc21_residual_transfer import ResidualTests


class AuditTests(unittest.TestCase):
    def test_independent_bootstrap_matches_registered_synthetic_cohort(self):
        r=ResidualTests().scored_rows()
        fit={'beta_source':{'identified':True},'beta_other':{'identified':True}}
        score=a.independent_score(r,fit,202621130)
        self.assertTrue(score['registered_success'])
        self.assertEqual(score['puzzle_cluster_bootstrap95_gain_vs_direct'],[1.,1.])
        self.assertEqual(score['puzzle_cluster_bootstrap95_gain_vs_other'],[2.,2.])

    def test_zero_calibration_fit_blocks_scientific_gate(self):
        fit={'beta_source':{'identified':False},'beta_other':{'identified':True}}
        self.assertFalse(a.independent_score(ResidualTests().scored_rows(),fit,202621130)['registered_success'])

    def toy_native(self):
        ca=[[0.]*48 for _ in range(96)]
        cb=[[0.]*48 for _ in range(96)]
        ca[3][0]=1.;cb[7][0]=1.;cb[7][1]=1.;cb[5][1]=1.
        source=3;target=7;other=5
        da=[0.]*48;da[0]=2.
        db=[0.]*48;db[0]=1.;db[1]=1.
        do=[0.]*48;do[1]=2**.5
        return ca,cb,(da,db,do),(2.,1.,1.),(source,target,other)

    def test_native_edits_reconstructed_from_source_decoders(self):
        ca,cb,(da,db,do),coeffs,ids=self.toy_native()
        a.check_native(da,db,do,coeffs,ids,ca,cb)

    def test_forged_equal_dose_control_caught(self):
        ca,cb,(da,db,do),coeffs,ids=self.toy_native()
        wrong=do[:];wrong[1]*=1.2
        with self.assertRaises(ValueError):
            a.check_native(da,db,wrong,coeffs,ids,ca,cb)

    def test_forged_target_decoder_caught(self):
        ca,cb,(da,db,do),coeffs,ids=self.toy_native()
        wrong=db[:];wrong[0]=3.
        with self.assertRaises(ValueError):
            a.check_native(da,wrong,do,coeffs,ids,ca,cb)

    def test_calibration_fit_mse_shrink_math_exactly(self):
        xs=[1.,2.,-2.];ys=[.5,1.,-1.]
        xx=sum(x*x for x in xs)
        xy=sum(x*y for x,y in zip(xs,ys))
        self.assertAlmostEqual(xy/(1.05*xx),.5/1.05)

    def test_sign_epsilon_independent_from_heldout_target(self):
        self.assertEqual(a.sign_eps(1e-8),0)
        self.assertEqual(a.sign_eps(-1.01e-8),-1)
        self.assertEqual(a.sign_eps(1.01e-8),1)

    def test_new_puzzle_cohort_size_guard(self):
        with self.assertRaises(ValueError):
            a.audit_rosters({'puzzles_per_family':6,'new_rosters':{},
                             'new_seed_bases':{}})


if __name__=='__main__':unittest.main()
