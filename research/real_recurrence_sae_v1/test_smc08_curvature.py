"""SMC-08 CPU toy tests; no external checkpoint, network or heldout labels."""
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_smc07_real_sae import Toy
from smc07_real_sae import split_ids,fit_sae
from smc08_curvature import (
    frozen_weight_certificate,goal_jacobian,sae_direction_set,
    synthetic_measure,aggregate_reports
)


class SMC08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.teacher=Toy().eval()
        cls.inputs=torch.randint(1,10,(8,81))
        cls.inputs[:,3]=0
        cls.split=split_ids(8)
        with torch.no_grad():
            cls.states=[z.detach() for z in cls.teacher.reason_states(cls.inputs)]
        cls.sae,cls.mu,cls.std,_=fit_sae(
            cls.states[1][cls.split['train']],cls.states[1][cls.split['val']],
            seed=71,steps=20)

    def test_global_weight_certificate_uses_only_frozen_teacher_parameters(self):
        x=frozen_weight_certificate(self.teacher)
        self.assertEqual(x['num_elementary_stages'],12)
        self.assertGreater(x['global_hessian_upper_log10_approx'],0)
        self.assertGreater(x['global_jacobian_upper_log10_approx'],0)
        self.assertIn('exact real',x['claim_scope'].replace('-',' '))

    def test_numeric_jacobian_matches_central_difference(self):
        i=self.split['test'][0]
        F,v,J,top,sv,ref=goal_jacobian(
            self.teacher.double(),self.states[1][i:i+1],3)
        self.assertEqual(tuple(J.shape),(10,48))
        self.assertGreater(float(sv[0]),0)
        self.assertAlmostEqual(float((J@top).norm()),float(sv[0]),places=7)
        delta=1e-4
        numeric=(F(v+delta*top)-F(v-delta*top))/(2*delta)
        self.assertLess(float((numeric-J@top).norm()),1e-3)

    def test_source_direction_and_finite_replay_have_explicit_controls(self):
        report=synthetic_measure(self.teacher,self.states,self.inputs,
                                 self.sae,self.mu,self.std,self.split,
                                 seed=71,radii=('0.02','0.2'))
        self.assertEqual(report['n_independent_test_puzzles'],2)
        self.assertEqual(len(report['per_puzzle']),2)
        for case in report['per_puzzle']:
            self.assertLessEqual(case['numerical_rank_rel1e-8'],10)
            for name in ('sae_active','sae_inactive','isotropic','jacobian_optimal'):
                row=case['directions'][name]
                self.assertLessEqual(row['fraction_of_optimal_jacobian_slope'],1+1e-8)
                self.assertIn('0.2',row['radii'])
                self.assertGreaterEqual(row['radii']['0.2']['linear_residual_l2_fp64'],0)
        aggregated=aggregate_reports((report,),radii=('0.02','0.2'))
        self.assertEqual(aggregated['0.2_active_minus_isotropic']['n_puzzle_clusters'],2)


if __name__=="__main__":
    unittest.main()
