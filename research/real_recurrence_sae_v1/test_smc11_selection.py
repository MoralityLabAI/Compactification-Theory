"""SMC-11 CPU smoke controls; source-locked training is run in hosted CI."""
import json
import sys
import tempfile
from pathlib import Path
from fractions import Fraction as Q
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from selection_cells_exact import (
    affine_normalized_encoder,topk_selection_certificate,strict_guard,
    sharp_boundary_displacement,scores_at
)
from smc11_selection import (
    goal_selection_cases,descriptive_all_token_radii,
    compose_with_previous,load_smc10_receipt
)
from smc07_real_sae import fit_sae,split_ids
from test_smc07_real_sae import Toy


class SAESelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(132)
        cls.teacher=Toy().eval()
        cls.inputs=torch.randint(1,10,(8,81))
        cls.inputs[:,3]=0
        with torch.no_grad():
            cls.stage=cls.teacher.reason_states(cls.inputs)[1].detach()
        cls.split=split_ids(8)
        cls.sae,cls.mu,cls.std,_=fit_sae(
            cls.stage[cls.split['train']],cls.stage[cls.split['val']],
            seed=71,steps=24)

    def test_exact_scaling_must_use_physical_metric(self):
        rows,offs=affine_normalized_encoder(
            ((2,0),(-2,0),(0,0)),(1,0,-1),(5,7),(2,4))
        a=scores_at(rows,offs,(6,7))
        self.assertEqual(a,(2,-1,-1))
        self.assertEqual(rows[0],(Q(1),Q(0)))

    def test_strict_exact_hyperplane_witness_and_guard(self):
        rows=((Q(1),),(Q(-1),),(Q(2),))
        ofs=(Q(2),Q(1),Q(0))
        cert=topk_selection_certificate(rows,ofs,(0,),k=2)
        self.assertEqual(cert['top1']['radius_squared'],Q(1,4))
        self.assertEqual(cert['topk']['radius_squared'],Q(1,9))
        self.assertTrue(strict_guard(cert['top1']['radius_squared'],Q(1,3)))
        self.assertFalse(strict_guard(cert['topk']['radius_squared'],Q(1,3)))
        w=sharp_boundary_displacement(rows,cert['exact_preactivations'],cert['topk'])
        self.assertEqual(w,(Q(1,3),))
        vals=scores_at(rows,ofs,w)
        self.assertEqual(vals[1],vals[2])

    def test_toy_trained_sae_selection_pipeline_abstains_on_ties(self):
        cases=goal_selection_cases(self.stage,self.inputs,self.sae,
                                    self.mu,self.std,self.split['test'],71)
        self.assertEqual(len(cases),2)
        for case in cases:
            self.assertIn(case['status'],('certified','abstain_ValueError',
                                          'abstain_machine_nominal_disagreement'))
            if case['status']=='certified':
                self.assertEqual(len(case['top4_features']),4)
                self.assertGreater(case['top1_radius_approx'],0)
                self.assertGreater(case['top4_radius_approx'],0)
                self.assertTrue(case['budget_guards']['0']['top1'])
                self.assertTrue(case['budget_guards']['0']['top4'])

    def test_all_token_fp64_survey_explicitly_not_exact(self):
        report=descriptive_all_token_radii(self.sae,self.stage,
                 self.mu,self.std,self.split['test'])
        self.assertEqual(report['n_test_tokens'],162)
        self.assertIn('descriptive',report['method'])
        self.assertGreaterEqual(report['strict_nominal_tokens'],0)
        self.assertLessEqual(report['strict_nominal_tokens'],162)

    def test_composed_policies_require_correct_active_or_topk_identity(self):
        cases=[{
          'seed':71,'puzzle_id':3,'status':'certified',
          'budget_guards':{
            '0':{'top1':True,'top4':True},
            '1/100000000':{'top1':True,'top4':False},
          }},
          {'seed':71,'puzzle_id':7,'status':'abstain_ValueError',
           'budget_guards':{'0':{'top1':False,'top4':False}}},
        ]
        raw=[
          {'seed':71,'puzzle_id':3,'radius':'1/5','common_l2_uncertainty':'1/100000000',
           'comparisons':{'isotropic':'sae_active_greater','sae_inactive':'control_greater',
                          'jacobian_optimal':'control_greater'}},
          {'seed':71,'puzzle_id':7,'radius':'1/5','common_l2_uncertainty':'0',
           'comparisons':{'isotropic':'sae_active_greater','sae_inactive':'control_greater',
                          'jacobian_optimal':'control_greater'}}]
        out=compose_with_previous(cases,raw)
        self.assertEqual(out['cases'][0]['comparisons']['isotropic']['composed_policy_certificate'],
                         'sae_active_greater')
        self.assertEqual(out['cases'][0]['comparisons']['sae_inactive']['composed_policy_certificate'],
                         'abstain_selection')
        self.assertEqual(out['cases'][0]['comparisons']['jacobian_optimal']['composed_policy_certificate'],
                         'control_greater')
        self.assertEqual(out['cases'][1]['comparisons']['isotropic']['composed_policy_certificate'],
                         'abstain_selection')
        self.assertEqual(out['summary']['1/5|1/100000000|isotropic']['active_greater'],1)

    def test_reject_any_changed_prior_certificate_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'forged.json'
            p.write_text('{"schema":"plausible","cases":[]}')
            with self.assertRaises(ValueError):
                load_smc10_receipt(p)


if __name__=='__main__':
    unittest.main()
