"""SMC-16 CPU isolated geometry and frozen-scoring regression tests."""
import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))

import smc16_selective_kill as s
from test_smc15_active_necessity import FakeSAE,LinearTeacher
from smc15_active_necessity import ablate_margin


class SelectiveKillTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make(self):
        sae=FakeSAE().eval()
        mean=torch.zeros(48);scale=torch.ones(48)
        h=torch.zeros(48)
        h[:4]=torch.tensor([3.,2.,1.,0.5])
        code=s.encode_one(sae,mean,scale,h)
        return sae,mean,scale,h,code,s.physical_covectors(sae,scale),s.physical_columns(sae,scale)

    def test_projection_kills_only_selected_covector(self):
        e=torch.eye(48,dtype=torch.float64)
        target=e[0]+e[1]*2+e[4]*4
        v=s.projected_normal(target,e[1:4])
        self.assertTrue(bool(torch.allclose(v,e[0]+e[4]*4,atol=1e-12)))
        self.assertTrue(bool(torch.allclose(e[1:4]@v,torch.zeros(3,dtype=torch.float64),atol=1e-12)))

    def test_rank_and_identifiability_obstructions_fail_closed(self):
        e=torch.eye(48,dtype=torch.float64)
        self.assertIsNone(s.projected_normal(e[0],torch.stack((e[1],e[1],e[2]))))
        self.assertIsNone(s.projected_normal(e[1],torch.stack((e[1],e[2],e[3]))))
        with self.assertRaises(ValueError):
            s.projected_normal(torch.zeros(49),e[1:4])

    def test_original_top_four_except_j_preserved_and_sham_preserves_all(self):
        sae,mean,scale,h,code,cov,dec=self.make()
        kill,sham,info=s.selective_vectors(h,code,0,sae,mean,scale,cov,dec,
                                           1,0,'toy',0)
        self.assertEqual(info['status'],'eligible')
        self.assertAlmostEqual(float(kill.norm()),float(sham.norm()),places=10)
        after=s.encode_one(sae,mean,scale,h.double()-kill)
        sham_code=s.encode_one(sae,mean,scale,h.double()-sham)
        self.assertEqual(float(after[0]),0.)
        for j in (1,2,3):
            self.assertAlmostEqual(float(after[j]),float(code[j]),places=5)
        for j in (0,1,2,3):
            self.assertAlmostEqual(float(sham_code[j]),float(code[j]),places=5)
        kill2,sham2,_=s.selective_vectors(h,code,0,sae,mean,scale,cov,dec,
                                         1,0,'toy',0)
        self.assertTrue(torch.equal(kill,kill2))
        self.assertTrue(torch.equal(sham,sham2))

    def test_inactive_feature_and_dose_cap_trigger_explicit_ineligibility(self):
        sae,mean,scale,h,code,cov,dec=self.make()
        _,_,missing=s.selective_vectors(h,code,45,sae,mean,scale,cov,dec,
                                        1,0,'toy',0)
        self.assertEqual(missing['status'],'matched_inactive')
        tiny=dec.clone();tiny[0]*=1e-6
        _,_,limited=s.selective_vectors(h,code,0,sae,mean,scale,cov,tiny,
                                        1,0,'toy',0)
        self.assertEqual(limited['status'],'dose_cap')

    def test_teacher_clean_minus_intervention_and_zero_sham(self):
        teacher=LinearTeacher().eval()
        stage=torch.zeros((1,81,48),dtype=torch.float32)
        stage[:,:,0]=2.
        goals=torch.arange(8).unsqueeze(0)
        truth=torch.ones((1,8),dtype=torch.long)
        base=teacher.head(stage.double())
        perturb=torch.zeros((1,8,48),dtype=torch.float64)
        perturb[:,:,0]=1.
        k,correct,flips=ablate_margin(teacher,stage,base,truth,goals,perturb)
        z,_,_=ablate_margin(teacher,stage,base,truth,goals,torch.zeros_like(perturb))
        self.assertTrue(bool(correct.all()))
        self.assertTrue(bool((k==2).all()))
        self.assertTrue(bool((z==0).all()))
        self.assertFalse(bool(flips.any()))

    def test_cluster_resampling_gate_and_flip_control(self):
        rows=[]
        for p in range(64):
            active=[p<24]+[False]*7
            rows.append({'eligible_mask':active,
                         'kill_margin_drop':[2.]+[None]*7,
                         'sham_margin_drop':[-1.]+[None]*7,
                         'clean_correct':[True]*8,
                         'kill_correct_to_wrong':[True]+[None]*7,
                         'sham_correct_to_wrong':[False]+[None]*7})
        stats=s.score(rows,seed=7)
        self.assertTrue(stats['registered_success'])
        self.assertEqual(stats['eligible_goals'],24)
        self.assertEqual(stats['nonempty_puzzle_clusters'],24)
        self.assertAlmostEqual(stats['mean_selective_kill_minus_sham_margin_drop'],3)
        self.assertAlmostEqual(stats['puzzle_cluster_bootstrap95'][0],3)
        for row in rows:
            row['kill_correct_to_wrong'][0]=False
        failed=s.score(rows,seed=7)
        self.assertFalse(failed['registered_success'])
        self.assertEqual(failed['kill_flip_rate_eligible'],0.)

    def test_bad_puzzle_or_payload_fails(self):
        with self.assertRaises(ValueError):
            s.new_rosters({},limit=5)
        with self.assertRaises(ValueError):
            s.score([{'eligible_mask':[True],'kill_margin_drop':[1],
                     'sham_margin_drop':[0]}],seed=0)


if __name__=='__main__':
    unittest.main()
