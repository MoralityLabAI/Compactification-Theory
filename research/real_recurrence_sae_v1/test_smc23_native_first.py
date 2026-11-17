"""SMC-23 source-independent CPU regression tests, requiring only Torch."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc07_real_sae import TopKSAE
from smc23_native_first import (
    TAU, MAX_DOSE_MULTIPLE, B_COMPONENTS, FAMILIES,
    affine_physical, fp32_code, plan_target_native, four_arm_logits,
    score, pick_calibration_features, protocol_digest, seed,
)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1,0]=2.

def make_sae():
    sae=TopKSAE(48,96,4)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.decoder.weight.zero_()
        sae.decoder.bias.zero_()
        for i in range(4):
            sae.encoder.weight[i,i]=1.
            sae.decoder.weight[i,i]=1.
    mean=torch.zeros(48)
    scale=torch.ones(48)
    h=torch.zeros(48,dtype=torch.float32)
    h[:4]=torch.tensor([1.,.8,.6,.4])
    return sae,mean,scale,h


class NativeFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_protocol_and_seed_precede_any_model_input(self):
        self.assertEqual(len(protocol_digest()),64)
        self.assertEqual(len(B_COMPONENTS),4)
        self.assertEqual(FAMILIES["smc23_novel30"],(202633030000,30))
        self.assertNotEqual(seed("X",2),seed("X",3))

    def test_exact_target_recode_and_four_active_sham(self):
        sae,mean,scale,h=make_sae()
        cov,dec=affine_physical(sae,mean,scale)
        info,kill,sham=plan_target_native(h,sae,mean,scale,cov,dec,"fixture|0")
        self.assertEqual(info["status"],"eligible")
        self.assertEqual(info["feature"],0)
        self.assertEqual(info["active_ids"],[0,1,2,3])
        self.assertEqual(info["candidate_count"],33)
        self.assertLessEqual(info["native_decoder_contribution"]*0.99,float(kill.norm()))
        self.assertLess(float(kill.norm()),MAX_DOSE_MULTIPLE)
        self.assertAlmostEqual(float(kill.norm()),float(sham.norm()),places=12)
        clean=fp32_code(sae,mean,scale,h)
        killed=fp32_code(sae,mean,scale,h.double()-kill)
        fake=fp32_code(sae,mean,scale,h.double()-sham)
        self.assertEqual(float(killed[0]),0.)
        self.assertTrue(torch.allclose(killed[1:4],clean[1:4],atol=1e-6))
        self.assertTrue(torch.allclose(fake,clean,atol=1e-6))
        self.assertLess(info["sham_active_max_recode_drift"],1e-5)
        self.assertTrue(torch.equal(kill,plan_target_native(h,sae,mean,scale,cov,dec,"fixture|0")[1]))
        self.assertAlmostEqual(TAU,.02)

    def test_native_forward_and_truth_margin_damage(self):
        sae,mean,scale,h=make_sae()
        cov,dec=affine_physical(sae,mean,scale)
        info,kill,sham=plan_target_native(h,sae,mean,scale,cov,dec,"fixture|1")
        model=TinyTeacher().eval()
        stage=torch.zeros((81,48),dtype=torch.float32)
        stage[17]=h
        logits=four_arm_logits(model,stage,17,kill,sham)
        self.assertTrue(torch.equal(logits[0],logits[1]))
        self.assertGreater(float(logits[0,1]),float(logits[2,1]))
        self.assertAlmostEqual(float(logits[3,1]),float(logits[0,1]),places=10)
        self.assertLess(float(logits[2,1]),0.)
        self.assertGreater(float(logits[4,1]),float(logits[0,1]))

    def test_selection_rejects_ties_and_invalid_decoder_budget(self):
        sae,mu,scale,h=make_sae()
        h[1]=1.
        cov,dec=affine_physical(sae,mu,scale)
        info,k,s=plan_target_native(h,sae,mu,scale,cov,dec,"fixture|tie")
        self.assertEqual(info["status"],"incomplete_or_tied_topfour")
        self.assertIsNone(k)
        h[1]=.8
        dec=dec*1e-5
        info,k,s=plan_target_native(h,sae,mu,scale,cov,dec,"fixture|dose")
        self.assertEqual(info["status"],"dose_cap")
        self.assertIsNone(s)

    @staticmethod
    def fixture_rows(n=64):
        rows=[]
        for p in range(n):
            rows.append({"puzzle_index":p,"grid_seed":202633030000+p,
              "goals":[{"status":"eligible","feature":11,
                        "paired_margin_effect":.75,
                        "kill_margin_damage":.8,"sham_margin_damage":.05,
                        "abs_specificity":.75,
                        "clean_correct":True,
                        "kill_clean_correct_to_wrong":True,
                        "sham_clean_correct_to_wrong":False,
                        "input_plan":{"dose":.5,"inactive_preweighted_mismatch":.1}}
                        ]+[{"status":"dose_cap","feature":12} for _ in range(7)]})
        return rows

    def test_cluster_statistics_and_adjusted_eight_cell_gate(self):
        rows=self.fixture_rows()
        r=score(rows,"newB_501_SAE_71","smc23_novel30")
        self.assertEqual(r["eligible_goals"],64)
        self.assertEqual(r["eligible_puzzles"],64)
        self.assertEqual(r["cluster_bootstrap_draws"],2000)
        self.assertTrue(r["registered_pass"])
        self.assertTrue(all(abs(v-.75)<1e-12 for v in r["bonferroni_99_375_ci"]))
        self.assertEqual(r["kill_correct_to_wrong"],64)
        under=score(rows[:10],"newB_501_SAE_71","smc23_novel30")
        self.assertEqual(under["status"],"underpowered_coverage")
        self.assertFalse(under["registered_pass"])
        src=score(rows,"pretrainedA_71","smc23_novel30")
        self.assertFalse(src["registered_pass"])

    def test_beneficial_feature_selection_is_calibration_only(self):
        rows=self.fixture_rows()
        c=pick_calibration_features(rows)
        self.assertEqual(c["frozen_ids"],[11])
        self.assertEqual(c["qualified_before_cap"],1)
        s=score(rows,"newB_501_SAE_71","smc23_novel30",chosen={11})
        self.assertEqual(s["eligible_goals"],64)
        self.assertIsNone(s["bonferroni_99_375_ci"])
        self.assertFalse(s["registered_pass"])


if __name__=="__main__":
    unittest.main()
