"""SMC-24 regression suite: frozen teacher gradient, test-label-free prediction,
source-input geometry, native finite effect and independent bootstrap parity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc07_real_sae import TopKSAE
from smc24_gradient_predict import (
    GOALS,PUZZLES,FAMILIES,CAL_SCHEMA,PRED_SCHEMA,RIDGE,
    selected_goal_gradient,fit_ridge,predict_gradient,proto_hash,seed,
)
from smc24_native_score import native_score_case,summarize,error_contrast
from audit_smc24 import independent_summary
from smc23_native_first import affine_physical,plan_target_native
from smc13_family import new_full_solution,dig_unique,count_solutions,rectangle_count


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
        sae.encoder.weight.zero_();sae.encoder.bias.zero_()
        sae.decoder.weight.zero_();sae.decoder.bias.zero_()
        for i in range(4):
            sae.encoder.weight[i,i]=1.
            sae.decoder.weight[i,i]=1.
    stage=torch.zeros(48,dtype=torch.float32)
    stage[:4]=torch.tensor([1.,.8,.6,.4],dtype=torch.float32)
    return sae,torch.zeros(48),torch.ones(48),stage


class GradientPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_protocol_is_predata_frozen_and_seeded(self):
        self.assertEqual(len(proto_hash()),64)
        self.assertEqual(RIDGE,32.)
        self.assertEqual(FAMILIES["smc24_novel30"],(202634030000,30))
        self.assertNotEqual(seed("B501",1),seed("B501",2))
        self.assertIn("calibration",CAL_SCHEMA)
        self.assertIn("label-free",PRED_SCHEMA)

    def test_puzzle_autograd_with_single_token_native_margin(self):
        teacher=TinyTeacher().double().eval().requires_grad_(False)
        stage=torch.zeros((81,48),dtype=torch.float32)
        pos=torch.arange(8,dtype=torch.long)+12
        targets=torch.ones(8,dtype=torch.long)
        grads,correct=selected_goal_gradient(teacher,stage,pos,targets)
        self.assertEqual(tuple(grads.shape),(8,48))
        self.assertTrue(torch.allclose(grads[:,0],torch.full((8,),2.,dtype=torch.float64)))
        self.assertTrue(torch.equal(grads[:,1:],torch.zeros((8,47),dtype=torch.float64)))
        self.assertEqual(correct,[False]*8)
        with self.assertRaises(ValueError):
            selected_goal_gradient(teacher,stage,pos[:3],targets)

    def test_calibrated_ridge_has_fixed_shape_and_shuffled_null(self):
        gen=torch.Generator().manual_seed(20262417)
        H=torch.randn((PUZZLES,GOALS,48),generator=gen,dtype=torch.float64)
        g=0.4*H+torch.arange(48,dtype=torch.float64).view(1,1,48)*.001
        fit=fit_ridge(H,g,seed_id=55)
        self.assertEqual(tuple(torch.tensor(fit["ridge_beta"]).shape),(49,48))
        self.assertEqual(tuple(torch.tensor(fit["slot_shuffled_ridge_beta"]).shape),(49,48))
        self.assertEqual(fit["n_calibration_goals"],512)
        h=H[0,0,:]
        predictions=predict_gradient(fit,h)
        self.assertEqual(set(predictions),{"ridge","mean","shuffled","zero"})
        self.assertLess(float((predictions["ridge"]-g[0,0]).norm()),
                        float((predictions["mean"]-g[0,0]).norm()))
        self.assertGreater(float((predictions["shuffled"]-g[0,0]).norm()),0)
        self.assertTrue(torch.equal(predict_gradient(fit,h)["ridge"],
                                    predict_gradient(fit,h)["ridge"]))

    def test_active_erasure_sham_and_finite_scoring(self):
        model=TinyTeacher().double().eval()
        sae,mu,sig,h=make_sae()
        cov,dec=affine_physical(sae,mu,sig)
        info,kill,sham=plan_target_native(h,sae,mu,sig,cov,dec,"SMC24|synthetic|1")
        self.assertEqual(info["status"],"eligible")
        self.assertAlmostEqual(float(kill.norm()),float(sham.norm()),places=12)
        self.assertEqual(info["feature"],0)
        batch=torch.zeros((81,48),dtype=torch.float32)
        batch[17]=h
        clean=model.head(batch.double())
        result=native_score_case(model,batch,clean,1,17,0,kill,sham)
        self.assertTrue(result["kill_correct_to_wrong"])
        self.assertFalse(result["sham_correct_to_wrong"])
        self.assertGreater(result["true_D"],2.)
        self.assertAlmostEqual(result["margins"]["clean"],result["margins"]["zero_self"])
        self.assertAlmostEqual(result["true_D"],
                               result["margins"]["sham"]-result["margins"]["kill"])

    @staticmethod
    def synthetic_rows():
        template={"status":"eligible",
                  "observed":{"true_D":1.,"kill_correct_to_wrong":True,
                              "sham_correct_to_wrong":False},
                  "predictions":{"ridge":1.,"mean":-1.,"shuffled":0.,"zero":0.}}
        return [{"puzzle_index":i,"goals":[template.copy()]+
                 [{"status":"dose_cap"} for _ in range(7)]}
                for i in range(PUZZLES)]

    def test_independent_corrected_bootstrap_parity(self):
        rows=self.synthetic_rows()
        a=summarize(rows,"newB_501_SAE_71","smc24_novel30")
        b=independent_summary(rows,"newB_501_SAE_71","smc24_novel30")
        self.assertTrue(a["registered_pass"])
        self.assertEqual(a["eligible_goals"],64)
        self.assertEqual(a["cluster_bootstrap_valid"],2000)
        self.assertEqual(a["ridge_sign_correct"],64)
        self.assertEqual(a["mean_sign_correct"],0)
        self.assertEqual(a["puzzle_bootstrap99_375_gain"],b["puzzle_bootstrap99_375_gain"])
        self.assertAlmostEqual(a["puzzle_mean_MAE"]["ridge"],0.)
        self.assertGreater(a["puzzle_mean_ridge_gain"]["mean"],1.)
        self.assertGreater(a["puzzle_mean_ridge_gain"]["shuffled"],.5)
        self.assertFalse(summarize(rows[:30]+[{"goals":[{"status":"dose_cap"}]*8}
                                   for _ in range(34)],
                                   "newB_501_SAE_71","smc24_novel30")["registered_pass"])

    def test_error_contrast_uses_same_native_outcome(self):
        p={"ridge":-.5,"mean":1.,"shuffled":0.,"zero":0.}
        a=error_contrast(-.5,p)
        self.assertAlmostEqual(a["mean"],1.5)
        self.assertAlmostEqual(a["shuffled"],.5)
        self.assertAlmostEqual(a["zero"],.5)
        with self.assertRaises(ValueError):
            error_contrast(0,{"ridge":0})

    def test_new_source_is_noncanonical_unique_and_positive(self):
        grid=new_full_solution(202634030000)
        self.assertGreater(rectangle_count(grid),0)
        x=dig_unique(grid,202634030000+1000000,30)
        self.assertEqual(x.count(0),30)
        self.assertEqual(count_solutions(x,2),1)


if __name__=="__main__":
    unittest.main()
