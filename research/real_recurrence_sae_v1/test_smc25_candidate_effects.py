"""SMC-25 source-independent CPU tests on the frozen all-digit causal estimator."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc25_candidate_effects import (
    SMALL,TEMPERATURES,protocol_digest,weighted_median,class_effects,
    candidate_predictions,label_blind_tangent,generate_fresh,FAMILIES,TEACHERS,sha
)
from smc25_native_score import summary,admitted_smc25_stage1
from smc13_family import new_full_solution,dig_unique,count_solutions,rectangle_count


class LinearToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            for i in range(1,10):
                self.head.weight[i,(i-1)%48]=(i+.4)/10


class CandidateEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_predata_frozen_temperature_and_seed(self):
        self.assertEqual(len(protocol_digest()),64)
        self.assertEqual(TEMPERATURES,(.5,1.,2.,4.,8.))
        self.assertEqual(FAMILIES["smc25_novel30"],(202635030000,30))
        self.assertEqual(SMALL,.08)

    def test_weighted_median_is_class_order_stable(self):
        vals=[-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.]
        self.assertEqual(weighted_median(vals,[1./9]*9),0.)
        self.assertEqual(weighted_median(vals,[0.,0.,.6,0.,.4,0.,0.,0.,0.]),-2.)
        self.assertEqual(weighted_median([1.]*9,[1./9]*9),1.)
        with self.assertRaisesRegex(ValueError,"probabilities"):
            weighted_median(vals,[.2]*9)

    def test_all_digit_linear_tangent_matches_native_finite_difference(self):
        model=LinearToyTeacher().eval()
        stage=torch.zeros((81,48),dtype=torch.float32)
        stage[17,0]=1.
        kill=torch.zeros(48,dtype=torch.float64);kill[0]=.5
        sham=torch.zeros(48,dtype=torch.float64);sham[1]=.5
        clean,tangent=label_blind_tangent(model,stage,17,kill,sham)
        effects,rivals=class_effects(clean,tangent)
        self.assertEqual(len(effects),9)
        self.assertEqual(len(rivals),9)
        for y in range(1,10):
            competitor=rivals[y-1]
            vk=stage.double().clone();vs=stage.double().clone()
            vk[17]-=kill;vs[17]-=sham
            with torch.no_grad():
                lk=model.head(vk)[17];ls=model.head(vs)[17]
            finite=float(ls[y]-ls[competitor]-lk[y]+lk[competitor])
            self.assertAlmostEqual(effects[y-1],finite,places=11)

    def test_label_free_posterior_and_rotated_control(self):
        clean=torch.arange(10,dtype=torch.float64)*.5
        slope=torch.arange(10,dtype=torch.float64)/2.
        p=candidate_predictions(clean,slope,temperature=2.)
        self.assertEqual(set(p["predictions"]),{
            "calibrated_median","uniform_median","top_digit_effect",
            "shifted_median","calibrated_mean","uniform_mean","zero_effect"})
        self.assertEqual(p["model_top_digit"],9)
        self.assertAlmostEqual(sum(p["calibrated_digit_distribution"]),1.,places=12)
        self.assertEqual(len(p["rival_for_digit"]),9)
        self.assertEqual(p["rival_for_digit"][-1],8)

    @staticmethod
    def _synthetic_rows():
        rows=[]
        for i in range(64):
            goal={
                "status":"eligible",
                "predictions":{
                    "calibrated_median":1.,"uniform_median":-1.,
                    "top_digit_effect":-.5,"shifted_median":-2.,
                    "calibrated_mean":.5,"uniform_mean":0.,"zero_effect":0.
                },
                "observed":{"true_D":1.},
            }
            rows.append({"puzzle_index":i,"goals":[goal]+[
                {"status":"target_erasure_recode"} for _ in range(7)]})
        return rows

    def test_cluster_resampling_and_adjusted_success_gate(self):
        rows=self._synthetic_rows()
        r=summary(rows,"smc25_novel30","newB_501_SAE_71")
        self.assertEqual(r["eligible_goals"],64)
        self.assertEqual(r["eligible_puzzles"],64)
        self.assertEqual(r["bootstrap_valid"],2000)
        self.assertTrue(r["registered_pass"])
        self.assertEqual(r["calibrated_median_sign_correct"],64)
        self.assertEqual(r["uniform_median_sign_correct"],0)
        for x in r["puzzle_bootstrap99_375_gain"].values():
            self.assertGreater(x[0],0)
        short=summary(rows[:20]+[{"goals":[{"status":"dose_cap"}]*8}
                                  for _ in range(44)],
                      "smc25_novel30","newB_501_SAE_71")
        self.assertFalse(short["registered_pass"])
        self.assertEqual(short["status"],"underpowered_coverage")

    def test_candidate_rejects_bad_rival_geometry(self):
        with self.assertRaisesRegex(ValueError,"ten finite"):
            class_effects(torch.zeros(9),torch.ones(9))
        with self.assertRaises(ValueError):
            candidate_predictions(torch.zeros(10),torch.ones(10),-1.)

    def test_smc25_canonical_replay_accepts_new_family_names(self):
        model=LinearToyTeacher().eval()
        stage=torch.zeros((64,81,48),dtype=torch.float32)
        item={"shape":[64,81,48],
              "tensor_sha256":sha(stage.contiguous().numpy().tobytes()),
              "tensor":stage.tolist()}
        frozen={"archived_source_stage1":{
            fam:{name:item for name in TEACHERS} for fam in FAMILIES
        }}
        ctx={name:{
            "cohorts":{fam:{"stage1":stage} for fam in FAMILIES},
            "model64":model
        } for name in TEACHERS}
        replay,drift=admitted_smc25_stage1(frozen,ctx)
        self.assertEqual(set(replay),set(FAMILIES))
        for fam in FAMILIES:
            for teacher in TEACHERS:
                self.assertEqual(drift[fam][teacher],0.)
                self.assertEqual(tuple(replay[fam][teacher]["clean"].shape),(64,81,10))

    def test_new_source_seed_has_unique_noncanonical_solution(self):
        y=new_full_solution(202635030000)
        x=dig_unique(y,202635030000+1000000,30)
        self.assertGreater(rectangle_count(y),0)
        self.assertEqual(x.count(0),30)
        self.assertEqual(count_solutions(x,2),1)


if __name__=="__main__":
    unittest.main()
