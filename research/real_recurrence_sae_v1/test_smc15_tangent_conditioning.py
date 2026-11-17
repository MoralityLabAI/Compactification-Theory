"""Source-independent SMC-15 CPU tests; no archived teacher artifacts required."""
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smc15_tangent_conditioning import (
    SIGMA_CUTOFF, SOURCES, PAIRS,
    normalised_frame, deterministic_sham, frozen_two_features,
    native_logits_from_stage, score_goal, summary_component, cross_coactivity,
)


class TinyLinearTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Identity()
        self.head = nn.Linear(48, 10, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1,0] = 1.
            self.head.weight[2,1] = 1.
            self.head.weight[3,0] = -.4
            self.head.weight[4,1] = .2


class MockSAE:
    def encode(self, x):
        z = torch.zeros(*x.shape[:-1], 96)
        z[0, 8] = 4.
        z[0, 10] = 3.
        z[0, 21] = 2.
        z[1, 8] = 4.
        z[1, 10] = 3.
        z[1, 21] = 3.
        return z


class SMC15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unit_normalised_tangent_rank(self):
        x = torch.zeros(9, dtype=torch.float64)
        y = torch.zeros(9, dtype=torch.float64)
        x[0] = 1
        y[1] = 1
        frame = normalised_frame(x,y)
        self.assertIsNotNone(frame)
        sig, vh, norms = frame
        expected = torch.linalg.svdvals(torch.stack((
            (x-x.mean())/(x-x.mean()).norm(),
            (y-y.mean())/(y-y.mean()).norm()), dim=1))
        self.assertTrue(torch.allclose(sig,expected,atol=1e-12))
        self.assertEqual(tuple(vh.shape),(2,2))
        self.assertAlmostEqual(float(sig[-1]),math.sqrt(1-1/8),places=10)
        self.assertIsNone(normalised_frame(torch.zeros(9),y))
        self.assertAlmostEqual(SIGMA_CUTOFF,.1)

    def test_goal_pair_selection_is_observable_only(self):
        sae=MockSAE()
        stage=torch.zeros((81,48),dtype=torch.float32)
        indices, valid, vals=frozen_two_features(sae,torch.zeros(48),torch.ones(48),
                                                   stage,torch.tensor([0,1]))
        self.assertEqual(indices.tolist(),[[8,10],[8,10]])
        self.assertEqual(valid.tolist(),[True,False])
        self.assertEqual(vals[0].tolist(),[4.,3.,2.])

    def test_normalized_sham_physical_dose_and_orthogonality(self):
        e=torch.eye(48,dtype=torch.float64)
        sham=deterministic_sham(e[0],e[1],.20*math.sqrt(2),seed=1507)
        self.assertIsNotNone(sham)
        self.assertAlmostEqual(float(sham.norm()),.20*math.sqrt(2),places=12)
        self.assertAlmostEqual(float(sham@e[0]),0,places=12)
        self.assertAlmostEqual(float(sham@e[1]),0,places=12)
        self.assertTrue(torch.equal(sham,deterministic_sham(e[0],e[1],
                         .20*math.sqrt(2),seed=1507)))
        self.assertIsNone(deterministic_sham(e[0],e[0],.20,seed=1507))

    def test_native_fixed_rival_and_joint_linear_additivity(self):
        model=TinyLinearTeacher().eval()
        stage=torch.zeros((81,48),dtype=torch.float64)
        clean=native_logits_from_stage(model,stage.unsqueeze(0))[0]
        table=torch.zeros((96,48),dtype=torch.float64)
        table[0,0]=1
        table[1,1]=1
        row=score_goal(model,stage,clean,token=12,truth=1,ids=[0,1],
                       vals=[4,3,2],eligible=True,unit_table=table,sham_seed=55)
        self.assertEqual(row["status"],"measured")
        self.assertEqual(row["selected_atoms"],[0,1])
        self.assertAlmostEqual(row["native_joint_margin_nonadditivity"],0,places=12)
        self.assertAlmostEqual(row["native_sham_margin_change"],0,places=12)
        self.assertAlmostEqual(row["joint_physical_norm"],math.sqrt(2)*.2,places=12)
        self.assertEqual(row["truth_digit"],1)
        self.assertTrue(row["clean_digit_correct"] is False)
        x=score_goal(model,stage,clean,12,1,[0,1],[4,3,3],False,table,55)
        self.assertEqual(x["status"],"abstain_top2_not_strict")

    def test_underpowered_and_puzzle_cluster_reporting(self):
        def goal(low, ratio):
            return {"status":"measured", "condition_low":low,
                    "minmax_finite_response_ratio":ratio,
                    "native_joint_margin_nonadditivity":0.,
                    "sigma_output_min":.01 if low else .5}
        rows=[{"goal_trials":[goal(True,.2),goal(False,.8)]} for _ in range(40)]
        s=summary_component(rows,"smc15_novel30",SOURCES[0])
        self.assertEqual(s["n_low_conditioning_puzzles"],40)
        self.assertEqual(s["n_high_conditioning_goals"],40)
        self.assertAlmostEqual(s["high_minus_low_puzzle_mean"],.6)
        self.assertEqual(s["puzzle_bootstrap95_high_minus_low"],[.6,.6])
        self.assertTrue(s["registered_conditional_gate"])
        under=summary_component(rows[:4],"pilot",SOURCES[0])
        self.assertEqual(under["status"],"underpowered_groups")

    def test_cross_model_coactivity_uses_only_support(self):
        dummy={}
        source=PAIRS[0][0]
        for i in range(4):
            a,b=PAIRS[i]
            dummy[a]=[{"selection_strict_mask":[True]*8,
                       "selected_top2":[[0,1]]*8}]
        frozen={i:{"matched":list(range(96))} for i in range(4)}
        supports={}
        for i in range(4):
            b=PAIRS[i][1]
            v=torch.zeros((1,8,96),dtype=torch.float32)
            v[:,:,0]=1
            v[:,:5,1]=1
            supports[b]={"pilot":{"goal_support":v}}
        report=cross_coactivity(dummy,supports,frozen,"pilot")
        self.assertEqual(len(report),4)
        self.assertTrue(all(x["n_both_mapped_target_active"]==5 for x in report))
        self.assertTrue(all(x["n_source_strict"]==8 for x in report))


if __name__ == "__main__":
    unittest.main()
