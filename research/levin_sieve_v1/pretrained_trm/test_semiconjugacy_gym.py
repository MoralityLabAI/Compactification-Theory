"""LSV-18 controls: exact affine semiconjugacy and frozen synthetic TRM smoke."""
import unittest

import torch

from semiconjugacy_gym import (
    LatentModel, SCHEMA, SITES, baselines, collect_trajectories,
    gather_observations, nonclosure_witness, perturbation, run,
    split_for_puzzle, train_candidate,
)
from upstream_arch import TinyRecursiveModel

class SemiconjugacyGymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1801)
        cls.teacher = TinyRecursiveModel().eval()
        cls.inputs = torch.randint(1,10,(8,81))
        cls.inputs[:,0]=0
        cls.targets = torch.randint(1,10,(8,81))

    def test_exact_affine_semiconjugacy_control(self):
        model = LatentModel(48, nonlinear=False).double()
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.zero_()
            model.encoder.weight[:16,:16]=torch.eye(16, dtype=torch.float64)
            model.delta.weight.zero_()
            model.delta.bias.zero_()
            model.delta.weight.copy_((0.8-1)/0.15*torch.eye(16,dtype=torch.float64))
        torch.manual_seed(201)
        x=torch.randn(5,4,48,dtype=torch.float64)
        for step in range(1,4):
            x[:,step,:16]=0.8*x[:,step-1,:16]
        z,_,_,comm=model(x)
        self.assertLess(float(comm.abs().max()),1e-12)
        self.assertEqual(tuple(z.shape),(5,4,16))

    def test_equal_norm_and_offsite_token_invariance(self):
        z=torch.randn(81,48,dtype=torch.float64)
        a=perturbation(z,0,"outside",2.0,71)
        b=perturbation(z,0,"goal",2.0,71)
        self.assertTrue(torch.equal(a[0],z[0]))
        self.assertTrue(torch.equal(b[17],z[17]))
        self.assertAlmostEqual(float((a-z).norm()),2.0,places=12)
        self.assertAlmostEqual(float((b-z).norm()),2.0,places=12)
        with self.assertRaises(ValueError):
            perturbation(z,0,"invalid",2.0,71)

    def test_context_dimensions_and_fixed_splits(self):
        torch.manual_seed(304)
        x=torch.randn(3,4,81,48)
        indices=torch.tensor([0,15,80])
        self.assertEqual(tuple(gather_observations(x,indices,False).shape),(3,4,48))
        self.assertEqual(tuple(gather_observations(x,indices,True).shape),(3,4,192))
        self.assertTrue(torch.allclose(gather_observations(x,indices,False),
                                       gather_observations(x,indices,True)[...,:48]))
        self.assertEqual([split_for_puzzle(i) for i in range(8)],
                         ["train","train","validation","test"]*2)
        with self.assertRaises(ValueError):
            gather_observations(x,torch.tensor([0,1,82]),True)

    def test_teacher_offsite_alias_on_real_architecture(self):
        data=collect_trajectories(self.teacher,self.inputs,self.targets,max_cases=8)
        self.assertEqual(tuple(data["states"].shape),(72,4,81,48))
        self.assertEqual(tuple(data["logits"].shape),(72,4,10))
        alias=nonclosure_witness(data)
        self.assertEqual(alias["pairs"],8)
        self.assertEqual(len(alias["local_only_logit_mse_floor_by_horizon"]),3)
        self.assertTrue(all(x>=0 for x in alias["local_only_logit_mse_floor_by_horizon"]))
        self.assertGreater(alias["max_future_logit_abs"],0.)
        self.assertEqual(len(baselines(data)),2)

    def test_end_to_end_grouped_student_control(self):
        result=run(self.teacher,self.inputs,self.targets,
                   "SYNTHETIC_RANDOM_MODEL","SYNTHETIC_PUZZLES",
                   max_cases=8,max_epochs=8)
        self.assertEqual(result["schema"],SCHEMA)
        self.assertEqual(result["puzzles_per_split"],
                         {"train":4,"validation":2,"test":2})
        self.assertEqual(result["samples_per_split"],
                         {"train":36,"validation":18,"test":18})
        for name,row in result["candidates"].items():
            self.assertEqual(row["test_sample_count"],18)
            self.assertEqual(row["test_puzzles"],2)
            self.assertEqual(len(row["future_teacher_label_agreement"]),3)
            self.assertEqual(len(row["per_puzzle"]),2)
            self.assertGreater(row["trainable_parameters"],0)
            self.assertGreaterEqual(row["scale_normalized_semiconjugacy_mse"],0)
        self.assertEqual(result["candidates"]["context_nonlinear"]["input_feature_dimension"],192)
        self.assertEqual(result["candidates"]["local_linear"]["input_feature_dimension"],48)

    def test_invalid_roster_or_absent_split(self):
        with self.assertRaises(ValueError):
            collect_trajectories(self.teacher,self.inputs,self.targets,max_cases=7)
        with self.assertRaises(ValueError):
            collect_trajectories(self.teacher,self.inputs,self.targets,max_cases=99)

if __name__=="__main__":
    unittest.main(verbosity=2)
