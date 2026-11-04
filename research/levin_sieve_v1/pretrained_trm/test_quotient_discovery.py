"""LSV-17 unit and negative controls; this tiny model is randomly initialized."""
import unittest
import torch
from torch import nn
from quotient_discovery import (SCHEMA,DIRECTIONS,unit,null_projection,goal_history,
    margin_gradients,directions_for_case,pre_ln2_gauge_control,partition,run,train_probes)

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(48)
        self.tok_mix=nn.Sequential(nn.Linear(81,12),nn.GELU(),nn.Linear(12,81))
        self.ln2=nn.LayerNorm(48)
        self.chan_mix=nn.Sequential(nn.Linear(48,16),nn.GELU(),nn.Linear(16,48))
    def forward(self,z):
        z=self.ln1(z+self.tok_mix(z.transpose(1,2)).transpose(1,2))
        return self.ln2(z+self.chan_mix(z))

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(10,48)
        self.pos=nn.Embedding(81,48)
        self.block=TinyBlock()
        self.head=nn.Linear(48,10)
    def reason_states(self,x):
        z=self.emb(x)+self.pos(torch.arange(81))
        out=[z]
        for _ in range(4):
            z=self.block(z)
            out.append(z)
        return out

class QuotientDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(117)
        cls.net=TinyModel().double().eval()

    def test_project_null_is_orthogonal(self):
        grad=torch.tensor([[1.,1.,0.,0.],[0.,1.,1.,0.]],dtype=torch.float64)
        v=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        n=null_projection(v,grad)
        self.assertLess(float((grad@n).abs().max()),1e-10)
        self.assertAlmostEqual(float(n.norm()),1.,places=12)

    def test_future_history_and_gradients(self):
        states=torch.randn(1,81,48,dtype=torch.float64)
        y,order,grad=margin_gradients(self.net,states,17)
        self.assertEqual(tuple(y.shape),(3,10))
        self.assertEqual(tuple(grad.shape),(3,48))
        self.assertEqual(tuple(goal_history(self.net,states,17).shape),(3,10))
        dirs=directions_for_case(grad,112)
        self.assertEqual(set(dirs),set(DIRECTIONS))
        self.assertLess(float(torch.dot(grad[0],dirs["null_one"]).abs()),1e-8)
        self.assertLess(float((grad@dirs["null_all"]).abs().max()),1e-8)

    def test_gauge_control_is_approximately_zero_in_fp64(self):
        z=torch.randn(1,81,48,dtype=torch.float64)
        rec=pre_ln2_gauge_control(self.net,z,17,1.0)
        self.assertLess(rec["gauge_future_max_logit_abs"],1e-10)
        self.assertGreater(rec["transverse_future_max_logit_abs"],1e-10)
        self.assertFalse(any(rec["gauge_flips"]))

    def test_puzzle_split_fixed_before_labels(self):
        self.assertEqual([partition(i) for i in range(8)],
                         ["train","train","validation","test"]*2)
        with self.assertRaises(ValueError):
            unit(torch.zeros(48,dtype=torch.float64))
        with self.assertRaises(ValueError):
            null_projection(torch.ones(48),torch.ones(2,47))

    def test_end_to_end_synthetic_evidence(self):
        torch.manual_seed(142)
        x=torch.randint(1,10,(8,81))
        x[:,0]=0
        target=torch.randint(1,10,(8,81))
        rec=run(self.net,x,target,"SYNTHETIC-NO-CHECKPOINT","SYNTHETIC-ROSTER",
                max_cases=8,radii=(0.2,1.0),probe_epochs=12)
        self.assertEqual(rec["schema"],SCHEMA)
        self.assertEqual(len(rec["case_rows"]),8*len(DIRECTIONS)*2)
        self.assertEqual(rec["probe"]["samples"],{
            "train":4*(1+len(DIRECTIONS)*2),
            "validation":2*(1+len(DIRECTIONS)*2),
            "test":2*(1+len(DIRECTIONS)*2)})
        self.assertLess(rec["gauge_bus_max_logit_abs"],1e-10)
        self.assertGreater(rec["transverse_bus_max_logit_abs"],1e-10)
        self.assertEqual(len(rec["heldout_aggregates"]),2*len(DIRECTIONS))
        self.assertTrue(all(r["direction_l2"]>0.999999 for r in rec["case_rows"]))
        self.assertTrue(all(len(r["flips"])==3 for r in rec["case_rows"]))
        self.assertEqual(rec["probe"]["models"]["nonlinear"]["trainable_parameters"],
                         48*16+16+16*30+30)

    def test_group_leakage_guard(self):
        X=torch.randn(4,48)
        Y=torch.randn(4,30)
        with self.assertRaisesRegex(ValueError,"partitions"):
            train_probes(X,Y,["train","train","test","test"],max_epochs=2)

if __name__=="__main__":
    unittest.main(verbosity=2)
