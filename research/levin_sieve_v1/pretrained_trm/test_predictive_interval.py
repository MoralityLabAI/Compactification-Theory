"""CPU tests for conditional real-valued interval reachability operators."""
import copy
import unittest
import torch
import torch.nn.functional as F

from upstream_arch import TinyRecursiveModel
from predictive_interval import (Interval, affine_box, gelu_box, layernorm_box,
                                 block_box, logit_box, certified_output_classes,
                                 predictive_goal_region, invariant_box_check)


class TestPredictiveInterval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(20261104)

    def assertEncloses(self, tube, x, slack=1e-9):
        self.assertTrue(torch.isfinite(tube.lo).all().item())
        self.assertTrue(torch.isfinite(tube.hi).all().item())
        self.assertTrue(bool((tube.lo <= x+slack).all() and (x <= tube.hi+slack).all()))

    def test_exact_point_affine_box(self):
        layer=torch.nn.Linear(3,2).double()
        x=torch.randn(2,4,3,dtype=torch.float64)
        tube=Interval.around(x,0)
        self.assertEncloses(affine_box(tube,layer),layer(x))

    def test_affine_interval_all_corners(self):
        layer=torch.nn.Linear(3,2).double()
        x=torch.randn(1,1,3,dtype=torch.float64)
        tube=Interval.around(x,0.2)
        out=affine_box(tube,layer)
        for i in range(8):
            choice=torch.tensor([bool(i>>j & 1) for j in range(3)]).reshape(1,1,3)
            point=torch.where(choice,tube.hi,tube.lo)
            self.assertEncloses(out,layer(point))

    def test_gelu_enclosure_points_and_intervals(self):
        center=torch.tensor([[[-7.,-3.,-0.9,-0.1,0.,0.2,1.,3.,8.]]],dtype=torch.float64)
        for eps in (0.0,0.001,0.2,0.7):
            tube=Interval.around(center,eps)
            out=gelu_box(tube)
            for a in torch.linspace(0,1,9):
                p=tube.lo*(1-a)+tube.hi*a
                self.assertEncloses(out,F.gelu(p),slack=1e-9)

    def test_layernorm_bounds_random_points(self):
        layer=torch.nn.LayerNorm(5).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([1.2,-0.4,0.5,0.0,2.0]))
            layer.bias.copy_(torch.tensor([0.,2.,-1.,1.,0.]))
        x=torch.randn(2,3,5,dtype=torch.float64)
        for eps in (0.,0.01,0.4,3.):
            tube=Interval.around(x,eps)
            out=layernorm_box(tube,layer)
            for a in (0.,0.25,0.5,1.):
                y=tube.lo*(1-a)+tube.hi*a
                self.assertEncloses(out,layer(y))

    def test_complete_pretrained_architecture_interval_contains_random_perturbations(self):
        m=TinyRecursiveModel().double().eval()
        z=torch.randn(1,81,48,dtype=torch.float64)
        eps=0.01
        interval=Interval.around(z,eps)
        bound=block_box(interval,m.block)
        for seed in (1,2,3):
            gen=torch.Generator().manual_seed(seed)
            x=z+(torch.rand(z.shape,generator=gen,dtype=torch.float64)*2-1)*eps
            self.assertEncloses(bound,m.block(x),slack=1e-8)

    def test_tighten_to_layernorm_absolute_bound(self):
        m=TinyRecursiveModel().double().eval()
        tube=Interval.around(torch.randn(1,81,48,dtype=torch.float64)*1e4,1e5)
        out=layernorm_box(tube,m.block.ln1)
        max_gamma=float(m.block.ln1.weight.detach().abs().max())
        max_beta=float(m.block.ln1.bias.detach().abs().max())
        self.assertLessEqual(float(out.lo.abs().amax()),7*max_gamma+max_beta+1e-6)
        self.assertLessEqual(float(out.hi.abs().amax()),7*max_gamma+max_beta+1e-6)

    def test_masked_radius_restricts_to_perturbed_cell(self):
        x=torch.zeros(1,81,48,dtype=torch.float64)
        mask=torch.zeros_like(x,dtype=torch.bool)
        mask[:,10,:]=True
        a=Interval.around(x,0.3,mask=mask)
        self.assertGreater(float(a.hi[:,10,:].amax()),0.29)
        self.assertLess(float(a.hi[:,9,:].amax()),1e-12)
        with self.assertRaises(ValueError):Interval.around(x,0.3,mask.float())

    def test_goal_classes_only_when_strictly_separated(self):
        lo=torch.tensor([[[2.,1.,0.],[1.,1.,2.]]],dtype=torch.float64)
        hi=torch.tensor([[[3.,1.5,0.],[2.,2.,3.]]],dtype=torch.float64)
        self.assertEqual(certified_output_classes(Interval(lo,hi)).tolist(),[[0,-1]])

    def test_predicted_readout_is_consistent_with_unperturbed_forward(self):
        m=TinyRecursiveModel().eval()
        x=torch.randint(0,10,(1,81))
        z=m.reason_states(x)[1]
        classes,stats=predictive_goal_region(m,z,0.,remaining_steps=2)
        self.assertEqual(stats['n_cells'],81)
        exact_model=copy.deepcopy(m).double().eval()
        q=z.double()
        with torch.no_grad():
            q=exact_model.block(exact_model.block(q))
            target=exact_model.head(q).argmax(-1)
        certified=(classes>=0)
        self.assertTrue(bool((classes[certified]==target[certified]).all()))

    def test_constant_block_invariant_goal_box(self):
        m=TinyRecursiveModel().double().eval()
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.head.bias[3]=2
        box=Interval.around(torch.zeros(1,81,48,dtype=torch.float64),0.5)
        expected=torch.full((1,81),3,dtype=torch.int64)
        r=invariant_box_check(box,m.block,m.head,expected)
        self.assertTrue(r['stable_all_future_steps_conditional'])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):Interval(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64))
        with self.assertRaises(ValueError):Interval.around(torch.zeros(2),-1)
        m=TinyRecursiveModel().eval()
        with self.assertRaises(ValueError):predictive_goal_region(m,torch.randn(1,48),0.02)


if __name__=='__main__':unittest.main()
