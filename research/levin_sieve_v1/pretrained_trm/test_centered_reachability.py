"""Tests: exact geometric formulas and sampled inclusion of ideal FP64 model."""
import unittest
import torch
import torch.nn.functional as F
from centered_reachability import (layernorm_difference_radius,centered_ln,
                                    centered_block,certify_centered,robust_classes)
from upstream_arch import TinyRecursiveModel


class TestCenteredReachability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261104)
        torch.set_num_threads(1)

    def test_zero_radius_stays_zero_through_three_steps(self):
        model=TinyRecursiveModel().double().eval()
        z=torch.randn((1,81,48),dtype=torch.float64)
        c,r=z,torch.zeros_like(z)
        for _ in range(3):
            c,r=centered_block(c,r,model.block)
            self.assertEqual(float(r.max()),0.)
        predicted,meta,_,r=certify_centered(model,z,0)
        self.assertEqual(meta['certified_cells'],81)
        self.assertEqual(float(r.max()),0.)
        self.assertTrue(bool((predicted==model.head(c).argmax(-1)).all()))

    def test_layernorm_radius_bounds_random_perturbations(self):
        ln=torch.nn.LayerNorm(48).double().eval()
        with torch.no_grad():ln.weight.copy_(torch.randn_like(ln.weight))
        for eps in (0,1e-7,0.01,0.5):
            c=torch.randn((2,5,48),dtype=torch.float64)
            r=torch.rand_like(c)*eps
            bound=layernorm_difference_radius(c,r,ln)
            nom=ln(c)
            for seed in (12,13,14,15):
                rng=torch.Generator().manual_seed(seed)
                delta=(torch.rand(c.shape,dtype=torch.float64,generator=rng)*2-1)*r
                actual=ln(c+delta)
                self.assertTrue(bool(((actual-nom).abs()<=bound+1e-10).all()))

    def test_centered_block_encloses_random_damages(self):
        m=TinyRecursiveModel().double().eval()
        z=torch.randn(1,81,48,dtype=torch.float64)
        for eps in (1e-5,0.03,0.3):
            r=torch.full_like(z,eps)
            c,b=centered_block(z,r,m.block)
            for seed in (1,9,18):
                rng=torch.Generator().manual_seed(seed)
                perturb=(torch.rand(z.shape,dtype=torch.float64,generator=rng)*2-1)*eps
                delta=(m.block(z+perturb)-c).abs()
                self.assertTrue(bool((delta<=b+1e-9).all()))

    def test_mask_only_one_cell(self):
        m=TinyRecursiveModel().double().eval()
        z=torch.randn(1,81,48,dtype=torch.float64)
        mask=torch.zeros_like(z,dtype=torch.bool);mask[:,15]=True
        labels,rec,c,r=certify_centered(m,z,0.001,mask=mask)
        self.assertEqual(rec['certified_cells'],int((labels>=0).sum()))
        self.assertGreaterEqual(float(r.amin()),0.)

    def test_margin_proof_uses_competing_weight_difference(self):
        h=torch.nn.Linear(2,3,bias=True).double()
        with torch.no_grad():
            h.weight.copy_(torch.tensor([[2.,0.],[1.,1.],[-1.,-1.]]))
            h.bias.zero_()
        c=torch.tensor([[[3.,0.]]],dtype=torch.float64)
        classes,meta=robust_classes(c,torch.tensor([[[.1,.1]]],dtype=torch.float64),h)
        self.assertEqual(classes.tolist(),[[0]])
        classes,_=robust_classes(c,torch.tensor([[[5.,5.]]],dtype=torch.float64),h)
        self.assertEqual(classes.tolist(),[[-1]])

    def test_invalid_inputs(self):
        m=TinyRecursiveModel().double()
        with self.assertRaises(ValueError):certify_centered(m,torch.zeros(1,48),0.02)
        with self.assertRaises(ValueError):certify_centered(m,torch.zeros(1,81,48),-1)
        with self.assertRaises(ValueError):certify_centered(m,torch.zeros(1,81,48),0.02,mask=torch.zeros(1,81,48))


if __name__=='__main__':unittest.main()
