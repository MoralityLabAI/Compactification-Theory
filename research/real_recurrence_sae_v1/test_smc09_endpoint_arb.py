"""SMC-09 unit tests of validated endpoint ordering without external weights."""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import torch
from flint import arb, arb_mat, ctx

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc09_endpoint_arb import (
    normalized_arb_direction,shifted_anchor,endpoint_logits,
    squared_logit_effect,classify_ordering
)


class LinearToyNet:
    """Identity three-stage block, linear ten-logit head."""
    def __init__(self):
        self.norms=(None,None)
    def mix(self,x,*args):
        return x,None
    def ln(self,x,*args):
        return x,None
    def logits(self,x,goal):
        return [2*x[goal,0],x[goal,1]]+[arb(0)]*8


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=160
        cls.net=LinearToyNet()
        cls.anchor=arb_mat([[arb(0) for _ in range(48)] for _ in range(81)])

    def test_fixed_exact_radius_and_normalized_arb_direction(self):
        a=torch.zeros(48,dtype=torch.float64)
        a[0]=3
        u,provenance=normalized_arb_direction(a)
        self.assertEqual(provenance['dyadic_input_norm_sq'],'9')
        norm=sum((x*x for x in u),arb(0))
        self.assertTrue(norm.lower()<=1<=norm.upper())
        h=shifted_anchor(self.anchor,7,u,Q(1,5))
        y=endpoint_logits(self.net,h,7)
        self.assertTrue(y[0].lower()<=Q(2,5)<=y[0].upper())
        self.assertTrue(y[1].lower()<=0<=y[1].upper())

    def test_equal_norm_endpoint_causal_effect_certificates(self):
        a=torch.zeros(48,dtype=torch.float64);a[0]=1
        b=torch.zeros(48,dtype=torch.float64);b[1]=1
        u,_=normalized_arb_direction(a)
        v,_=normalized_arb_direction(b)
        baseline=endpoint_logits(self.net,self.anchor,7)
        effects=[]
        for direction in (u,v):
            out=endpoint_logits(self.net,shifted_anchor(self.anchor,7,direction,Q(1,5)),7)
            effects.append(squared_logit_effect(out,baseline))
        z=classify_ordering(*effects)
        self.assertEqual(z['proof'],'sae_active_greater')
        self.assertTrue(z['squared_effect_gap_midpoint_float']>0)
        inverted=classify_ordering(effects[1],effects[0])
        self.assertEqual(inverted['proof'],'control_greater')
        tied=classify_ordering(effects[0],effects[0])
        self.assertEqual(tied['proof'],'inconclusive')

    def test_input_errors_and_true_null_direction(self):
        with self.assertRaises(ValueError):
            normalized_arb_direction(torch.zeros(48,dtype=torch.float64))
        with self.assertRaises(ValueError):
            normalized_arb_direction(torch.ones(47,dtype=torch.float64))
        a=torch.zeros(48,dtype=torch.float64);a[2]=1
        u,_=normalized_arb_direction(a)
        base=endpoint_logits(self.net,self.anchor,7)
        change=squared_logit_effect(
            endpoint_logits(self.net,shifted_anchor(self.anchor,7,u,'0.2'),7),base)
        self.assertTrue(change.lower()<=0<=change.upper())
        with self.assertRaises(ValueError):
            shifted_anchor(self.anchor,7,u,'-0.2')
        with self.assertRaises(ValueError):
            endpoint_logits(self.net,self.anchor,99)


if __name__=='__main__':
    unittest.main()
