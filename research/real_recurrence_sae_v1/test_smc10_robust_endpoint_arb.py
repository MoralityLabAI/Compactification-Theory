"""SMC-10 toy tests of rigorous Euclidean uncertainty boxes (no checkpoint)."""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import torch
from flint import arb,arb_mat,ctx

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc10_robust_endpoint_arb import (
    arb_sym_ball,uncertain_anchor,robust_effects,summary_table
)


class LinearToyNet:
    """Three identity recurrences, exact two-coordinate linear goal readout."""
    def __init__(self):
        self.norms=(None,None)
    def mix(self,z,*args):
        return z,None
    def ln(self,z,*args):
        return z,None
    def logits(self,z,goal):
        return [2*z[goal,0],z[goal,1]]+[arb(0)]*8


class RobustEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=160
        cls.net=LinearToyNet()
        cls.anchor=arb_mat([[arb(0) for _ in range(48)] for _ in range(81)])
        def e(index):
            return [arb(int(j==index)) for j in range(48)]
        cls.directions={
            'sae_active':e(0),'sae_inactive':e(1),
            'isotropic':e(1),'jacobian_optimal':e(0)
        }

    def test_arb_rational_radius_inflates_outward(self):
        b=arb_sym_ball(Q(1,1000000000000))
        self.assertTrue(b.lower()<=arb('-1/1000000000000'))
        self.assertTrue(b.upper()>=arb('1/1000000000000'))
        self.assertEqual(arb_sym_ball(Q(0)),arb(0))
        with self.assertRaises(ValueError):
            arb_sym_ball(Q(-1,8))

    def test_full_goal_token_box_contains_euclidean_uncertainty(self):
        clean=uncertain_anchor(self.anchor,7,None,0,Q(1,32))
        active=uncertain_anchor(self.anchor,7,self.directions['sae_active'],
                                Q(1,5),Q(1,32),Q(1,64))
        self.assertTrue(clean[7,0].lower() <= arb('-1/32'))
        self.assertTrue(clean[7,0].upper() >= arb('1/32'))
        self.assertTrue(active[7,0].lower() <= arb('1/5')-arb('11/320'))
        self.assertTrue(active[7,0].upper() >= arb('1/5')+arb('11/320'))
        self.assertEqual(active[6,0],arb(0))
        self.assertEqual(active[7,1].mid(),arb(0))

    def test_robust_toy_ranking_and_abstention(self):
        for delta in (Q(0),Q(1,10000),Q(1,100)):
            row=robust_effects(self.net,self.anchor,7,self.directions,
                               Q(1,5),delta,{k:delta for k in self.directions})
            self.assertEqual(row['comparisons']['isotropic']['proof'],
                             'sae_active_greater')
            self.assertEqual(row['comparisons']['sae_inactive']['proof'],
                             'sae_active_greater')
            self.assertEqual(row['comparisons']['jacobian_optimal']['proof'],
                             'inconclusive')
            self.assertFalse(row['failures'])
        too_wide=robust_effects(self.net,self.anchor,7,self.directions,
                                Q(1,5),Q(1,4),{k:Q(1,4) for k in self.directions})
        self.assertEqual(too_wide['comparisons']['isotropic']['proof'],
                         'inconclusive')

    def test_report_aggregation_and_input_guard(self):
        row=robust_effects(self.net,self.anchor,7,self.directions,
                           Q(1,5),Q(0))
        cases=[{'radius':str(Q(1,5)),'anchor_l2_budget':'0',
                'comparisons':row['comparisons']}]
        summary=summary_table(cases,('0.2',),('0',))
        self.assertEqual(summary['1/5|0|isotropic']['active_greater'],1)
        with self.assertRaises(ValueError):
            uncertain_anchor(self.anchor,7,self.directions['sae_active'],0,0,0)
        with self.assertRaises(ValueError):
            uncertain_anchor(self.anchor,7,None,0,-1)
        with self.assertRaises(ValueError):
            robust_effects(self.net,self.anchor,7,self.directions,'0',0)


if __name__=='__main__':
    unittest.main()
