"""SMC-10C unit checks for interval local-Lipschitz transference."""
import sys
import unittest
from pathlib import Path
from fractions import Fraction as Q

from flint import arb,ctx
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc10_local_lipschitz import (
    exact_effect_norm_interval,validated_lipschitz_on_ball,
    certified_effect_interval,rank_interval_effects
)


class FakeNet:
    def transport(self,tape,radius,dimensions):
        assert dimensions==1
        assert tape==['f','g']
        return [
            {'name':'f','a':arb(2),'R':arb(1),
             'fallback':False},
            {'name':'g','a':arb(3),'R':arb(6),
             'fallback':False},
        ]


class LipschitzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=160

    def test_exact_transport_product_and_metadata(self):
        lip,meta=validated_lipschitz_on_ball(FakeNet(),['f','g'],arb(5),Q(1,10))
        self.assertEqual(lip,arb(30))
        self.assertEqual(meta['layers_validated'],2)
        self.assertEqual(meta['layernorm_epsilon_floor_fallbacks'],0)

    def test_finite_robust_output_effect_interval(self):
        norm=exact_effect_norm_interval(arb(4))
        self.assertEqual(norm,(arb(2),arb(2)))
        a,error=certified_effect_interval(norm,arb(2),arb(2),
                                          Q(1,5),Q(1,100),Q(1,100))
        self.assertAlmostEqual(float(a[0]),1.956,places=10)
        self.assertAlmostEqual(float(a[1]),2.044,places=10)
        self.assertAlmostEqual(float(arb(error)),0.044,places=8)
        b,_=certified_effect_interval((arb(1),arb(1)),
                                       arb(2),arb(2),
                                       Q(1,5),Q(1,100),Q(1,100))
        self.assertEqual(rank_interval_effects(a,b),'sae_active_greater')
        self.assertEqual(rank_interval_effects(b,a),'control_greater')

    def test_abstention_and_invalid_input(self):
        a,b=certified_effect_interval((arb(0),arb(0)),arb(5),arb(5),
                                       Q(1,2),Q(1,10),Q(1,10))[0]
        self.assertEqual(a,arb(0))
        self.assertTrue(b>0)
        self.assertEqual(rank_interval_effects((arb(0),arb(3)),
                                               (arb(1),arb(2))),'inconclusive')
        with self.assertRaises(ValueError):
            certified_effect_interval((arb(1),arb(1)),arb(2),arb(2),0,0,0)
        with self.assertRaises(ValueError):
            validated_lipschitz_on_ball(FakeNet(),['f','g'],arb(5),-1)


if __name__=='__main__':
    unittest.main()
