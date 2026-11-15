"""SMC-08B Arb interval wiring tests; source checkpoint is tested in CI."""
from fractions import Fraction
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from flint import arb,arb_mat
import torch
from smc08_local_arb import certified_radius,log10_upper,largest_certified_sign_radius


class IdentityNet:
    def transport(self,tape,radius,dimensions):
        assert dimensions==1
        return [{'name':'identity','R':arb(1)/10,
                 'defect':arb(0),'fallback':False}]




class QuadraticNet:
    def transport(self,tape,radius,dimensions):
        r=arb(radius.numerator)/radius.denominator
        return [{'name':'quadratic','R':r,
                 'defect':r*r/2,'fallback':False}]

class LocalArbTests(unittest.TestCase):
    def test_exact_linear_margin_sign_enclosure(self):
        rows=[[arb(0) for _ in range(48)] for _ in range(81)]
        rows[3][0]=arb(3)
        G=arb_mat(rows)
        d=torch.zeros(48,dtype=torch.float64)
        d[0]=1
        cert=certified_radius(IdentityNet(),[None],G,{'q':[arb(1)]},3,
                              d,Fraction(1),'1e-4')
        self.assertEqual(cert['certified_displacement_sign'],'positive')
        self.assertAlmostEqual(cert['linear_signed_margin_mid_fp64'],0.0003,places=12)
        self.assertEqual(cert['rigorous_remainder_upper_arb'],'0')
        self.assertIsNone(cert['rigorous_remainder_log10'])
        self.assertFalse(cert['ln_epsilon_floor_used'])
        d[0]=-1
        cert=certified_radius(IdentityNet(),[None],G,{'q':[arb(1)]},3,
                              d,Fraction(1),'1e-4')
        self.assertEqual(cert['certified_displacement_sign'],'negative')


    def test_monotone_frontier_bisection_finds_quadratic_threshold(self):
        rows=[[arb(0) for _ in range(48)] for _ in range(81)]
        rows[3][0]=arb(1)
        d=torch.zeros(48,dtype=torch.float64);d[0]=1
        z=largest_certified_sign_radius(
            QuadraticNet(),[None],arb_mat(rows),{'q':[arb(1)]},3,
            d,Fraction(1),limit=Fraction(3),iterations=18)
        self.assertFalse(z['upper_limit_is_certified'])
        lower=Fraction(z['certified_radius_lower'])
        upper=Fraction(z['first_inconclusive_radius_upper'])
        self.assertLess(lower,2)
        self.assertGreater(lower,Fraction(199,100))
        self.assertGreaterEqual(upper,2)
        ok=largest_certified_sign_radius(
            IdentityNet(),[None],arb_mat(rows),{'q':[arb(1)]},3,
            d,Fraction(1),limit=Fraction(1),iterations=8)
        self.assertTrue(ok['upper_limit_is_certified'])

    def test_bounds_and_invalid_radius(self):
        self.assertEqual(log10_upper(arb(1000)),3)
        rows=[[arb(0) for _ in range(48)] for _ in range(81)]
        z=torch.zeros(48,dtype=torch.float64);z[0]=1
        out=certified_radius(IdentityNet(),[None],arb_mat(rows),
                             {'q':[arb(1)]},3,z,Fraction(1),'1e-6')
        self.assertEqual(out['certified_displacement_sign'],'inconclusive')
        with self.assertRaises(ValueError):
            certified_radius(IdentityNet(),[None],arb_mat(rows),
                             {'q':[arb(1)]},3,z,Fraction(1),'-1e-6')


if __name__=='__main__':
    unittest.main()
