import random
import unittest
from fractions import Fraction as Q
from goal_projected import (scored_affine_recurrence,independent_corner_oracle,
                            square_matrix,exact)

class ProjectedSensitivityTests(unittest.TestCase):
    def test_goal_stable_despite_unbounded_hidden_amplification(self):
        A=((10,10),(-10,-10))
        readout=((1,1),(0,0))
        result=scored_affine_recurrence(A,(0,0),(0,0),readout,(1,0),1,(1,1))
        self.assertEqual(result['winner'],0)
        self.assertTrue(result['certified'])
        self.assertEqual(result['comparisons'][0]['gradient'],(Q(0),Q(0)))
        self.assertEqual(result['comparisons'][0]['max_margin_decrease'],Q(0))
        self.assertGreater(max(sum(abs(v) for v in row) for row in A),1)

    def test_exact_oracle_agrees_with_all_positive_certificates(self):
        random.seed(20261104)
        for i in range(180):
            a=((random.randrange(-3,4),random.randrange(-3,4)),
               (random.randrange(-3,4),random.randrange(-3,4)))
            x=(Q(random.randrange(-3,4),3),Q(random.randrange(-3,4),3))
            b=(Q(random.randrange(-2,3),4),Q(random.randrange(-2,3),4))
            w=((random.randrange(-2,3),random.randrange(-2,3)),
               (random.randrange(-2,3),random.randrange(-2,3)))
            bias=(Q(0),Q(1,3))
            r=(Q(random.randrange(0,6),10),Q(random.randrange(0,6),10))
            steps=i%4
            cert=scored_affine_recurrence(a,b,x,w,bias,steps,r)
            corners=independent_corner_oracle(a,b,x,w,bias,steps,r)
            if cert['certified']:
                for winner,scores in corners:
                    self.assertEqual(winner,cert['winner'])
                    self.assertGreater(scores[cert['winner']],scores[1-cert['winner']])

    def test_certificates_can_fail_on_expanding_goal_direction(self):
        cert=scored_affine_recurrence(((3,0),(0,0)),(0,0),(0,0),
                         ((1,0),(0,0)),(1,0),2,(1,0))
        self.assertFalse(cert['certified'])
        self.assertEqual(cert['comparisons'][0]['max_margin_decrease'],Q(9))

    def test_input_validation_and_exact_fraction(self):
        with self.assertRaises(ValueError):square_matrix(((1,2),(3,)))
        with self.assertRaises(TypeError):exact(0.25)
        with self.assertRaises(ValueError):scored_affine_recurrence(
                ((1,),),(0,),(0,),((1,),(0,)),(1,0),-1,(1,))
        with self.assertRaises(ValueError):scored_affine_recurrence(
                ((1,),),(0,),(0,),((1,),(0,)),(1,0),1,(-1,))

if __name__=='__main__':unittest.main()
