"""Numerical checks of categorical exponential-family identities."""
from math import log
from random import Random
import unittest
from research.algernon.belief_geometry.info_geometry import (
    log_partition, softmax, fisher, kl_categorical, partition_bregman,
    alpha_tie, curved_tie_y, curved_tie_second_derivative,
)


class InformationGeometryTests(unittest.TestCase):
    def test_softmax_normalization_and_gauge(self):
        a=[-1.2,0.3,5.9]
        p=softmax(a)
        self.assertAlmostEqual(sum(p),1.0)
        for x,y in zip(p,softmax([v+200 for v in a])):
            self.assertAlmostEqual(x,y)

    def test_partition_gradient_and_hessian(self):
        rng=Random(20261104)
        h=1e-4
        for _ in range(100):
            t=[rng.uniform(-3,3) for _ in range(4)]
            p=softmax(t)
            mat=fisher(t)
            for i in range(4):
                right=list(t);left=list(t)
                right[i]+=h;left[i]-=h
                deriv=(log_partition(right)-log_partition(left))/(2*h)
                self.assertAlmostEqual(deriv,p[i],delta=2e-8)
                dr=softmax(right);dl=softmax(left)
                for j in range(4):
                    self.assertAlmostEqual((dr[j]-dl[j])/(2*h),mat[j][i],delta=2e-8)
            for row in mat:
                self.assertAlmostEqual(sum(row),0.0,delta=1e-12)

    def test_kl_bregman(self):
        rng=Random(1205)
        for _ in range(200):
            theta=[rng.uniform(-4,4) for _ in range(3)]
            eta=[rng.uniform(-4,4) for _ in range(3)]
            val=kl_categorical(theta,eta)
            breg=partition_bregman(theta,eta)
            self.assertAlmostEqual(val,breg,delta=1e-12)
            self.assertGreater(val,-1e-12)

    def test_curved_decision_boundary(self):
        # STOP: reward on state 2, CONTINUE: reward on states 0 and 1.
        a=(1,1,0); b=(0,0,1)
        for x in [-3,-2,-1,-0.5]:
            y=curved_tie_y(x)
            self.assertAlmostEqual(alpha_tie(a,b,[x,y,0]),0.0,delta=1e-14)
            self.assertLess(curved_tie_second_derivative(x),0)
            h=0.001
            numerical=(curved_tie_y(x+h)-2*y+curved_tie_y(x-h))/(h*h)
            self.assertAlmostEqual(numerical,curved_tie_second_derivative(x),delta=2e-4)

    def test_invalid(self):
        with self.assertRaises(ValueError):log_partition([])
        with self.assertRaises(ValueError):curved_tie_y(0)
        with self.assertRaises(ValueError):kl_categorical([0,1],[0,1,2])


if __name__ == '__main__':
    unittest.main()
