"""LSV-11: exact rank lower bounds for differentiable goal factorization.

Exact examples test finite-dimensional linear algebra, alongside the numerical
SVD diagnostic's declared rank threshold. They are different evidence classes.
"""
import unittest
from fractions import Fraction as Q
import torch
from goal_horizon import rref
from run_goal_horizon import tangent_profile


class GoalFactorizationRankTests(unittest.TestCase):
    def test_polynomial_goal_history_has_rank_three(self):
        # y0=x1, y1=x1+x2^2, y2=x1+x2^2+x3.
        # At x=(0,1,0), Jacobian rows are as follows.
        J=((1,0,0),(1,2,0),(1,2,1))
        self.assertEqual(len(rref(J)),3)
        self.assertEqual(Q(1)*Q(2)*Q(1),2)  # nonzero determinant witness

    def test_smooth_two_dimensional_factorization_has_rank_at_most_two(self):
        # pi=(x1+x2,x2+x3); y=(pi0,pi0+2*pi1,3*pi1).
        Dpi=((1,1,0),(0,1,1))
        Dreadout=((1,0),(1,2),(0,3))
        J=tuple(tuple(sum((v*Dpi[k][j] for k,v in enumerate(row)),Q(0))
                      for j in range(3)) for row in Dreadout)
        self.assertEqual(J,((Q(1),Q(1),Q(0)),(Q(1),Q(3),Q(2)),(Q(0),Q(3),Q(3))))
        self.assertEqual(len(rref(J)),2)

    def test_fp64_tangent_rank_identity_and_full_controls(self):
        eye=torch.eye(4,dtype=torch.float64)
        same=torch.ones((1,4,20),dtype=torch.float64)
        full=torch.zeros((1,4,20),dtype=torch.float64)
        full[0,:,:4]=eye
        self.assertEqual(tangent_profile(same)['rank'].tolist(),[1])
        self.assertEqual(tangent_profile(full)['rank'].tolist(),[4])
        near=torch.zeros((1,4,20),dtype=torch.float64)
        near[0,:,:4]=torch.diag(torch.tensor([1.,1e-2,1e-3,1e-4],dtype=torch.float64))
        self.assertEqual(tangent_profile(near,rank_relative_tolerance=1e-3)['rank'].tolist(),[2])

    def test_rank_threshold_is_explicit_and_input_is_finite(self):
        with self.assertRaises(ValueError):tangent_profile(torch.zeros((1,4,3)),0)
        with self.assertRaises(ValueError):tangent_profile(torch.tensor([[[float('nan')]]]))
        with self.assertRaises(ValueError):tangent_profile(torch.zeros((0,4,3)))

if __name__=='__main__':unittest.main()
