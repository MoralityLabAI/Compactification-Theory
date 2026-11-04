import unittest
from fractions import Fraction as Q
import torch
from sparse_minor import exact_det,greedy_minor


class SparseMinorTests(unittest.TestCase):
    def test_exact_four_by_four_minor(self):
        # Sparse full-rank witness hidden inside 4-by-100 matrix.
        J=torch.zeros((4,100),dtype=torch.float64)
        J[:,[9,23,71,88]]=torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64))
        p=greedy_minor(J)
        self.assertEqual(len(set(p['columns'])),4)
        self.assertEqual(set(p['columns']),{9,23,71,88})
        M=[[Q(str(x)) for x in row] for row in J[:,p['columns']].tolist()]
        self.assertNotEqual(exact_det(M),0)
        self.assertEqual(p['required_verified_abs_entry_error_lt'],Q(1,4))

    def test_rank_deficient_rejected(self):
        J=torch.ones((4,50),dtype=torch.float64)
        with self.assertRaises(ValueError):greedy_minor(J)

    def test_perturbation_stability_budget(self):
        J=torch.zeros((4,6),dtype=torch.float64)
        J[:,:4]=torch.eye(4,dtype=torch.float64)
        p=greedy_minor(J)
        M=J[:,p['columns']]
        r=4
        for sign in (-1,1):
            noisy=M+torch.full_like(M,sign*.24)
            self.assertGreater(torch.linalg.svdvals(noisy)[-1],0)
        self.assertLess(r*.24,p['observed_sigma_min'])

    def test_validation(self):
        for J in (torch.zeros(3),torch.zeros(4,3),torch.ones((2,4),dtype=torch.int64)):
            with self.assertRaises(ValueError):greedy_minor(J)
        with self.assertRaises(ValueError):greedy_minor(torch.tensor([[float('nan'),1.]]))
        with self.assertRaises(ValueError):exact_det([[1,2],[3]])

if __name__=='__main__':unittest.main()
