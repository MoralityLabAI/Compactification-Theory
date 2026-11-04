"""Exact rational Neumann/small-matrix certificates; no numerical backend."""
import unittest
from fractions import Fraction as Q
from robust_inverse import inverse4,inverse_perturbation_witness


class TestRationalInverseCertificate(unittest.TestCase):
    def test_inverse_exact_nondiagonal(self):
        a=[[Q(3),Q(1),0,0],[Q(2),Q(1),0,0],[0,0,4,0],[0,0,0,5]]
        inv=inverse4(a)
        for i in range(4):
            for j in range(4):
                self.assertEqual(sum(a[i][k]*inv[k][j] for k in range(4)),int(i==j))

    def test_invertible_identity_uniform_interval(self):
        b=[[('[1 +/- 0.01]' if i==j else '[+/- 0.01]') for j in range(4)] for i in range(4)]
        ref=[[(float(int(i==j))).hex() for j in range(4)] for i in range(4)]
        got=inverse_perturbation_witness(b,ref)
        self.assertTrue(got['rank_four_by_neumann'])
        self.assertEqual(Q(got['strict_neumann_norm_upper_rational']),Q(1,25))

    def test_too_wide_interval_returns_no_certificate(self):
        b=[[('[1 +/- 1]' if i==j else '0') for j in range(4)] for i in range(4)]
        ref=[[(float(int(i==j))).hex() for j in range(4)] for i in range(4)]
        self.assertFalse(inverse_perturbation_witness(b,ref)['rank_four_by_neumann'])

    def test_singular_reference_fails(self):
        with self.assertRaises(ValueError):inverse4([[0]*4 for _ in range(4)])

    def test_floating_reference_exact_hex(self):
        ref=[[(float(int(i==j))).hex() for j in range(4)] for i in range(4)]
        b=[['1' if i==j else '0' for j in range(4)] for i in range(4)]
        self.assertEqual(Q(inverse_perturbation_witness(b,ref)['strict_neumann_norm_upper_rational']),0)

if __name__=='__main__':unittest.main()
