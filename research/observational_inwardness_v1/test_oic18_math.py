"""OIC-18 standalone finite math and receipt-integrity tests (no torch)."""
import copy
import math
import unittest

from oic18_verify import (
    emission, sigmoid, load_all, require, verify_record_fast,
    visible_permutation_invariant, filter_posterior, oracle,
)


class OIC18Theorems(unittest.TestCase):
    def test_von_mises_uniform_nearest_neighbor_gap(self):
        for kappa in (0.1, 1., 4., 13.):
            for n in (2,3,4,5,6,8,24,32,64,128,512):
                gap=kappa*(n/5)**2*(1-math.cos(2*math.pi/n))
                self.assertGreaterEqual(gap+1e-10,8*kappa/25)
                self.assertLessEqual(gap,2*math.pi**2*kappa/25+1e-10)

    def test_off_marker_gaussian_tail(self):
        for n in (2,4,7,16,24,64,256):
            a,k=1.25,4.
            off=sum(sigmoid(a-k*(n/5)**2*(1-math.cos(2*math.pi*z/n)))
                    for z in range(1,n))
            upper=2*math.exp(a)*sum(math.exp(-8*k*(z**2)/25) for z in range(1,100))
            self.assertLessEqual(off,upper+1e-10)

    def test_polynomial_temperature_gap_under_positive_coefficients(self):
        for n in (2,3,4,5,8,24,64,256):
            rho,c1,c2=.7,4.,.25
            temp=1+rho*((n/5)**2-1)
            gap=temp*(c1*(1-math.cos(2*math.pi/n)) +c2*(1-math.cos(4*math.pi/n)))
            self.assertGreaterEqual(gap+1e-10,8*rho*c1/25)

    def test_harmonic_fixed_coefficient_neighbor_gap_decays(self):
        for n in (8,16,32,64,128,256):
            c1,c3=5.0,3.0
            gap=c1*(1-math.cos(2*math.pi/n))+c3*(1-math.cos(6*math.pi/n))
            self.assertLessEqual(abs(gap),(2*math.pi**2*c1+18*math.pi**2*c3)/n**2+1e-10)

    def test_non_dihedral_gauge_and_reflection(self):
        for n in (4,5,6,7,8,12):
            perm=list(range(n));perm[1],perm[2]=perm[2],perm[1]
            for marker in (0,2):
                self.assertTrue(visible_permutation_invariant(n,'101'*n,marker,perm))
                self.assertTrue(visible_permutation_invariant(n,'1'*n,marker,list(reversed(range(n)))))

    def test_passive_is_observationally_identical(self):
        for n in (4,7,16,64):
            for f in (0,1):
                for z in range(n):
                    self.assertEqual(oracle(n,f,z,'0'*(2*n)),[i%2 for i in range(2*n+1)])

    def test_receipt_structure_and_mutation_detection(self):
        _,records=load_all()
        self.assertEqual(len(records),60)
        sample=copy.deepcopy(records[0])
        self.assertTrue(verify_record_fast(sample))
        sample['panels']['24']['full_active']['probabilities'][0] = 0.12345678
        with self.assertRaises(ValueError):
            verify_record_fast(sample)

    def test_fitted_von_mises_parameters_stay_positive(self):
        _,records=load_all()
        for r in records:
            if r['architecture']=='von_mises':
                p=emission(64,'von_mises',r['weights'])
                self.assertEqual(max(range(64),key=lambda i:p[i]),0)
                self.assertGreater(p[0],p[1])


if __name__=='__main__':
    unittest.main()
