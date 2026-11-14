"""Exact tests for the KSR-01C dependence extension."""
import random
import unittest
from fractions import Fraction as F
from dependence import (coarse_dependence, finite_map_floor, exact_controls,
    histogram_convolve, iid_histogram_power, shared_innovation_joint,
    shared_innovation_moment, symmetric_projection, total_variation)


class DependenceTests(unittest.TestCase):
    def test_hidden_mixture_creates_exact_sibling_dependence(self):
        r=coarse_dependence([0,0],[(1,0),(0,1)],2)
        self.assertEqual(r['weighted_sibling_diagonal_excess'],'1/2')
        self.assertFalse(r['coarse_sibling_independence_exact'])
        self.assertEqual(r['by_parent_label'][0]['max_same_parent_label_child_TV'],'1')
        self.assertEqual(r['by_parent_label'][0]['sibling_joint'],[['1/2','0'],['0','1/2']])

    def test_constant_kernel_is_sibling_independent(self):
        r=coarse_dependence([0,0,1],[(F(3,4),F(1,4))]*2+[(F(1,4),F(3,4))],2)
        self.assertEqual(r['weighted_sibling_diagonal_excess'],'0')
        self.assertTrue(r['coarse_sibling_independence_exact'])

    def test_symmetric_fit_and_missing_rows(self):
        self.assertFalse(symmetric_projection([None,[0,1]])['available'])
        r=symmetric_projection([[F(13,15),F(1,15),F(1,15)],
                                [F(1,15),F(13,15),F(1,15)],
                                [F(1,15),F(1,15),F(13,15)]])
        self.assertEqual(r['lambda'],'4/5')
        self.assertEqual(r['max_row_TV'],'0')

    def test_histogram_convolution(self):
        law={(1,0):F(1,2),(0,1):F(1,2)}
        self.assertEqual(iid_histogram_power(law,2),
                         {(2,0):F(1,4),(1,1):F(1,2),(0,2):F(1,4)})

    def test_nonuniform_prior_and_constant_observation(self):
        law={(0,(1,1)):F(3,4),(1,(1,1)):F(1,4)}
        result=finite_map_floor(law,2)
        self.assertEqual(result['covariance_floor'],'3/4')
        self.assertEqual(result['count_MAP'],'3/4')
        law={(0,(2,0)):F(3,4),(1,(0,2)):F(1,4)}
        result=finite_map_floor(law,2)
        self.assertEqual(result['covariance_floor'],'1')

    def test_random_finite_population_cauchy_floor(self):
        rng=random.Random(940229240)
        for q in (2,3,4):
            for _ in range(15):
                law={}
                for c in range(q):
                    for i in range(q):
                        count=tuple(2*int(j==i) for j in range(q))
                        law[c,count]=F(rng.randrange(1,20))
                z=sum(law.values()); law={k:v/z for k,v in law.items()}
                stats=finite_map_floor(law,q)
                self.assertLessEqual(F(stats['covariance_floor']),F(stats['count_MAP']))

    def test_exact_shared_noise_counterexample(self):
        x=exact_controls()['correlated_children']
        self.assertEqual(x['naive_IID_floor'],'5/7')
        self.assertEqual(x['true_count_MAP'],'2/3')
        self.assertEqual(x['dependence_corrected_floor'],'1/2')
        self.assertGreater(F(x['naive_IID_floor']),F(x['true_count_MAP']))

    def test_shared_noise_moments_from_separate_enumeration(self):
        for q in (2,3):
            for b in (2,3):
                for depth in (1,2):
                    for rho in (F(0),F(1,3),F(1)):
                        lam=F(4,5)
                        law=shared_innovation_joint(q,b,lam,depth,rho)
                        moment=sum(p*((q*counts[0]-b**depth)/(b*lam)**depth)**2
                                   for (root,counts),p in law.items())/(q-1)
                        self.assertEqual(moment,shared_innovation_moment(b,lam,depth,rho))
                        stats=finite_map_floor(law,q)
                        self.assertEqual(F(stats['covariance_floor']),F(1,q)+F(q-1,q)/moment)

    def test_perfect_channel_and_monotone_noise_penalty(self):
        for depth in range(8):
            self.assertEqual(shared_innovation_moment(2,1,depth,1),1)
            values=[shared_innovation_moment(2,F(4,5),depth,r)
                    for r in (F(0),F(1,2),F(1))]
            self.assertEqual(values,sorted(values))

    def test_invalid_laws_rejected(self):
        with self.assertRaises(ValueError): total_variation([1,1],[1,0])
        with self.assertRaises(ValueError): coarse_dependence([0],[(1,0,0)],2)
        with self.assertRaises(ValueError): finite_map_floor({(0,(1,0)):F(1,2)},2)
        with self.assertRaises(ValueError): shared_innovation_joint(3,4,F(1,2),3,1)


if __name__=='__main__': unittest.main()
