"""SMC-03 standard-library tests; torch is optional and tested on CPU separately."""
import unittest
from fractions import Fraction
from itertools import permutations
import relational_probe as rp
from transport_lift import relabel


class RelationalProbeTests(unittest.TestCase):
    def test_orbit_stratified_permutation_holdout(self):
        for k,per_fiber in ((2,8),(3,48)):
            split=rp.orbit_split(k)
            m=len(split['states'])
            self.assertEqual(len(split['perms']),m*per_fiber)
            self.assertEqual(len(set(split['train']) & set(split['test'])),0)
            for g in range(m):
                train=sum(split['orbit_index'][p]==g for p in split['train'])
                test=sum(split['orbit_index'][p]==g for p in split['test'])
                self.assertEqual(train+test,per_fiber)
                self.assertEqual(test,per_fiber//4)

    def test_exact_marginal_only_and_additive_logloss_null(self):
        for k,rawdim in ((2,22),(3,51)):
            report=rp.exact_population_baseline(k)
            self.assertEqual(report['raw_feature_dim'],rawdim)
            self.assertEqual(report['additive_population_gradient_at_zero'],'0')
            self.assertEqual(report['additive_logloss_optimal_balanced_accuracy'],'1/2')

    def test_structured_event_is_exact_and_gauge_equivariant(self):
        split=rp.orbit_split(3)
        gs=split['states']
        perms=split['perms']
        _,_,overlap=rp.pair_features(split)
        k=3;m=len(gs)
        self.assertEqual(sum(overlap[i][split['orbit_index'][i]]==k for i in split['test']),len(split['test']))
        self.assertEqual(sum(overlap[i][j]==k for i in split['test'] for j in range(m)),len(split['test']))
        sigma=(2,1,0,3,4,5)
        for i in split['test']:
            p=perms[i];g=gs[split['orbit_index'][i]]
            self.assertEqual(relabel(sigma,g),
                             relabel(tuple(sigma[p[j]] for j in range(6)),split['base']))

    def test_disjoint_permutation_cohort_preserves_both_class_marginals(self):
        split=rp.orbit_split(3)
        m=len(split['states'])
        for part in ('train','test'):
            ids=split[part]
            positive=[split['orbit_index'][i] for i in ids]
            for g in range(m):
                self.assertEqual(positive.count(g)*m,len(ids))
            self.assertEqual(Fraction(1,2)*(1+Fraction(m-1,m)),
                             Fraction(29,30))


if __name__=='__main__':
    unittest.main()
