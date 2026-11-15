"""SMC-03b exact split controls. CPU training is a separate optional command."""
import unittest
from fractions import Fraction
import family_holdout as fh


class FamilyHoldoutTests(unittest.TestCase):
    def test_disjoint_entire_matching_classes(self):
        data=fh.family_split()
        self.assertEqual(len(data['train_g']),12)
        self.assertEqual(len(data['test_g']),3)
        self.assertEqual(len(data['train_pi']),576)
        self.assertEqual(len(data['test_pi']),144)
        self.assertFalse(set(data['train_g']) & set(data['test_g']))
        self.assertFalse(set(data['train_pi']) & set(data['test_pi']))

    def test_permutation_fiber_sizes_and_marginal_match(self):
        data=fh.family_split()
        spl=data['split']
        for suffix in ('train','test'):
            classes=data[suffix+'_g']
            perms=data[suffix+'_pi']
            self.assertEqual(len(perms),48*len(classes))
            for g in classes:
                positive_count=sum(spl['orbit_index'][i]==g for i in perms)
                self.assertEqual(positive_count,48)
                positive_probability=Fraction(positive_count,len(perms))
                negative_probability=Fraction(1,len(classes))
                self.assertEqual(positive_probability,negative_probability)

    def test_exact_family_bayes_limits(self):
        d=fh.family_split()
        self.assertEqual(1-Fraction(1,2*len(d['train_g'])),Fraction(23,24))
        self.assertEqual(1-Fraction(1,2*len(d['test_g'])),Fraction(5,6))

    def test_bounded_split_argument(self):
        for k,hold in ((2,1),(3,0),(3,15)):
            with self.assertRaises(ValueError):
                fh.family_split(k,hold)


if __name__=='__main__':
    unittest.main()
