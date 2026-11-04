import unittest
from fractions import Fraction as Q
from itertools import product
from math import comb

from research.ingression_dynamics_v1.identification_sample_floor import (
    binomial_sequence_tv, binomial_bayes_risk, first_budget_for_bayes_error,
    adaptive_policy_tv, symmetric_kl, pinsker_tv_upper, pinsker_budget_lower,
)


def direct_sequence_tv(draws, p, q):
    """Independent 2^B trace enumerator, with no binomial-count grouping."""
    value=Q(0)
    for word in product((0,1),repeat=draws):
        pa=Q(1)
        pb=Q(1)
        for bit in word:
            pa *= p if bit else 1-p
            pb *= q if bit else 1-q
        value+=abs(pa-pb)
    return value/2


class IdentificationFloorTests(unittest.TestCase):
    def test_exact_bitstring_oracle_matches_binomial_formula(self):
        count=0
        for p,q in ((Q(1,2),Q(5,8)),(Q(1,4),Q(3,4)),
                    (Q(2,5),Q(3,5))):
            for B in range(11):
                self.assertEqual(binomial_sequence_tv(B,p,q),
                                 direct_sequence_tv(B,p,q))
                self.assertEqual(binomial_bayes_risk(B,p,q),
                                 (1-direct_sequence_tv(B,p,q))/2)
                count+=1
        self.assertEqual(count,33)

    def test_exact_one_quarter_bayes_budget(self):
        p,q=Q(1,2),Q(5,8)
        self.assertEqual(first_budget_for_bayes_error(p,q,Q(1,4),64),29)
        self.assertGreater(binomial_bayes_risk(28,p,q),Q(1,4))
        self.assertLessEqual(binomial_bayes_risk(29,p,q),Q(1,4))
        self.assertIsNone(first_budget_for_bayes_error(p,q,Q(1,4),28))

    def test_all_128_adaptive_policies_cannot_beat_all_informative(self):
        B=3
        best=binomial_sequence_tv(B,Q(1,2),Q(5,8))
        count=0
        for mask in range(1<<((1<<B)-1)):
            observed=adaptive_policy_tv(B,Q(1,2),Q(5,8),Q(1,4),mask)
            self.assertLessEqual(observed,best)
            count+=1
        self.assertEqual(count,128)
        self.assertEqual(adaptive_policy_tv(B,Q(1,2),Q(5,8),
                                           Q(1,4),127),best)
        self.assertEqual(adaptive_policy_tv(B,Q(1,2),Q(5,8),
                                           Q(1,4),0),Q(0))

    def test_exact_information_monotonicity(self):
        p,q=Q(1,2),Q(5,8)
        previous=Q(0)
        for B in range(65):
            tv=binomial_sequence_tv(B,p,q)
            self.assertGreaterEqual(tv,previous)
            self.assertLessEqual(tv,Q(1))
            previous=tv

    def test_pinsker_numerical_outer_bound(self):
        for delta in (Q(1,8),Q(1,10),Q(1,20)):
            for B in (0,1,4,8,16,32,64,128):
                exact=float(binomial_sequence_tv(B,Q(1,2),
                                                 Q(1,2)+delta))
                upper=pinsker_tv_upper(B,delta)
                self.assertLessEqual(exact,upper+1e-12)
        self.assertLess(pinsker_budget_lower(Q(1,8),Q(1,4)),29)

    def test_small_delta_requires_more_samples(self):
        p=Q(1,2)
        self.assertGreater(binomial_bayes_risk(128,p,p+Q(1,20)),Q(1,4))
        self.assertEqual(binomial_bayes_risk(0,p,p+Q(1,8)),Q(1,2))

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            binomial_sequence_tv(-1,Q(1,2),Q(3,4))
        with self.assertRaises(ValueError):
            binomial_sequence_tv(2,Q(0),Q(3,4))
        with self.assertRaises(ValueError):
            adaptive_policy_tv(3,Q(1,2),Q(5,8),Q(1,4),128)
        with self.assertRaises(ValueError):
            symmetric_kl(Q(0))
        with self.assertRaises(ValueError):
            pinsker_budget_lower(Q(1,8),Q(3,4))

if __name__=="__main__":
    unittest.main()
