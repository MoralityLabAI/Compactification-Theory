import random
import unittest
from fractions import Fraction as Q
from itertools import product
from math import exp

from research.ingression_dynamics_v1.rate_sampling import (
    max_estimation_error, partition_radius, empirical_midpoints,
    frozen_macro_error, empirical_certificate,
    hoeffding_union_radius, hoeffding_union_failure_bound,
    per_pair_samples_for_regret, exact_binomial_violation,
)


def direct_assignment_optimum(points, budget):
    """Independent brute-force class assignments, with no candidate-box solver."""
    n = len(points)
    return min(
        max(((max(points[i][a] for i in range(n) if labels[i]==z) -
              min(points[i][a] for i in range(n) if labels[i]==z)) / 2
             for z in set(labels) for a in range(len(points[0]))),default=Q(0))
        for labels in product(range(budget),repeat=n)
    )


class RateSamplingTests(unittest.TestCase):
    def test_exact_one_step_certificate_is_sharp(self):
        real = ((Q(0),), (Q(1),))
        measured = ((Q(1,10),), (Q(9,10),))
        result = empirical_certificate(real,measured,1)
        self.assertEqual(result["uniform_rate_error"],Q(1,10))
        self.assertEqual(result["empirical_optimum"],Q(2,5))
        self.assertEqual(result["true_optimum"],Q(1,2))
        self.assertEqual(result["true_frozen_macro_defect"],Q(1,2))
        self.assertEqual(result["frozen_macro_upper_bound"],Q(1,2))
        self.assertEqual(result["true_partition_regret"],Q(0))

    def test_bad_measurement_can_change_best_partition(self):
        real = ((Q(0),),(Q(2,5),),(Q(9,10),))
        measured = ((Q(0),),(Q(3,5),),(Q(9,10),))
        audit = empirical_certificate(real, measured, 2)
        self.assertEqual(audit["true_optimum"],Q(1,5))
        self.assertEqual(audit["empirical_optimum"],Q(3,20))
        self.assertEqual(audit["true_fixed_partition_defect"],Q(1,4))
        self.assertEqual(audit["true_partition_regret"],Q(1,20))
        self.assertLessEqual(audit["true_partition_regret"],
                             audit["optimization_regret_upper_bound"])

    def test_seeded_two_action_estimators_vs_independent_optima(self):
        rng=random.Random(20261104)
        cases=0
        for n in (2,3,4):
            for _ in range(30):
                real=tuple(tuple(Q(rng.randrange(11),10) for _ in range(2))
                           for _ in range(n))
                estimate=tuple(tuple(max(Q(0),min(Q(1),v+Q(rng.randrange(-2,3),10)))
                                     for v in p) for p in real)
                eta=max_estimation_error(real,estimate)
                for k in range(1,n+1):
                    result=empirical_certificate(real, estimate, k)
                    self.assertEqual(result["true_optimum"],
                                     direct_assignment_optimum(real,k))
                    self.assertEqual(result["empirical_optimum"],
                                     direct_assignment_optimum(estimate,k))
                    self.assertLessEqual(result["true_frozen_macro_defect"],
                                         result["empirical_optimum"]+eta)
                    self.assertLessEqual(result["true_partition_regret"],2*eta)
                    self.assertLessEqual(
                        abs(result["true_optimum"]-result["empirical_optimum"]),eta)
                    cases += 1
        self.assertEqual(cases,270)

    def test_midpoint_kernel_matches_empirical_radius(self):
        pts=((Q(0),Q(1)),(Q(1,4),Q(1,2)),(Q(1),Q(0)))
        labels=(0,0,1)
        midpoint=empirical_midpoints(pts,labels)
        self.assertEqual(frozen_macro_error(pts,labels,midpoint),
                         partition_radius(pts,labels))
        self.assertEqual(midpoint[0],(Q(1,8),Q(3,4)))

    def test_exact_bernoulli_tails_below_hoeffding(self):
        cases=0
        for N in range(1,14):
            for p in (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)):
                for eta in (Q(1,5),Q(1,3),Q(1,2)):
                    actual=float(exact_binomial_violation(p,N,eta))
                    upper=2*exp(-2*N*float(eta)**2)
                    self.assertLessEqual(actual,upper+1e-12)
                    cases+=1
        self.assertEqual(cases,195)

    def test_union_bound_and_budget(self):
        n,d,alpha,target=41,12,0.05,0.1
        N=per_pair_samples_for_regret(n,d,target,alpha)
        self.assertGreater(N,0)
        self.assertLessEqual(hoeffding_union_radius(n,d,N,alpha),
                             target/2 + 1e-12)
        self.assertLessEqual(hoeffding_union_failure_bound(
            n,d,N,target/2),alpha+1e-12)
        self.assertGreaterEqual(N*n*d,n*d)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            empirical_certificate(((Q(1),),),((Q(2),),),1)
        with self.assertRaises(ValueError):
            empirical_certificate(((Q(1),),),((Q(1),),),0)
        with self.assertRaises(ValueError):
            hoeffding_union_radius(2,0,10,.05)
        with self.assertRaises(ValueError):
            per_pair_samples_for_regret(1,2,1.5,.05)
        with self.assertRaises(ValueError):
            exact_binomial_violation(Q(2),3,Q(1,2))


if __name__ == "__main__":
    unittest.main()
