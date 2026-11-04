import random
import unittest
from fractions import Fraction as Q
from research.ingression_dynamics_v1.goal_aware_partition import partitions
from research.ingression_dynamics_v1.hit_rate_dp import (
    optimal_rate_groups, optimal_with_state_bits, uniform_hit_closed_form,
    min_bits_for_uniform_error,
)


def independent_partition_oracle(rates, k):
    """Independent exhaustive Bell-number oracle; no DP and no sorting."""
    n = len(rates)
    best = None
    for assignment in partitions(tuple(range(n))):
        if len(set(assignment.values())) > k:
            continue
        groups = {}
        for i, value in enumerate(rates):
            groups.setdefault(assignment[i], []).append(Q(value))
        cost = max(((max(v)-min(v))/2 for v in groups.values()),default=Q(0))
        if best is None or cost < best:
            best = cost
    return best


class RateDPTests(unittest.TestCase):
    def test_qo01_five_state_frontier(self):
        rates=[Q(i,5) for i in range(1,5)]
        for b,expected in ((1,Q(3,10)),(2,Q(1,10)),(3,Q(0))):
            got,assignment,blocks=optimal_with_state_bits(rates,b)
            self.assertEqual(got,expected)
            self.assertEqual(got,uniform_hit_closed_form(4,b))
            self.assertLessEqual(len(blocks)+1,2**b)
            self.assertEqual(len(assignment),4)

    def test_random_small_against_independent_bell_oracle(self):
        rng=random.Random(20261104)
        cases=0
        for n in range(1,7):
            for _ in range(12):
                rates=[Q(rng.randrange(0,11),10) for _ in range(n)]
                for k in range(1,n+1):
                    expected=independent_partition_oracle(rates,k)
                    obtained, assignment, groups=optimal_rate_groups(rates,k)
                    self.assertEqual(obtained,expected)
                    self.assertEqual(len(set(assignment)),k)
                    self.assertEqual(len(groups),k)
                    cases+=1
        self.assertEqual(cases,252)

    def test_uniform_closed_form_many_sizes(self):
        for n in range(1,21):
            rates=[Q(i,n+1) for i in range(1,n+1)]
            for bits in range(1,6):
                computed,_,_=optimal_with_state_bits(rates,bits)
                self.assertEqual(computed,uniform_hit_closed_form(n,bits))

    def test_inverse_minimum_bits_bound(self):
        for n in range(1,25):
            for eps in (Q(0),Q(1,100),Q(1,20),Q(1,10),Q(1,4),Q(1,2)):
                b=min_bits_for_uniform_error(n,eps)
                self.assertLessEqual(uniform_hit_closed_form(n,b),eps)
                if b>1:
                    self.assertGreater(uniform_hit_closed_form(n,b-1),eps)

    def test_invalid_inputs(self):
        for rates,k in (([],1),([Q(1,2)],0),([Q(-1,2)],1),([Q(3,2)],1)):
            with self.assertRaises(ValueError):
                optimal_rate_groups(rates,k)
        with self.assertRaises(ValueError):
            optimal_with_state_bits([Q(1,2)],0)

if __name__=='__main__':
    unittest.main()
