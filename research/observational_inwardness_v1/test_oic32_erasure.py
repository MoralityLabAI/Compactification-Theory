"""OIC-32 global one-erasure prime-size theorem tests (no assert statements)."""
import random
import unittest
from oic32_erasure import analytic_erasure_gap,exhaustive_small,reference,two_marker_code
from oic32_global import minimum_cross_shift_distance,pair_moment_identity


class OIC32ErasureTests(unittest.TestCase):
    def test_uniform_prime_analytic_inequality(self):
        for p in (5,7,11,13,17,19,23,29):
            row=analytic_erasure_gap(p)
            self.assertEqual(row['optimal_queries'],2*p)
            self.assertEqual(row['line_pair_lower_for_all_words']-row['line_pair_upper_if_erasure_robust'],row['contradiction_gap'])
            self.assertGreater(row['contradiction_gap'],0)
        self.assertEqual(analytic_erasure_gap(7)['contradiction_gap'],9)

    def test_prime_two_block_corrects_one_erasure(self):
        for p in (2,3,5,7,11,13):
            w=two_marker_code(p)
            self.assertEqual(set(w),{0,1})
            self.assertEqual(len(w),2*p)
            self.assertEqual(minimum_cross_shift_distance(p,w),2)
            if p>=5:self.assertEqual(pair_moment_identity(p,w)['identity_rhs'],pair_moment_identity(p,w)['line_pair_moment'])

    def test_exhaustive_two_and_three_phase_minima(self):
        expected={(2,3):(4,1),(2,4):(8,2),(3,5):(81,1),(3,6):(243,2)}
        for (p,q),(count,maxdist) in expected.items():
            row=exhaustive_small(p,q)
            self.assertEqual((row['checked_normalized_schedules'],row['max_minimum_cross_shift_distance']), (count,maxdist))
            self.assertEqual(minimum_cross_shift_distance(p,tuple(row['example'])),maxdist)

    def test_random_below_bound_is_ambiguous(self):
        rng=random.Random(912203)
        for p in (5,7,11):
            for _ in range(18):
                word=tuple(rng.randrange(p) for _ in range(2*p-1))
                self.assertLessEqual(minimum_cross_shift_distance(p,word),1)

    def test_replay_and_validation(self):
        result=reference()
        self.assertEqual(len(result['small_primes']),4)
        self.assertEqual(len(result['analytic_primes']),4)
        with self.assertRaises(ValueError):analytic_erasure_gap(3)
        with self.assertRaises(ValueError):analytic_erasure_gap(9)
        with self.assertRaises(ValueError):two_marker_code(1)
        with self.assertRaises(ValueError):exhaustive_small(5,5)

if __name__=='__main__':unittest.main()
