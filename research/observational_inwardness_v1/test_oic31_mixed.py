"""Exact pure-integer checks for OIC-31 mixed-substitution-and-erasure result."""
import unittest
from itertools import product
from oic31_mixed import (omission_lower_bound,prime_block_schedule,check_prime_attainment,
 lower_bound_witness,hamming_tolerant_collision,count_rotating_reports,reference)
from oic30_prime_blocks import encoded

class MixedErrorTests(unittest.TestCase):
    def test_attainment_primes(self):
        for p,e,r in ((5,0,1),(5,0,2),(5,1,0),(5,1,1),
                      (7,1,0),(7,1,1),(7,2,0),(11,1,1)):
            a=check_prime_attainment(p,e,r)
            self.assertEqual(a['query_horizon'],p*(2*e+r+1))
            self.assertEqual(a['minimum_distance'],2*e+r+1)

    def test_proper_marker_pigeonhole_bound(self):
        for n,e,r in ((3,0,1),(4,1,0),(5,1,1)):
            Q=omission_lower_bound(n,e,r)-1
            # Several deterministic patterns with a deliberately missing marker n-1.
            for word in (tuple(t%(n-1) for t in range(Q)),(0,)*Q,
                         tuple((3*t+2)%(n-1) for t in range(Q))):
                v=lower_bound_witness(word,n,e,r)
                self.assertIsNotNone(v)
                self.assertLessEqual(v['distance'],2*e+r)
                self.assertEqual(sum(v['ones_per_rotating_phase']),Q)
                self.assertIsNotNone(hamming_tolerant_collision(0,encoded(word,n,1,v['rotating_phase']),Q,e,r))

    def test_exhaustive_all_short_proper_words(self):
        # Exhaust 4^Q words for n=5 in small Q ranges and check every collision.
        for n,e,r,Q in ((3,0,1,4),(4,1,0,5),(5,0,1,5)):
            for word in product(range(n-1),repeat=Q):
                v=lower_bound_witness(word,n,e,r)
                self.assertIsNotNone(v)
                self.assertLessEqual(v['distance'],2*e+r)

    def test_explicit_error_erasure_collision(self):
        a=int('101001001',2);b=a^((1<<1)|(1<<4)|(1<<8))
        obs=hamming_tolerant_collision(a,b,9,1,1)
        self.assertIsNotNone(obs)
        self.assertEqual(sum(x is None for x in obs),1)
        for code in (a,b):
            self.assertLessEqual(sum((x is not None and x!=((code>>i)&1)) for i,x in enumerate(obs)),1)
        self.assertIsNone(hamming_tolerant_collision(0b10001,0b01010,5,0,1))

    def test_bound_requires_label_omission(self):
        self.assertIsNone(lower_bound_witness(tuple(range(5)),5,1,0))
        with self.assertRaises(ValueError):prime_block_schedule(5,0,0)
        with self.assertRaises(ValueError):prime_block_schedule(5,2,0)
        with self.assertRaises(ValueError):omission_lower_bound(5,-1,0)

    def test_receipt(self):
        r=reference()
        self.assertEqual(len(r['attainable_prime_cases']),8)
        self.assertEqual(r['schema'],'OIC31-MIXED-FAULT-PROPER-MARKER-V1')

if __name__=='__main__':unittest.main()
