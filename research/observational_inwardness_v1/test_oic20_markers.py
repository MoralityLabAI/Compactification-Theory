"""OIC-20 exact finite sensor-design verification, standard library only."""
from __future__ import annotations
import unittest
from itertools import product
from math import gcd

from oic20_markers import (
    trace_distribution, shift_signature, two_marker_equivalent,
    partitions, identifies_every_shift, sufficient_divisor_markers,
    prime_power_minimum_markers, exhaustive_minimum_markers, reference,
)


class MarkerTheorems(unittest.TestCase):
    def test_single_marker_is_exactly_gcd_class(self):
        for n in range(2, 22):
            for a in range(n):
                for b in range(n):
                    eq = trace_distribution(n, a, (0,)) == trace_distribution(n, b, (0,))
                    self.assertEqual(eq, gcd(n, a) == gcd(n, b), (n, a, b))

    def test_two_marker_lag_characterization(self):
        for n in range(3, 18):
            for k in range(1, n):
                codes = {d:trace_distribution(n, d, (0,k)) for d in range(n)}
                for a in range(n):
                    for b in range(n):
                        self.assertEqual(codes[a] == codes[b], two_marker_equivalent(n,a,b,k),
                                         (n,k,a,b,shift_signature(n,a,k),shift_signature(n,b,k)))

    def test_full_cycle_observations_determine_every_future(self):
        # Every hidden orbit has period dividing n. Hence any length-n block
        # determines its infinite periodic extension for the same shift model.
        for n in range(2,11):
            for d in range(n):
                for marks in ((0,), (0,1)):
                    a=trace_distribution(n,d,marks,n)
                    b=trace_distribution(n,d,marks,2*n)
                    self.assertEqual(len(a),len(b))
                    self.assertEqual(
                        {word+word:count for word,count in a.items()}, b,
                        (n,d,marks))

    def test_coprime_second_marker_identifies_all_unit_shifts(self):
        for n in range(3,35):
            for k in range(1,n):
                if gcd(n,k)!=1:continue
                sigs={shift_signature(n,d,k) for d in range(n) if gcd(n,d)==1}
                self.assertEqual(len(sigs),sum(gcd(n,d)==1 for d in range(n)))

    def test_composite_two_marker_witness(self):
        self.assertTrue(identifies_every_shift(6,(0,2)))
        self.assertTrue(identifies_every_shift(10,(0,2)))
        self.assertFalse(identifies_every_shift(8,(0,1)))
        self.assertFalse(identifies_every_shift(12,(0,4)))
        self.assertTrue(identifies_every_shift(12,(0,1,4)))

    def test_divisor_sensor_construction(self):
        for n in range(2,34):
            marks=sufficient_divisor_markers(n)
            self.assertEqual(marks[0],0)
            self.assertEqual(len(marks),sum(n % d==0 for d in range(1,n+1)))
            self.assertTrue(identifies_every_shift(n,marks))

    def test_prime_power_optimal_counts(self):
        for p,a in ((2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(5,1),(5,2),(7,1)):
            n=p**a
            expected=a if p==2 else a+1
            m,marks=prime_power_minimum_markers(p,a)
            self.assertEqual(m,expected)
            self.assertTrue(identifies_every_shift(n,marks))
            if n<=16:
                self.assertEqual(exhaustive_minimum_markers(n)[0],m)

    def test_brute_minimal_sensor_sets(self):
        expected={2:1,3:2,4:2,5:2,6:2,7:2,8:3,9:3,10:2,11:2,
                  12:3,13:2,14:2,15:3,16:4}
        actual={row['n']:row['minimal_label_count'] for row in reference()['cases']}
        self.assertEqual(actual,expected)

    def test_phase_translation_is_gauge(self):
        for n in range(3,11):
            for d in range(n):
                for k in range(1,n):
                    a=trace_distribution(n,d,(0,k))
                    shift=2%n
                    b=trace_distribution(n,d,(shift,(shift+k)%n))
                    self.assertEqual(a,b)

    def test_invalid_parameters(self):
        bad=((1,0,(0,)),(3,'bad',(0,)),(4,0,(0,0)),(4,0,()),
             (5,0,(5,)),(4,0,(-1,)))
        for args in bad:
            with self.assertRaises(ValueError):trace_distribution(*args)
        with self.assertRaises(ValueError):shift_signature(4,1,0)
        with self.assertRaises(ValueError):prime_power_minimum_markers(4,1)

if __name__ == '__main__':unittest.main()
