"""OIC-30 prime-block construction, analytic witness, and receipt tests."""
import json
import random
import unittest
from itertools import product
from pathlib import Path

from oic30_prime_blocks import (
    prime_blocks, cross_shift_spectrum, check_all_flip_paths,
    sensor_counts_per_residue, reconstruct_counts, two_marker_obstruction,
    count_two_marker_obstructions, marker_restricted_lower_bound,
    verify_main_witness, encoded,
)

HERE=Path(__file__).resolve().parent


class PrimeBlockTests(unittest.TestCase):
    def test_prime_error_correction_four_sizes(self):
        for p in (5,7,11,13):
            w=prime_blocks(p,(0,1,2))
            hist=cross_shift_spectrum(w,p)
            self.assertEqual(min(hist),3)
            self.assertEqual(sum(hist.values()),p*p*p*(p-1)//2)
            self.assertEqual(len(w),3*p)

    def test_prime_p3_all_markers(self):
        w=prime_blocks(3,(0,1,2))
        self.assertEqual(min(cross_shift_spectrum(w,3)),4)

    def test_general_k_for_primes(self):
        for p in (5,7,11):
            for k in range(2,p):
                word=prime_blocks(p,tuple(range(k)))
                self.assertEqual(min(cross_shift_spectrum(word,p)),k)

    def test_single_erasure_two_blocks(self):
        for p in (3,5,7,11):
            self.assertEqual(min(cross_shift_spectrum(prime_blocks(p,(0,1)),p)),2)

    def test_n7_exact_fault_receipts(self):
        w=prime_blocks(7,(0,1,2))
        checks=check_all_flip_paths(w,7)
        self.assertEqual(checks['true_world_fault_paths'],49*22)
        self.assertEqual(checks['distinct_received_words'],1012)

    def test_count_reconstruction_keeps_all_code_distances(self):
        rng=random.Random(17030)
        for _ in range(20):
            counts=tuple(rng.randrange(4) for _ in range(7))
            original=reconstruct_counts(7,counts)
            # Rearrange the three readings inside each phase-residue class.
            reword=list(original)
            for r in range(7):
                indices=[r,r+7,r+14]
                values=[reword[i] for i in indices]
                rng.shuffle(values)
                for i,v in zip(indices,values):reword[i]=v
            self.assertEqual(sensor_counts_per_residue(tuple(reword),7,(0,1)),
                             sensor_counts_per_residue(original,7,(0,1)))
            self.assertEqual(cross_shift_spectrum(tuple(reword),7),
                             cross_shift_spectrum(original,7))

    def test_two_marker_analytic_witness_all_residue_counts(self):
        for p in (3,5,7):
            result=count_two_marker_obstructions(p)
            self.assertTrue(result['all_rejected_for_one_substitution'])
            self.assertEqual(result['count_vectors_checked'],4**p)
            for key in result['analytic_witness_distance_histogram']:
                self.assertLessEqual(int(key),2)

    def test_two_marker_nonconstant_and_constant_witnesses(self):
        for counts in ((0,1,2,3,0,1,2),(1,)*7,(2,)*7,(3,)*7,(0,)*7):
            o=two_marker_obstruction(7,counts)
            self.assertLessEqual(o['hamming_distance'],2)
            self.assertNotEqual(o['world_a'][0],o['world_b'][0])
            word=reconstruct_counts(7,counts)
            a=encoded(word,7,*o['world_a'])
            b=encoded(word,7,*o['world_b'])
            self.assertEqual((a^b).bit_count(),o['hamming_distance'])

    def test_proper_label_subset_count_lower_bound(self):
        rng=random.Random(30983)
        for n in (4,5,7,9):
            for Q in (1,n,2*n,3*n-1):
                for _ in range(25):
                    word=tuple(rng.randrange(n-1) for _ in range(Q))
                    # Fixed, stationary initial phase n-1 is silent.
                    silent=encoded(word,n,0,n-1)
                    self.assertEqual(silent,0)
                    weights=[encoded(word,n,1,z).bit_count() for z in range(n)]
                    self.assertEqual(sum(weights),Q)
                    self.assertLess(min(weights),3)
            self.assertEqual(marker_restricted_lower_bound(n,1),3*n)

    def test_parameter_validation_and_optimized_mode(self):
        for p,markers in ((4,(0,1,2)),(5,(0,0)),(5,(0,5)),(5,(0,)),(3,(0,1,2,3))):
            with self.assertRaises(ValueError):prime_blocks(p,markers)
        with self.assertRaises(ValueError):count_two_marker_obstructions(11)
        with self.assertRaises(ValueError):reconstruct_counts(7,(1,2))

    def test_pinned_receipt_exact_match(self):
        saved=json.loads((HERE/'results/oic30_reference.json').read_text(encoding='utf8'))
        self.assertEqual(saved,verify_main_witness())
        self.assertEqual(saved['distance_spectrum'],{'3':168,'4':315,'6':420,'8':126})
        self.assertEqual(saved['two_marker_q21_exhaustive']['count_vectors_checked'],16384)
        self.assertEqual(saved['first_episode_bits_with_literal_offline_preprogramming'],72)


if __name__=='__main__':
    unittest.main()
