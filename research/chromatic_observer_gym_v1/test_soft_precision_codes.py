"""COE-03 audit: exact rational certificates and separately labeled CPU pilot."""
import itertools
import json
import math
import unittest
from fractions import Fraction as F

try:
    from .soft_precision_codes import (
        SoftCode, seeded_scaled_code, k5_soft_exact, soft_l1,
        soft_distance_matrix, quantize_threshold, binary_diameter_two,
        multiplex_onehot, validate_transcript, transcript_distance,
        hoeffding_sufficient_width, reference_report,
        DEMO_N, DEMO_K, DEMO_SEED, DEMO_GRID_BITS, DEMO_GRID_AMPLITUDE,
    )
    from .train_soft_probe import fit_soft_probe, run_pilot
except ImportError:
    from soft_precision_codes import (
        SoftCode, seeded_scaled_code, k5_soft_exact, soft_l1,
        soft_distance_matrix, quantize_threshold, binary_diameter_two,
        multiplex_onehot, validate_transcript, transcript_distance,
        hoeffding_sufficient_width, reference_report,
        DEMO_N, DEMO_K, DEMO_SEED, DEMO_GRID_BITS, DEMO_GRID_AMPLITUDE,
    )
    from train_soft_probe import fit_soft_probe, run_pilot


EXPECTED_DIGEST_1024 = '2a99809deb1babd57d9d10e959775429470fafa90d390331b6d82f6f581dd5eb'


class TestSoftPrecision(unittest.TestCase):
    def test_five_state_three_coordinate_exact_certificate(self):
        z = k5_soft_exact()
        self.assertEqual(len(z), 5)
        self.assertTrue(all(len(x) == 3 for x in z))
        self.assertTrue(all(F(0) <= v <= 1 for x in z for v in x))
        d=soft_distance_matrix(z)
        self.assertEqual(set(d[i][j] for i in range(5) for j in range(i)),
                         {F(1),F(4,3),F(2)})
        self.assertEqual(d[0][4], F(1))
        self.assertEqual(d[1][4], F(4,3))
        # The code is linear in the frozen one-hot state representation.
        self.assertEqual([tuple(z[j][i] for i in range(3)) for j in range(5)], list(z))

    def test_thresholds_any_coordinate_policy_cannot_preserve_k5(self):
        z=k5_soft_exact()
        # Between 0,1/3,1 there are four distinct threshold patterns per head.
        selectors=(F(-1),F(1,6),F(2,3),F(2))
        patterns=set()
        for thresholds in itertools.product(selectors, repeat=3):
            hard=quantize_threshold(z,thresholds)
            patterns.add(hard)
            self.assertFalse(binary_diameter_two(hard))
        self.assertGreater(len(patterns),0)
        self.assertEqual(quantize_threshold(z,(F(1,2),)*3)[0], 0)
        self.assertEqual(quantize_threshold(z,(F(1,2),)*3)[4], 0)
        # Independent combinatorial oracle: at most four vertices fit a
        # diameter-two subset of the full three-dimensional Boolean cube.
        self.assertTrue(all(not binary_diameter_two(tuple(subset))
                            for subset in itertools.combinations(range(8),5)))

    def test_binary_four_coordinates_are_sharp_for_k5(self):
        self.assertTrue(binary_diameter_two((0,1,2,4,8)))

    def test_pinned_1024_rational_grid(self):
        code=seeded_scaled_code(DEMO_N,DEMO_K,DEMO_SEED,DEMO_GRID_AMPLITUDE)
        self.assertEqual(code.digest(),EXPECTED_DIGEST_1024)
        self.assertEqual(code.extreme_hamming(),(91,168))
        self.assertTrue(code.check(F(1),F(2)))
        self.assertEqual(code.extreme_hamming()[0]*code.amplitude,F(364,341))
        self.assertEqual(code.extreme_hamming()[1]*code.amplitude,F(672,341))
        self.assertEqual(code.amplitude*DEMO_K,F(1024,341))
        self.assertLessEqual(code.amplitude,1)
        self.assertEqual(code.amplitude.denominator,341)
        self.assertLess(code.k,code.n-1)
        self.assertEqual(DEMO_GRID_BITS,10)
        self.assertEqual(code.amplitude*1023,12)
        # A fixed conventional half-threshold zeroes all coordinates.
        self.assertLess(code.amplitude,F(1,2))
        # A low threshold restores the Hamming code, violating L1<=2.
        self.assertGreater(code.extreme_hamming()[1],2)

    def test_hoeffding_existence_window(self):
        for n in (5,16,256,1024):
            k=hoeffding_sufficient_width(n)
            self.assertGreater(k,18*math.log(n*(n-1)))
            self.assertLess(n*(n-1)*math.exp(-k/18),1)
        # Grid variants use delta=1/10 and q>=2k, giving b=O(log log n).
        n=1024
        k=hoeffding_sufficient_width(n,F(1,10))
        q=(1<<((2*k+1).bit_length()))-1
        self.assertGreaterEqual(q,2*k)
        a=F((5*q + 2*k-1)//(2*k),q)
        self.assertGreaterEqual(a,F(5,2*k))
        self.assertLessEqual(a,F(3,k))
        self.assertLessEqual(F(3,5)*k*a,F(9,5))
        self.assertGreaterEqual(F(2,5)*k*a,1)

    def test_recurrent_space_time_exact_witnesses(self):
        for n,k in ((5,1),(6,2),(17,4),(65,8)):
            transcript=multiplex_onehot(n,k)
            self.assertEqual(len(transcript[0]),math.ceil((n-1)/k))
            self.assertTrue(validate_transcript(transcript,k))
            self.assertEqual(min(transcript_distance(a,b)
                                 for a,b in itertools.combinations(transcript,2)),1)
            self.assertEqual(max(transcript_distance(a,b)
                                 for a,b in itertools.combinations(transcript,2)),2)
        with self.assertRaises(ValueError):
            multiplex_onehot(5,0)

    def test_bounded_coordinated_extension_cones(self):
        # Finite input samples are regular-simplex Hilbert points e_i/sqrt(2).
        # Each coordinate has scaled values 0 or a; its clipped McShane
        # extension is a-Lipschitz; summing k heads costs k*a <= constant.
        # Check sample interpolation and 100 deterministic nearby pairs.
        from random import Random
        code=seeded_scaled_code(8,32,119,F(3,32))
        n,k=code.n,code.k
        points=[[1/math.sqrt(2) if i==j else 0.0 for j in range(n)]
                for i in range(n)]
        def extend(x):
            ds=[math.dist(x,p) for p in points]
            return tuple(min(float(code.amplitude),
                             min((float(code.amplitude)*((code.words[v]>>j)&1)
                                  +float(code.amplitude)*ds[v]) for v in range(n)))
                         for j in range(k))
        for i,p in enumerate(points):
            target=tuple(float(code.amplitude)*((code.words[i]>>j)&1)
                         for j in range(k))
            for actual,expected in zip(extend(p), target):
                self.assertAlmostEqual(actual,expected,places=12)
        rng=Random(3502)
        for _ in range(100):
            x=[rng.uniform(-2,2) for _ in range(n)]
            y=[rng.uniform(-2,2) for _ in range(n)]
            self.assertLessEqual(sum(abs(a-b) for a,b in zip(extend(x),extend(y))),
                                 3*math.dist(x,y)+1e-11)

    def test_training_exploratory_seed_replay(self):
        all_pairs=run_pilot()
        self.assertEqual(all_pairs['soft_feasible_count'],10)
        self.assertEqual(all_pairs['threshold_half_hard_feasible_count'],0)
        self.assertEqual(all_pairs['train_feasible_count'],10)
        # Exploratory split, selected before this committed replay, not a
        # preregistered estimate of real-model generalization performance.
        heldout=((0,4),(1,3),(2,4))
        split=run_pilot(heldout_edges=heldout)
        self.assertEqual(split['train_feasible_count'],10)
        self.assertEqual(split['heldout_feasible_count'],2)
        self.assertEqual(split['soft_feasible_count'],2)
        self.assertEqual(split['threshold_half_hard_feasible_count'],0)
        for p in (all_pairs,split):
            self.assertEqual([x['seed'] for x in p['runs']],list(range(10)))
            self.assertTrue(all(0<=v<=1 for x in p['runs'] for row in x['state'] for v in row))

    def test_reference_receipt_stability(self):
        r=reference_report()
        self.assertEqual(r['demo_digest'],EXPECTED_DIGEST_1024)
        self.assertEqual(r['soft_k5_pairwise_distances'],['1','2','4/3'])
        self.assertEqual(r['binary_one_shot_min_channels'],1023)
        self.assertEqual(r['demo_l1_max'],'672/341')


if __name__=='__main__':
    unittest.main()
