"""Independent topological checks using the SCM-01 synthetic braid generator."""
import math
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knot import InvalidBraidTrace, synthetic_braid, extract_closed_braid
from winding import (pairwise_windings, braid_pairwise_linking,
                     verify_pure_braid, verified_sampling_bound, compression_braid_certificate)


class WindingTests(unittest.TestCase):
    def test_hopf_link_orientation(self):
        for word, expected in [([1,1],-1), ([-1,-1],1), ([1,-1],0), ([1,1,1,1],-2)]:
            frames=synthetic_braid(2,word)
            self.assertEqual(pairwise_windings(frames)[0,1], expected)
            self.assertEqual(braid_pairwise_linking(extract_closed_braid(frames))[0,1], -expected)

    def test_three_strand_independent_check(self):
        for word in ([1,1,2,2],[1,2,2,1],[1,1,-2,-2],[-1,-1,2,2]):
            frames=synthetic_braid(3,word)
            self.assertEqual(verify_pure_braid(frames,extract_closed_braid(frames)),
                             pairwise_windings(frames))

    def test_random_pure_inserted_inverse_pairs(self):
        rng=random.Random(1407)
        for _ in range(40):
            word=[]
            for _ in range(4):
                g=rng.choice([-2,-1,1,2])
                h=rng.choice([-2,-1,1,2])
                word.extend([h,g,g,-h])
            frames=synthetic_braid(3,word)
            self.assertEqual(verify_pure_braid(frames,extract_closed_braid(frames)),
                             pairwise_windings(frames))

    def test_open_labeled_cycle_rejected(self):
        frames=synthetic_braid(2,[1])
        with self.assertRaisesRegex(InvalidBraidTrace,'pure braid'):
            pairwise_windings(frames)

    def test_segment_collision_detected(self):
        frames=[[(1.,0.),(0.,0.)],[(-1.,0.),(0.,0.)],[(1.,0.),(0.,0.)]]
        with self.assertRaisesRegex(InvalidBraidTrace,'collision'):
            pairwise_windings(frames)

    def test_sampling_bound_is_external(self):
        count=64
        frames=[[(math.cos(2*math.pi*k/count),math.sin(2*math.pi*k/count)),(0.,0.)]
                for k in range(count+1)]
        dt=[2*math.pi/count]*count
        self.assertTrue(verified_sampling_bound(frames,interval_lengths=dt,
                        pair_derivative_bounds={(0,1):1.0}))
        self.assertFalse(verified_sampling_bound(frames,interval_lengths=dt,
                         pair_derivative_bounds={(0,1):100.0}))
        with self.assertRaises(ValueError):
            verified_sampling_bound(frames,interval_lengths=dt,pair_derivative_bounds={})

    def test_topology_preserved_by_bounded_compression(self):
        frames=synthetic_braid(2,[1,1])
        compressed=[[(x+0.02*(k%3), y-0.01*(k%3)) for x,y in row]
                    for k,row in enumerate(frames)]
        # Keep the compressed trajectory closed as required.
        compressed[-1]=compressed[0]
        receipt=compression_braid_certificate(frames,compressed)
        self.assertTrue(receipt['certified'],receipt)
        self.assertEqual(pairwise_windings(frames),pairwise_windings(compressed))
        # A common translation is a configuration-space gauge; relative error is zero.
        translated=[[(x+100,y-100) for x,y in row] for row in frames]
        self.assertTrue(compression_braid_certificate(frames,translated)['certified'])

    def test_compression_certificate_refuses_collapse(self):
        frames=synthetic_braid(2,[1,1])
        collapsed=[[tuple(p) for p in row] for row in frames]
        collapsed[1][0]=collapsed[1][1]
        self.assertFalse(compression_braid_certificate(frames,collapsed)['certified'])

    def test_near_collision_rejected(self):
        frames=[[(0.0, 0.0),(1.e-8,0.0)],[(0.,0.),(0.0,1.e-8)],
                [(0.,0.),(1.e-8,0.)]]
        with self.assertRaises(InvalidBraidTrace):
            pairwise_windings(frames)


if __name__=='__main__': unittest.main()
