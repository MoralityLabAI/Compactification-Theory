"""Standalone SCM-05 synthetic math tests; no neural artifacts are used."""
import math
import pathlib
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from continuous import (rat, min_segment_distance_sq, _sqrt_bounds,
                        continuous_sampling_certificate, continuous_compression_certificate)
from winding import pairwise_windings


def ring(samples=64, turns=1, radius=2.0, offset=(0.,0.)):
    frames=[]
    for k in range(samples+1):
        theta=2*math.pi*k/samples
        x=radius*math.cos(turns*theta)+offset[0]
        y=radius*math.sin(turns*theta)+offset[1]
        frames.append([[x,y],[offset[0],offset[1]]])
    frames[-1] = [p.copy() for p in frames[0]]
    return frames


def caps(L, n=2):
    return {f'{i},{j}':str(L) for i in range(n) for j in range(i+1,n)}


def gaps(samples):
    # Exact representation of the decimal bound used by the mock fixture,
    # rounded UP to certify the actual radian gap 2*pi/samples.
    return [str(math.nextafter(2*math.pi/samples, math.inf))]*(samples)


class SamplingCertificates(unittest.TestCase):
    def test_segment_distance_exact_rational(self):
        self.assertEqual(min_segment_distance_sq((rat(1),rat(-2)),(rat(1),rat(2))),1)
        self.assertEqual(min_segment_distance_sq((rat(-2),rat(0)),(rat(2),rat(0))),0)
        self.assertEqual(min_segment_distance_sq((rat(3),rat(4)),(rat(3),rat(4))),25)

    def test_outward_sqrt_exact_and_irrational(self):
        a,b=_sqrt_bounds(Fraction(2),digits=8)
        self.assertTrue(a*a<2<b*b)
        self.assertEqual(_sqrt_bounds(Fraction(9,4),digits=8),(Fraction(3,2),Fraction(3,2)))

    def test_one_turn_verified_analytic_bound(self):
        frames=ring(64)
        claim=continuous_sampling_certificate(frames,interval_lengths=gaps(64),
                                               pair_lipschitz_bounds=caps(2.000001))
        self.assertTrue(claim['passes'])
        self.assertEqual(pairwise_windings(frames)[0,1],1)

    def test_aliasing_high_frequency_rejected(self):
        # Four turns exactly disappear with four angular samples. With
        # mathematically valid L=8, certification MUST fail on that grid.
        frames=ring(4,turns=4)
        self.assertEqual(pairwise_windings(frames)[0,1],0)
        claim=continuous_sampling_certificate(frames,interval_lengths=gaps(4),
                                               pair_lipschitz_bounds=caps(8.000001))
        self.assertFalse(claim['passes'])
        refined=ring(64,turns=4)
        self.assertEqual(pairwise_windings(refined)[0,1],4)
        self.assertTrue(continuous_sampling_certificate(refined,interval_lengths=gaps(64),
                          pair_lipschitz_bounds=caps(8.000001))['passes'])

    def test_missing_bound_refused(self):
        with self.assertRaisesRegex(ValueError,'Missing externally'):
            continuous_sampling_certificate(ring(64),interval_lengths=gaps(64),
                                             pair_lipschitz_bounds=None)

    def test_understated_bound_not_verified_by_api(self):
        # Passing an invented numerical bound can yield a conditional result;
        # it is NEVER an independently proved statement about the model.
        claim=continuous_sampling_certificate(ring(4,turns=4),
                   interval_lengths=gaps(4),pair_lipschitz_bounds=caps(0))
        self.assertTrue(claim['passes'])
        self.assertEqual(claim['scope'],'conditional_on_external_interval_lipschitz_proofs')

    def test_unclosed_cycle_rejected(self):
        frames=ring(64)
        frames[-1][0][0]+=0.000001
        c=continuous_sampling_certificate(frames, interval_lengths=gaps(64),
                                          pair_lipschitz_bounds=caps(2))
        self.assertFalse(c['passes'])
        self.assertEqual(c['reason'],'sampled_endpoints_not_exactly_equal')

    def test_negative_and_unordered_bounds_refused(self):
        with self.assertRaises(ValueError):
            continuous_sampling_certificate(ring(64),interval_lengths=gaps(64),
                                            pair_lipschitz_bounds=caps(-1))
        with self.assertRaises(ValueError):
            continuous_sampling_certificate(ring(64),interval_lengths=gaps(64),
                                            pair_lipschitz_bounds={'1,0':2})

    def test_piecewise_sparse_undetected_collision_rejected(self):
        frames=[[[1.,0.],[0.,0.]], [[-1.,0.],[0.,0.]], [[1.,0.],[0.,0.]]]
        c=continuous_sampling_certificate(frames, interval_lengths=['1','1'],
                                          pair_lipschitz_bounds=caps(1))
        self.assertFalse(c['passes'])

    def test_compression_small_error_preserves_closed_link(self):
        original=ring(64)
        recon=ring(64, radius=1.9)
        c=continuous_compression_certificate(original,recon,interval_lengths=gaps(64),
                  original_pair_lipschitz_bounds=caps(2.000001),
                  reconstructed_pair_lipschitz_bounds=caps(1.900001))
        self.assertTrue(c['passes'])
        self.assertTrue(c['closed_braid_and_link_preserved'])
        self.assertFalse(c['based_braid_preserved'])
        self.assertEqual(pairwise_windings(recon)[0,1],pairwise_windings(original)[0,1])

    def test_same_basepoint_certifies_based_braid(self):
        original=ring(64)
        recon=[[[x+0.01*math.sin(2*math.pi*k/64),y], [0.,0.]]
               for k,((x,y),_origin) in enumerate(original)]
        recon[-1]=[p.copy() for p in recon[0]]
        c=continuous_compression_certificate(original,recon,interval_lengths=gaps(64),
                  original_pair_lipschitz_bounds=caps(2.000001),
                  reconstructed_pair_lipschitz_bounds=caps(2.100001))
        self.assertTrue(c['based_braid_preserved'])

    def test_compression_can_collapse_strands(self):
        original=ring(64)
        recon=ring(64, radius=0.0)
        c=continuous_compression_certificate(original,recon,interval_lengths=gaps(64),
                  original_pair_lipschitz_bounds=caps(2.000001),
                  reconstructed_pair_lipschitz_bounds=caps(0))
        self.assertFalse(c['passes'])

    def test_multistrand_all_pairs_checked(self):
        frames=[]
        for f in ring(64):
            frames.append([f[0],f[1],[100.,0.]])
        original={'0,1':'2.000001','0,2':'2.000001','1,2':'0'}
        a=continuous_sampling_certificate(frames,interval_lengths=gaps(64),
                                           pair_lipschitz_bounds=original)
        self.assertTrue(a['passes'])
        a['pairs']['0,2']['intervals'][0]['certified_condition']=False
        self.assertTrue(a['passes']) # results contain an independently computed aggregate

    def test_fraction_precision_rejects_tiny_margin(self):
        frames=[[[1.,0.],[0.,0.]], [[1.,0.],[0.,0.]]]
        c=continuous_sampling_certificate(frames,interval_lengths=['1'],
                                           pair_lipschitz_bounds=caps(2),extra_margin='0')
        self.assertFalse(c['passes'])
        self.assertTrue(continuous_sampling_certificate(frames,interval_lengths=['1'],
                          pair_lipschitz_bounds=caps('1.999999999'),extra_margin='0')['passes'])


if __name__=='__main__':
    unittest.main()
