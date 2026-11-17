import math
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from test_continuous import ring, caps, gaps
from quantization import uniform_quantize, continuous_to_quantized_polygon, quantization_bit_knee
from continuous import rat
from winding import pairwise_windings


class QuantizedBraidTests(unittest.TestCase):
    def test_quantizer_uniform_error(self):
        src=[[[-1.,0],[1.,0]], [[-1.,0],[1.,0]]]
        q=uniform_quantize(src,bits=4,range_abs='1')
        for rowa,rowb in zip(src,q):
            for a,b in zip(rowa,rowb):
                for p,r in zip(a,b):
                    self.assertLessEqual(abs(r-rat(p)),rat(1)/16)

    def test_unquantized_winding_matches_proved_knee(self):
        frames=ring(64)
        check=quantization_bit_knee(frames,range_abs='3',interval_lengths=gaps(64),
                                    pair_lipschitz_bounds=caps(2.000001),max_bits=16)
        self.assertTrue(check['certified'])
        self.assertLessEqual(check['bit_knee'],8)
        q=uniform_quantize(frames,bits=check['bit_knee'],range_abs='3')
        self.assertEqual(pairwise_windings(q),pairwise_windings(frames))

    def test_bit_knee_monotonic_worst_case(self):
        frames=ring(64)
        results=[continuous_to_quantized_polygon(frames,bits=b,range_abs='3',
                interval_lengths=gaps(64),pair_lipschitz_bounds=caps(2.000001))
                for b in range(1,11)]
        passing=[i for i,r in enumerate(results) if r['worst_case_passes']]
        self.assertTrue(passing)
        self.assertEqual(passing,list(range(passing[0],len(results))))

    def test_saturation_refused(self):
        frames=ring(64,radius=2)
        with self.assertRaisesRegex(ValueError,'saturation'):
            uniform_quantize(frames,bits=8,range_abs='1')

    def test_unclosed_loop_refused(self):
        frames=ring(64)
        frames[-1][0][0] += 0.01
        report=continuous_to_quantized_polygon(frames,bits=8,range_abs='3',
                   interval_lengths=gaps(64),pair_lipschitz_bounds=caps(2))
        self.assertFalse(report['passes'])

    def test_aliasing_requires_true_L_even_with_high_bits(self):
        frames=ring(4,turns=4)
        result=continuous_to_quantized_polygon(frames,bits=30,range_abs='3',
                   interval_lengths=gaps(4),pair_lipschitz_bounds=caps(8.000001))
        self.assertFalse(result['passes'])
        self.assertFalse(result['worst_case_passes'])

    def test_actual_error_may_certify_before_worst_case(self):
        frames=ring(64)
        results=[continuous_to_quantized_polygon(frames,bits=b,range_abs='3',
                interval_lengths=gaps(64),pair_lipschitz_bounds=caps(2.000001))
                for b in range(1,9)]
        a=next(i for i,r in enumerate(results) if r['passes'])
        b=next(i for i,r in enumerate(results) if r['worst_case_passes'])
        self.assertLessEqual(a,b)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            uniform_quantize(ring(64),bits=0,range_abs='3')
        with self.assertRaises(ValueError):
            uniform_quantize(ring(64),bits=True,range_abs='3')


if __name__ == '__main__':
    unittest.main()

class ThreeStrandLinkPreservation(unittest.TestCase):
    def test_milnor_survives_certified_quantization_even_when_extractor_refuses(self):
        from scm05_reference import C, refine_poly, exact_PL_upper_bounds
        from knot import synthetic_braid, extract_closed_braid, InvalidBraidTrace
        from milnor import triple_milnor_nilpotent
        raw=synthetic_braid(3,C)
        self.assertEqual(triple_milnor_nilpotent(3,extract_closed_braid(raw).word)['mu123'],1)
        frames=refine_poly(raw,4)
        result=quantization_bit_knee(frames,range_abs='3',
                    interval_lengths=['1']*(len(frames)-1),
                    pair_lipschitz_bounds=exact_PL_upper_bounds(frames),max_bits=12)
        self.assertTrue(result['certified'])
        self.assertEqual(result['bit_knee'],3)
        rounded=uniform_quantize(frames,bits=3,range_abs='3')
        with self.assertRaisesRegex(InvalidBraidTrace,'x-coordinate tie'):
            extract_closed_braid(rounded)
        # A certified collision-free homotopy preserves the oriented closure
        # despite the crossing extractor's deliberately strict chart rejection.
        self.assertTrue(result['certificate']['worst_case_passes'])
