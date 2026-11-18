"""NOR-47 exact GMP kernel parity, overflow limits and source signature controls."""
from __future__ import annotations
import copy
import random
import unittest
from contextlib import contextmanager

from research.nor46_exact_integer_trace import trace_verifier as previous
from research.nor46_exact_integer_trace import historical_smoke as old_smoke
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp


@contextmanager
def benchmark_only_override(fn):
    """Global patch is single-threaded, restores on every exit, never deploy."""
    saved=previous.exact_convolution_prefix
    previous.exact_convolution_prefix=fn
    try:yield
    finally:previous.exact_convolution_prefix=saved


class GMPExactCertificateTests(unittest.TestCase):
    def test_all_random_exact_signed_convolutions_against_integer_oracle(self):
        rng=random.Random(2026114711)
        for n in (1,2,3,7,17,32,79):
            for bits in (0,2,7,63,255,4097):
                bound=(1<<bits) if bits else 0
                a=tuple(rng.randint(-bound,bound) for _ in range(n))
                b=tuple(rng.randint(-bound,bound) for _ in range(n))
                ref=tuple(sum(a[j]*b[i-j] for j in range(i+1))
                          for i in range(n))
                self.assertEqual(previous.exact_convolution_prefix(a,b),ref)
                self.assertEqual(gmp.full_gmp(a,b),ref)
                self.assertEqual(gmp.hybrid_gmp(a,b),ref)

    def test_edge_zero_max_signed_limb_and_inverse_tamper(self):
        for n in (1,2,4,37):
            pairs=[
                (tuple(0 for _ in range(n)),tuple(0 for _ in range(n))),
                (tuple((1<<4019)*(-1)**i for i in range(n)),
                 tuple((1<<4019)*(-1)**(i+1) for i in range(n))),
                (tuple((1<<4019)-1 for _ in range(n)),
                 tuple(-((1<<4019)-1) for _ in range(n)))
            ]
            for a,b in pairs:
                reference=previous.exact_convolution_prefix(a,b)
                self.assertEqual(gmp.full_gmp(a,b),reference)
                self.assertEqual(gmp.hybrid_gmp(a,b),reference)
        rng=random.Random(4711)
        for n in (3,17,127):
            f=(1,)+tuple(rng.randint(-3,3) for _ in range(n-1))
            # NOR45 independently computes inverse, never uses GMP.
            from research.nor45_authorized_program import independent_execution_verifier as old
            inverse=old.unary(f,"R")
            for kernel in (gmp.full_gmp,gmp.hybrid_gmp):
                with benchmark_only_override(kernel):
                    previous.validate_inverse(f,inverse)
                    damaged=list(inverse);damaged[-1]+=1
                    with self.assertRaises(previous.TraceError):
                        previous.validate_inverse(f,damaged)

    def test_reject_same_invalid_domain_and_caps_as_original(self):
        cases=[
            ((),()),
            ((1,),()),
            ((1,True),(1,0)),
            ((1,0),(False,0)),
            ((1<<32769,),(1,)),
            ((1,),(1<<32769,)),
        ]
        for a,b in cases:
            for backend in (previous.exact_convolution_prefix,gmp.full_gmp,gmp.hybrid_gmp):
                with self.assertRaises((previous.TraceError,ValueError,TypeError)):
                    backend(a,b)
        for fn in (gmp.full_gmp,gmp.hybrid_gmp):
            with self.assertRaises(gmp.GMPTraceError):
                gmp.exact_convolution_prefix_gmp((1,),(1,),packing="not a backend")

    def test_original_16_signed_authority_negative_cases_unchanged(self):
        # Complete NOR46 source-locked first receipt, including valid signed
        # execution and dishonest authorized program controls, is deterministic.
        saved=previous.exact_convolution_prefix
        archived=old_smoke.run()
        self.assertEqual(archived["historical_pair_count"],16)
        from research.nor46_exact_integer_trace import original_archive_audit as lock
        original_result=lock.audit()
        self.assertEqual(original_result["old_source_pairs"],16)
        for kernel in (gmp.full_gmp,gmp.hybrid_gmp):
            with benchmark_only_override(kernel):
                reproduced=old_smoke.run()
            self.assertEqual(reproduced,archived)
        self.assertIs(previous.exact_convolution_prefix,saved)

    def test_override_releases_global_even_on_failure(self):
        initial=previous.exact_convolution_prefix
        with self.assertRaises(RuntimeError):
            with benchmark_only_override(gmp.full_gmp):
                self.assertIs(previous.exact_convolution_prefix,gmp.full_gmp)
                raise RuntimeError("test exception")
        self.assertIs(previous.exact_convolution_prefix,initial)


if __name__=="__main__":
    unittest.main()
