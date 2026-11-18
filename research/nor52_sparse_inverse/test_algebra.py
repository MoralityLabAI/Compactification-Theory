"""NOR52 source-free exact sparse inverse and security controls.

Only toy and previously archived NOR51 inputs; never generates new 60 sources.
"""
from __future__ import annotations
import itertools,random,unittest
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as previous
from research.nor52_sparse_inverse import algebra,cohort,phase_kernel

def multiply_naive(f,w):
    return tuple(sum(f[j]*w[i-j] for j in range(i+1)) for i in range(len(f)))

class SparseExactTests(unittest.TestCase):
    def test_exhaustive_toy_finite_truncated_series(self):
        for stride in (1,4,8):
            for L in (2,3,4,5):
                for coefficients in itertools.product((-1,0,1),repeat=L-1):
                    compressed=(1,)+coefficients
                    full_length=(L-1)*stride+1
                    full=algebra.expand(compressed,full_length,stride)
                    reference=previous.native_single_inverse(full)
                    restored,root=algebra.native_compressed_root(full,stride)
                    self.assertEqual(restored,reference)
                    self.assertEqual(root,native.independently_computed_root(reference))
                    self.assertEqual(algebra.compress(reference,stride),
                                     previous.native_single_inverse(compressed))
                    self.assertEqual(multiply_naive(full,reference),
                                     (1,)+(0,)*(full_length-1))
                    self.assertEqual(algebra.gmp_compressed_root(full,reference,stride),
                                     (reference,root))
                    self.assertEqual(algebra.gmp_full_root(full,reference),
                                     (reference,root))
            self.assertEqual(algebra.compressed_length(3073,stride),
                             (3072//stride)+1)

    def test_source_and_witness_tampering(self):
        f=[1]+[0]*24
        f[4]=2
        f[8]=-3
        f=tuple(f)
        witness=previous.native_single_inverse(f)
        self.assertEqual(algebra.gmp_compressed_root(f,witness,4),
                         algebra.gmp_full_root(f,witness))
        self.assertEqual(algebra.native_compressed_root(f,4),
                         algebra.native_full_root(f))
        for wrong_stride in (0,-1,2,3,7,9,True,"4",8):
            with self.assertRaises((TypeError,ValueError)):
                algebra.gmp_compressed_root(f,witness,wrong_stride)
        poisoned=list(witness);poisoned[5]=1
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(f,poisoned,4)
        modified=list(f);modified[3]=1
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(modified,witness,4)
        modified=list(f);modified[4]+=1
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(modified,witness,4)
        poisoned=list(witness);poisoned[8]*=-1
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(f,poisoned,4)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(f,witness[:-1],4)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(f,["1",*witness[1:]],4)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root(f,[True,*witness[1:]],4)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root((2,*f[1:]),witness,4)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root((1,True,*f[2:]),witness,1)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root((1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),
                                        (1,0),1)
        with self.assertRaises((TypeError,ValueError)):
            algebra.gmp_compressed_root((1,0),
                                        (1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),1)

    def test_dense_and_phase_diagnostics_exact(self):
        saved=nor46.exact_convolution_prefix
        for length in (1,2,4,13,33):
            rng=random.Random(12000+length)
            f=(1,)+tuple(rng.randint(-2,2) for _ in range(length-1))
            w=previous.native_single_inverse(f)
            self.assertEqual(algebra.native_full_root(f),algebra.native_compressed_root(f,1))
            self.assertEqual(algebra.gmp_full_root(f,w),algebra.gmp_compressed_root(f,w,1))
            summary=phase_kernel.instrumented_exact_product(f,w)
            self.assertEqual(set(phase_kernel.PHASES),{
                 "parameters_ns","packing_ns","multiplication_ns",
                 "extraction_ns","correction_ns"})
            self.assertEqual(summary["phase_total_ns"],
                             sum(summary[k] for k in phase_kernel.PHASES))
            self.assertTrue(summary["exact_signed_identity"])
            self.assertTrue(summary["reference_kernel_equal"])
        self.assertIs(saved,nor46.exact_convolution_prefix)

    def test_old_archive_and_registered_source_contract(self):
        from research.nor51_exact_r_crossover import original_archive_audit
        assert original_archive_audit.audit()["first_verdict"]=="FAIL (2/5)"
        assert len(cohort.first_prior_source_ids())==48
        self.assertEqual(cohort.REGISTERED,"9526627e21b429e7fea1928b3ddfb0e1dadc5e74")
        self.assertEqual(cohort.PROFILES,("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16"))
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)

if __name__=="__main__":
    unittest.main()
