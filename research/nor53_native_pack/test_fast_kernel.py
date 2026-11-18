"""NOR53 source-free exact signed pack/unpack, attack and archive tests.

Only toy vectors and prior NOR51/NOR52 immutable source archives are opened.
Do not generate the fresh NOR53 60-source fixture or first timing here.
"""
from __future__ import annotations
import random,unittest
import gmpy2
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as original
from research.nor51_exact_r_crossover import algebra as inverse
from research.nor52_sparse_inverse import algebra as sparse
from research.nor53_native_pack import fast_kernel as fast,cohort

class NativeGMPPackExactTests(unittest.TestCase):
    def test_pack_order_and_raw_mpz_identity(self):
        self.assertIn(fast.pack_orientation(),
                      ("least_significant_first","most_significant_first"))
        self.assertTrue(fast.unpack_is_little_endian())
        for width,nums in (
            (4,[1,2,3]),(4,[0,1,0]),(8,[255,0,1]),
            (33,[1,(1<<32)+1,0,17]),
            (4098,[1<<4097,(1<<4096)+13,0,71])):
            self.assertEqual(fast._pack_low_first(nums,width),
                             original._gmp_pack(nums,0,width))
            raw=fast._pack_low_first(nums,width)
            unpacked=[int(x) for x in gmpy2.unpack(raw,width)]
            unpacked.extend([0]*max(0,len(nums)-len(unpacked)))
            self.assertEqual(unpacked[:len(nums)],nums)
        with self.assertRaises((ValueError,TypeError)):
            fast._pack_low_first([-1,1],4)
        with self.assertRaises((ValueError,TypeError)):
            fast._pack_low_first([True,1],4)
        with self.assertRaises((ValueError,TypeError)):
            fast._pack_low_first([16],4)

    def test_exact_signed_convolution_parity_toy_and_4097_bit(self):
        for seed in (3,29,103,211):
            rng=random.Random(seed)
            for m in (1,2,3,5,17,31,65,127):
                for _ in range(2):
                    a=(1,)+tuple(rng.choice((-5,-2,-1,0,0,1,3,5))
                                 for i in range(m-1))
                    b=(1,)+tuple(rng.choice((-7,-4,-1,0,0,1,2,6))
                                 for i in range(m-1))
                    actual=fast.exact_convolution_prefix_fast(a,b)
                    self.assertEqual(actual,original.full_gmp(a,b))
                    self.assertEqual(actual,nor46.exact_convolution_prefix(a,b))
        a=(1, (1<<4096)+5, -(1<<4000),7)
        b=(1, -(1<<4096)+3, (1<<4050)+11,-9)
        self.assertEqual(fast.exact_convolution_prefix_fast(a,b),
                         original.full_gmp(a,b))
        with self.assertRaises((ValueError,TypeError)):
            fast.exact_convolution_prefix_fast((1,True),(1,0))
        with self.assertRaises((ValueError,TypeError)):
            fast.exact_convolution_prefix_fast((1,0),(1,))
        with self.assertRaises((ValueError,TypeError)):
            fast.exact_convolution_prefix_fast(
                (1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),(1,0))

    def test_full_and_sparse_checked_witness_attacks(self):
        global_function=nor46.exact_convolution_prefix
        for stride in (1,4,8):
            length=17*stride+1
            rng=random.Random(5300+stride)
            f=tuple([1]+[(rng.choice((-2,0,3)) if i%stride==0 else 0)
                           for i in range(1,length)])
            reference=inverse.native_single_inverse(f)
            full=fast.full_verified_root(f,reference)
            compressed=fast.compressed_verified_root(f,reference,stride)
            self.assertEqual(full,compressed)
            self.assertEqual(full,inverse.gmp_witness_with_root(f,reference))
            self.assertEqual(compressed,sparse.gmp_compressed_root(f,reference,stride))
            tampered=list(reference);tampered[-1]+=1
            with self.assertRaises((ValueError,TypeError)):
                fast.full_verified_root(f,tampered)
            with self.assertRaises((ValueError,TypeError)):
                fast.compressed_verified_root(f,tampered,stride)
            with self.assertRaises((ValueError,TypeError)):
                fast.full_verified_root(f,reference[:-1])
            with self.assertRaises((ValueError,TypeError)):
                fast.full_verified_root(f,[True,*reference[1:]])
            with self.assertRaises((ValueError,TypeError)):
                fast.full_verified_root((2,*f[1:]),reference)
            if stride!=1:
                altered=list(reference);altered[1]=1
                with self.assertRaises((ValueError,TypeError)):
                    fast.compressed_verified_root(f,altered,stride)
                altered=list(f);altered[1]=1
                with self.assertRaises((ValueError,TypeError)):
                    fast.compressed_verified_root(altered,reference,stride)
            for bad in (-1,0,2,3,True,"4"):
                if bad!=stride:
                    with self.assertRaises((ValueError,TypeError)):
                        fast.compressed_verified_root(f,reference,bad)
        self.assertIs(global_function,nor46.exact_convolution_prefix)

    def test_original_archives_and_no_new_source_generation(self):
        from research.nor52_sparse_inverse import original_archive_audit
        assert original_archive_audit.audit()["verdict"]=="PASS (5/5)"
        self.assertEqual(len(cohort.original_source_ids()),108)
        self.assertEqual(cohort.REGISTERED,
                         "ae56f414251a8485d127fc8f9899a24237870adf")
        self.assertEqual((len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES)),60)

if __name__=="__main__":unittest.main()
