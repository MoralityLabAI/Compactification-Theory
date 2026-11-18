"""NOR54 source-free controls using only toy and immutable NOR51–53 archives.

The first NOR54 60 source identities must be generated only by a separate,
registration-gated upstream source-only CI job.
"""
from __future__ import annotations
import random,unittest
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as full
from research.nor52_sparse_inverse import algebra as sparse
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer,cohort

class SourceFreeMatchedFrontierTests(unittest.TestCase):
    def test_five_full_vector_root_parity_and_codec_on_fresh_toys(self):
        saved=nor46.exact_convolution_prefix
        for stride in (1,4,8):
            for m in (1,2,9,17,37,65):
                rng=random.Random(54000+13*stride+m)
                f=(1,)+tuple((rng.choice((-3,-2,-1,0,1,3))
                      if i%stride==0 else 0) for i in range(1,m))
                reference=full.native_single_inverse(f)
                expected=(reference,native.independently_computed_root(reference))
                full_witness=full.direct_with_root(f)
                cached=sparse.native_compressed_root(f,stride)
                fast_full=fast_kernel.full_verified_root(f,reference)
                fast_compressed=fast_kernel.compressed_verified_root(f,reference,stride)
                e2e=producer.compressed_produce_serialize_fast_verify_root(f,stride)
                self.assertEqual(full_witness,expected)
                self.assertEqual(cached,expected)
                self.assertEqual(fast_full,expected)
                self.assertEqual(fast_compressed,expected)
                self.assertEqual(e2e,expected)
                fresh=producer.produce_compressed_full_witness(f,stride)
                self.assertEqual(fresh,reference)
                wire=producer.canonical_witness_bytes(fresh)
                self.assertEqual(producer.parse_witness_bytes(wire,m),fresh)
        self.assertIs(saved,nor46.exact_convolution_prefix)

    def test_untrusted_witness_codec_negative_and_tampering(self):
        bad=(
            b'["01","2"]',b'["-0","1"]',
            b'["1","-"]',b'["1",true]',b'["1",2]',
            b'{"0":"1"}',b'["1"]',b'["1","2","3"]',
            b'["1","2.0"]',b'["1","+2"]',b'not-json',
            b'["1","\xff"]'
        )
        for wire in bad:
            with self.subTest(wire=wire),self.assertRaises((ValueError,TypeError)):
                producer.parse_witness_bytes(wire,2)
        with self.assertRaises((ValueError,TypeError)):
            producer.parse_witness_bytes('["1","2"]',2)
        with self.assertRaises((ValueError,TypeError)):
            producer.parse_witness_bytes(b'["1","2"]',True)
        with self.assertRaises((ValueError,TypeError)):
            producer.canonical_witness_bytes([True,1])
        with self.assertRaises((ValueError,TypeError)):
            producer.canonical_witness_bytes(["1",1])
        f=[1]+[0]*40
        f[4]=-2
        f[8]=3
        f=tuple(f)
        witness=full.native_single_inverse(f)
        self.assertEqual(producer.compressed_produce_serialize_fast_verify_root(f,4),
                         full.direct_with_root(f))
        wrong=list(f);wrong[1]=1
        for candidate in (wrong,(2,*f[1:])):
            with self.assertRaises((ValueError,TypeError)):
                producer.compressed_produce_serialize_fast_verify_root(candidate,4)
        wrong_witness=list(witness);wrong_witness[1]=1
        with self.assertRaises((ValueError,TypeError)):
            fast_kernel.compressed_verified_root(f,wrong_witness,4)
        wrong_witness=list(witness);wrong_witness[8]+=1
        with self.assertRaises((ValueError,TypeError)):
            fast_kernel.compressed_verified_root(f,wrong_witness,4)
        with self.assertRaises((ValueError,TypeError)):
            fast_kernel.compressed_verified_root(f,witness[:-1],4)
        for bad_stride in (0,2,3,7,9,True,"4"):
            with self.assertRaises((ValueError,TypeError)):
                producer.compressed_produce_serialize_fast_verify_root(f,bad_stride)
        bad_root="0"*64
        self.assertNotEqual(full.root(witness),bad_root)
        self.assertNotEqual(producer.compressed_produce_serialize_fast_verify_root(f,4)[1],
                            bad_root)

    def test_original_resource_limits_and_immutable_archive(self):
        for bad in (
            (1,True),(1,1<<(nor46.MAX_COEFFICIENT_BITS+1))
        ):
            with self.assertRaises((ValueError,TypeError)):
                producer.produce_compressed_full_witness(bad,1)
        from research.nor53_native_pack import original_archive_audit
        report=original_archive_audit.audit()
        assert report["verdict"]=="PASS (5/5)"
        self.assertEqual(len(cohort.first_previous_ids()),168)
        self.assertEqual(cohort.REGISTERED,
                         "10e7ac3c3fdefec89d90d7a98a947099db8af86e")
        self.assertEqual(cohort.SOURCE_DOMAIN,"NOR54/SOURCE/v1")
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)

if __name__=="__main__":
    unittest.main()
