"""NOR55 pre-data only tests: canonical source-bound compact signed envelope.

Tests synthetic toy sources and already SHA-archived first NOR54 evidence.
Does not generate NOR55's new 60-source cohort or observe its timing.
"""
from __future__ import annotations
import itertools,secrets,unittest
from hashlib import sha256

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as native
from research.nor52_sparse_inverse import algebra as support
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import codec,envelope,cohort

def source_fixture(stride=4):
    f=[1]+[0]*(5*stride)
    f[stride]=-2
    f[2*stride]=3
    return tuple(f)

def trusted(source,stride,fmt,seq=1,nonce=None,source_sha=None,root=None,session=None):
    inverse=native.native_single_inverse(source)
    return envelope.make_context(
       source=source,
       source_sha256=source_sha or sha256(b"NOR55/TOY/SOURCE/1234").hexdigest(),
       stride=stride,
       expected_root_sha256=root or native.root(inverse),
       payload_format=fmt,
       session_id=session or f"nor55-test-{fmt}",
       sequence=seq,
       nonce=nonce or secrets.token_hex(32)
    )

class ExactCompactAndAuthenticationTests(unittest.TestCase):
    def test_exhaustive_small_exact_integer_codec_and_full_root(self):
        for stride in (1,4,8):
            for coeffs in itertools.product((-2,-1,0,1,2),repeat=3):
                full=support.expand((1,)+coeffs,3*stride+1,stride)
                reference=native.native_single_inverse(full)
                raw=codec.encode_full_supported(reference,stride)
                self.assertEqual(codec.decode_full_supported(raw,len(full),stride),reference)
                ctx=trusted(full,stride,"compact-v1")
                env=envelope.make_signed_envelope(reference,ctx)
                actual=envelope.verify_signed_envelope(env,ctx,envelope.ReplayGuard())
                self.assertEqual(actual,(reference,native.root(reference)))
                self.assertEqual(support.compress(reference,stride),
                                 native.native_single_inverse((1,)+coeffs))

    def test_varint_magnitude_canonical_and_malicious_wire_bytes(self):
        for n in (0,1,127,128,255,16384,2**28):
            b=codec.encode_uvarint(n)
            self.assertEqual(codec.decode_uvarint(b,0),(n,len(b)))
        for raw in (b"\x80",b"\x81\x00",b"\x80\x00",b"\xff"*6):
            with self.assertRaises((ValueError,TypeError)):
                codec.decode_uvarint(raw,0)
        for invalid in (
            codec.MAGIC+b"\x01\x00\x01\x00",  # 0 encoded as leading byte 00
            codec.MAGIC+b"\x01\x01\x00",      # negative zero
            codec.MAGIC+b"\x01\x02\x01\x01", # noncanonical sign
            codec.MAGIC+b"\x01\x00\x01\x00", # nonminimal magnitude
            codec.MAGIC+b"\x01\x00\x01\x01\x00",
            codec.MAGIC+b"\x81\x00\x00\x01\x01",
            codec.MAGIC+b"\x01\x00\x80",
            b"invalid"
        ):
            with self.assertRaises((ValueError,TypeError)):
                codec.decode_reduced(invalid,1)
        a=(1,1<<4096,-(1<<4095),0)
        raw=codec.encode_reduced(a)
        self.assertEqual(codec.decode_reduced(raw,len(a)),a)
        with self.assertRaises((ValueError,TypeError)):
            codec.encode_reduced((1,1<<(nor46.MAX_COEFFICIENT_BITS+1)))
        with self.assertRaises((ValueError,TypeError)):
            codec.encode_reduced((1,True))
        with self.assertRaises((ValueError,TypeError)):
            codec.decode_reduced(raw+b"\x00",len(a))
        with self.assertRaises((ValueError,TypeError)):
            codec.decode_reduced(raw,3)

    def test_both_signed_codecs_and_replay_guard(self):
        f=source_fixture(4)
        reference=native.native_single_inverse(f)
        for fmt in envelope.FORMATS:
            guard=envelope.ReplayGuard()
            for seq in (1,2,3):
                ctx=trusted(f,4,fmt,seq=seq,session=f"exact-session-{fmt}")
                wire=envelope.make_signed_envelope(reference,ctx)
                self.assertEqual(envelope.verify_signed_envelope(wire,ctx,guard),
                                 (reference,native.root(reference)))
                with self.assertRaises((ValueError,TypeError)):
                    envelope.verify_signed_envelope(wire,ctx,guard)
            # Validate guard after a failed request leaves the challenge unused.
            ctx=trusted(f,4,fmt,seq=4,session=f"exact-session-{fmt}")
            good=envelope.make_signed_envelope(reference,ctx)
            bad=good[:-1]+bytes([good[-1]^1])
            with self.assertRaises((ValueError,TypeError)):
                envelope.verify_signed_envelope(bad,ctx,guard)
            self.assertEqual(envelope.verify_signed_envelope(good,ctx,guard),
                             (reference,native.root(reference)))

    def test_cross_source_nonce_key_root_provider_and_mutation(self):
        f=source_fixture(4)
        reference=native.native_single_inverse(f)
        ctx=trusted(f,4,"compact-v1",nonce="1"*64)
        wire=envelope.make_signed_envelope(reference,ctx)
        variants=[
           trusted(f,4,"compact-v1",nonce="2"*64),
           trusted(f,4,"compact-v1",nonce="1"*64,source_sha="a"*64),
           trusted(f,4,"compact-v1",nonce="1"*64,root="0"*64),
           trusted(f,4,"compact-v1",nonce="1"*64,session="another-session"),
           envelope.make_context(
             source=f,source_sha256=ctx.source_sha256,stride=4,
             expected_root_sha256=ctx.expected_root_sha256,
             payload_format="compact-v1",session_id=ctx.session_id,
             sequence=1,nonce=ctx.nonce,provider_id="forged-provider"
           ),
           envelope.make_context(
             source=f,source_sha256=ctx.source_sha256,stride=4,
             expected_root_sha256=ctx.expected_root_sha256,
             payload_format="compact-v1",session_id=ctx.session_id,
             sequence=1,nonce=ctx.nonce,
             public_key=Ed25519PrivateKey.from_private_bytes(b"z"*32).public_key()
           )
        ]
        for other in variants:
            with self.subTest(other=other.session_id),self.assertRaises((ValueError,TypeError)):
                envelope.verify_signed_envelope(wire,other,envelope.ReplayGuard())
        other=list(f);other[4]=-3
        modified=envelope.make_context(
            source=other,source_sha256=ctx.source_sha256,stride=4,
            expected_root_sha256=ctx.expected_root_sha256,
            payload_format="compact-v1",session_id=ctx.session_id,
            sequence=1,nonce=ctx.nonce
        )
        with self.assertRaises((ValueError,TypeError)):
            envelope.verify_signed_envelope(wire,modified,envelope.ReplayGuard())
        for bad in (
            wire[:-1],wire[:-65]+wire[-64:],
            b"WRNG"+wire[4:],
            wire[:4]+(2**32-1).to_bytes(4,"big")+wire[8:],
            wire[:-64]+b"\x00"*64,
            wire+b"\x00"
        ):
            with self.assertRaises((ValueError,TypeError)):
                envelope.verify_signed_envelope(bad,ctx,envelope.ReplayGuard())
        # Known invalid exact inverse is still signed by a test fixture key;
        # signature alone does not imply an honest inverse.
        bad_inverse=list(reference);bad_inverse[8]+=1
        bad_wire=envelope.make_signed_envelope(tuple(bad_inverse),ctx)
        with self.assertRaises((ValueError,TypeError)):
            envelope.verify_signed_envelope(bad_wire,ctx,envelope.ReplayGuard())
        for bad_stride in (0,2,3,7,True,"4"):
            with self.assertRaises((ValueError,TypeError)):
                envelope.make_context(
                    source=f,source_sha256=ctx.source_sha256,stride=bad_stride,
                    expected_root_sha256=ctx.expected_root_sha256,
                    payload_format="compact-v1",session_id="foo",sequence=1)

    def test_bad_source_and_witness_caps(self):
        f=source_fixture(4)
        w=native.native_single_inverse(f)
        c=trusted(f,4,"compact-v1")
        changed=list(w);changed[1]=1
        with self.assertRaises((ValueError,TypeError)):
            envelope.make_signed_envelope(changed,c)
        changed=list(f);changed[1]=1
        with self.assertRaises((ValueError,TypeError)):
            trusted(changed,4,"compact-v1")
        with self.assertRaises((ValueError,TypeError)):
            codec.encode_full_supported((1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),1)
        with self.assertRaises((ValueError,TypeError)):
            codec.encode_full_supported((1,True),1)
        with self.assertRaises((ValueError,TypeError)):
            envelope.make_signed_envelope(w,c,signer="not-an-ed25519-key")

    def test_historical_archive_and_source_only_registration_contract(self):
        from research.nor54_matched_frontier import original_archive_audit
        self.assertEqual(original_archive_audit.audit()["verdict"],"FAIL (4/5)")
        self.assertEqual(len(cohort.earlier_source_ids()),228)
        self.assertEqual(cohort.REGISTERED,
                         "9f7fb42a2e0a8648db63301bed381cb7f3aa0dc2")
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)
        self.assertEqual(envelope.WIRE_MAGIC,b"N55E")

if __name__=="__main__":unittest.main()
