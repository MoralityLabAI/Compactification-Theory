"""NOR58 source-free indexed static Merkle parity and signed hostile tests.

Uses tiny toy inputs and old immutable original NOR57B archives only.
It NEVER generates fresh NOR58 public identities or new benchmark timings.
"""
from __future__ import annotations
import itertools,random,unittest
from hashlib import sha256
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as prior
from research.nor52_sparse_inverse import algebra as support
from research.nor55_compact_witness import envelope as signed
from research.nor56_batch_codec import signed_v2
from research.nor58_static_merkle import cohort,template,verify

def toy_source(stride:int,m:int=41):
    f=[1]+[0]*(m-1)
    for i in range(stride,m,stride):
        f[i]=(-2 if (i//stride)%2 else 3)
    return tuple(f)

def trusted(source,stride,ref,seq=1,nonce="a"*64,source_sha=None,
            session_id="nor58-source-free-replay"):
    return signed_v2.make_context_v2(
        source=source,
        source_sha256=source_sha or sha256(b"NOR58/TOY/SOURCE").hexdigest(),
        stride=stride,expected_root_sha256=nor45.independently_computed_root(ref),
        session_id=session_id,sequence=seq,nonce=nonce)

class StaticMerkleTests(unittest.TestCase):
    def test_exhaustive_small_tree_and_full_root_byte_identity(self):
        for m in (1,2,3,4,5,7,8,9,15,17,33):
            for stride in (1,4,8):
                t=template.build_template(m,stride)
                self.assertEqual(t.dynamic_leaf_indices,tuple(range(0,m,stride)))
                self.assertEqual(t.static_zero_leaf_count,
                                 m-len(range(0,m,stride)))
                self.assertEqual(t.static_padding_leaf_count,t.size-m)
                self.assertGreater(t.canonical_template_bytes,0)
                self.assertEqual(len(t.template_sha256),64)
                self.assertEqual(t,template.build_template(m,stride))
                rng=random.Random(5800+m*100+stride)
                for _ in range(4):
                    values=[1]+[0]*(m-1)
                    for i in range(stride,m,stride):
                        values[i]=rng.choice((-1,0,1,2,-4))
                    values=tuple(values)
                    self.assertEqual(t.root(values),
                                     nor45.independently_computed_root(values))
                    for i in range(t.size):
                        if i>=m:
                            expected=sha256(nor45.PAD+nor45._word(m)+nor45._word(i)).digest()
                            self.assertEqual(t.static_nodes[t.size+i],expected)
                        elif i%stride:
                            expected=sha256(
                                nor45.LEAF+nor45._word(m)+nor45._word(i)+
                                nor45._integer_bytes(0)).digest()
                            self.assertEqual(t.static_nodes[t.size+i],expected)
                        else:
                            self.assertIsNone(t.static_nodes[t.size+i])

    def test_toy_large_and_signed_independent_root_parity(self):
        for m,stride in ((769,1),(769,4),(1537,8),(3073,4),(3073,8)):
            t=template.build_template(m,stride)
            w=[1]+[0]*(m-1)
            w[stride]=-(1<<4096)
            w[2*stride]=1<<32767
            self.assertEqual(t.root(tuple(w)),
                             nor45.independently_computed_root(tuple(w)))
            self.assertNotEqual(t.static_nodes[t.size+1],
                                t.static_nodes[t.size+3]) if stride>1 else None
            self.assertLess(t.canonical_template_bytes,250_000)
        with self.assertRaises((ValueError,TypeError)):
            template.build_template(0,4)
        with self.assertRaises((ValueError,TypeError)):
            template.build_template(4097,4)
        with self.assertRaises((ValueError,TypeError)):
            template.build_template(769,True)
        with self.assertRaises((ValueError,TypeError)):
            template.build_template(769,2)
        t=template.build_template(41,4)
        a=list(toy_source(4))
        a[1]=1
        with self.assertRaises((ValueError,TypeError)):
            t.root(a)
        with self.assertRaises((ValueError,TypeError)):
            t.root(tuple(a[:-1]))
        with self.assertRaises((ValueError,TypeError)):
            t.root((True,*a[1:]))

    def test_all_authenticated_signed_verifier_modes_match_original(self):
        prev_global=nor46.exact_convolution_prefix
        for stride in (1,4,8):
            source=toy_source(stride,49)
            ref=prior.native_single_inverse(source)
            root=nor45.independently_computed_root(ref)
            t=template.build_template(len(source),stride)
            guards={mode:signed.ReplayGuard() for mode in verify.MODES}
            original=signed.ReplayGuard()
            for rep in (1,2,3):
                ctx=trusted(source,stride,ref,seq=rep,nonce=f"{rep:064x}",
                            session_id=f"nor58-toy-{stride}")
                wire=signed_v2.make_signed_envelope_v2(ref,ctx)
                self.assertEqual(signed_v2.verify_signed_envelope_v2(
                    wire,ctx,original),(ref,root))
                for mode in verify.MODES:
                    self.assertEqual(verify.verify_same_signed_witness(
                        wire,ctx,guards[mode],mode,t if mode==verify.WARM else None),
                        (ref,root))
                    with self.assertRaises((TypeError,ValueError)):
                        verify.verify_same_signed_witness(
                            wire,ctx,guards[mode],mode,
                            t if mode==verify.WARM else None)
        self.assertIs(prev_global,nor46.exact_convolution_prefix)

    def test_all_modes_reject_signed_tampering_support_key_nonce_and_forged_root(self):
        f=toy_source(4,41)
        w=prior.native_single_inverse(f)
        t=template.build_template(len(f),4)
        ctx=trusted(f,4,w)
        good=signed_v2.make_signed_envelope_v2(w,ctx)
        bad=list(w);bad[8]+=1
        bad_identity=signed_v2.make_signed_envelope_v2(tuple(bad),ctx)
        for mode in verify.MODES:
            def attempt(wire,context=ctx,tt=t):
                return verify.verify_same_signed_witness(
                    wire,context,signed.ReplayGuard(),mode,
                    tt if mode==verify.WARM else None)
            with self.assertRaises((TypeError,ValueError)):
                attempt(bad_identity)
            with self.assertRaises((TypeError,ValueError)):
                attempt(good[:-1]+bytes([good[-1]^1]))
            with self.assertRaises((TypeError,ValueError)):
                attempt(good+b"\x00")
            changed=trusted(f,4,w,nonce="b"*64)
            with self.assertRaises((TypeError,ValueError)):
                attempt(good,changed)
            other=trusted(f,4,w,source_sha="b"*64)
            with self.assertRaises((TypeError,ValueError)):
                attempt(good,other)
            wrongkey=signed_v2.make_context_v2(
                source=f,source_sha256=ctx.source_sha256,stride=4,
                expected_root_sha256=ctx.expected_root_sha256,
                session_id=ctx.session_id,sequence=1,nonce=ctx.nonce,
                public_key=Ed25519PrivateKey.from_private_bytes(
                    b"z"*32).public_key())
            with self.assertRaises((TypeError,ValueError)):
                attempt(good,wrongkey)
            self.assertEqual(attempt(good),(w,nor45.independently_computed_root(w)))
        for wrong_template in (template.build_template(42,4),
                               template.build_template(41,1)):
            with self.assertRaises((ValueError,TypeError)):
                verify.verify_same_signed_witness(
                    good,ctx,signed.ReplayGuard(),verify.WARM,wrong_template)
        tampered_source=list(f);tampered_source[1]=2
        with self.assertRaises((ValueError,TypeError)):
            signed_v2.make_context_v2(
                source=tampered_source,source_sha256=ctx.source_sha256,stride=4,
                expected_root_sha256=ctx.expected_root_sha256,
                session_id="fraud",sequence=1)
        with self.assertRaises((ValueError,TypeError)):
            support.checked_full_witness((1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),
                                         (1,0),1)

    def test_preregistered_468_historical_source_and_original_failures(self):
        self.assertEqual(cohort.REGISTERED,
                         "ddbabc02cf8702fe2da2c9962174ddcfcbed39a4")
        self.assertEqual(cohort.SOURCE_DOMAIN,"NOR58/SOURCE/v1")
        self.assertEqual(len(cohort.historical_ids()),468)
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)
        from research.nor57b_independent import original_archive_audit
        self.assertEqual(original_archive_audit.read_original()["first_verdict"],
                         "FAIL (4/5)")

if __name__=="__main__":
    unittest.main()
