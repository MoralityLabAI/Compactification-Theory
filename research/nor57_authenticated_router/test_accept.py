"""NOR57 source-free prior-only/toy identity, selector and signed tamper tests.

Never generates NOR57's new 60 source identities or first CPU timings.
"""
from __future__ import annotations
import unittest
from hashlib import sha256
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as independent
from research.nor52_sparse_inverse import algebra as sparse
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import signed_v2,codec_v2
from research.nor57_authenticated_router import accept,cohort

def toy(stride=4,length=41,bit_width=2):
    out=[1]+[0]*(length-1)
    for i in range(stride,length,stride):
        out[i]=(-1 if (i//stride)%2 else 1)*(1<<(bit_width-1))
    return tuple(out)

def context_for(f,reference,source_sha,sequence=1,nonce="a"*64,
                session_id="NOR57-TOY-SESSION",provider_id=v1.PROVIDER_ID,
                public_key=None,stride=4):
    return signed_v2.make_context_v2(
        source=f,source_sha256=source_sha,stride=stride,
        expected_root_sha256=independent.root(reference),
        session_id=session_id,sequence=sequence,nonce=nonce,
        provider_id=provider_id,public_key=public_key)

class RegisteredSourceFreeNOR57Tests(unittest.TestCase):
    def test_source_only_frozen_rule_assigns_trusted_public_complexity(self):
        self.assertEqual(accept.RULE_ID,"NOR57-FROZEN-PUBLIC-SELECTOR-v1")
        self.assertEqual(accept.MODES,("always_gmp","always_native","frozen_selector"))
        self.assertEqual(accept.select_from_public_source(toy(4,769,16),4),accept.NATIVE)
        self.assertEqual(accept.select_from_public_source(toy(4,1537,16),4),accept.NATIVE)
        self.assertEqual(accept.select_from_public_source(toy(4,3073,16),4),accept.GMP)
        self.assertEqual(accept.select_from_public_source(toy(4,1537,8),4),accept.GMP)
        self.assertEqual(accept.select_from_public_source(toy(8,1537,16),8),accept.GMP)
        self.assertEqual(accept.select_from_public_source(toy(1,769,16),1),accept.GMP)
        for wrong in (0,2,3,7,True,"4"):
            with self.assertRaises((TypeError,ValueError)):
                accept.select_from_public_source(toy(),wrong)
        bad=list(toy());bad[1]=2
        with self.assertRaises((TypeError,ValueError)):
            accept.select_from_public_source(bad,4)
        bad=list(toy());bad[4]=True
        with self.assertRaises((TypeError,ValueError)):
            accept.select_from_public_source(bad,4)

    def test_identical_signed_envelope_exact_vectors_merkle_and_replay(self):
        original_nor46=nor46.exact_convolution_prefix
        samples=[(toy(4,41,2),4),(toy(8,49,2),8),(toy(1,25,2),1),
                 (toy(4,769,16),4),(toy(4,1537,16),4)]
        for f,stride in samples:
            reference=independent.native_single_inverse(f)
            exact=(reference,independent.root(reference))
            source_sha=sha256(f"NOR57/TOY/{stride}/{len(f)}".encode()).hexdigest()
            guards={mode:v1.ReplayGuard() for mode in accept.MODES}
            old_guard=v1.ReplayGuard()
            for seq in range(1,4):
                ctx=context_for(f,reference,source_sha,stride=stride,sequence=seq,
                                nonce=f"{seq:064x}",
                                session_id=f"nor57-toy-{stride}-{len(f)}")
                wire=signed_v2.make_signed_envelope_v2(reference,ctx)
                self.assertEqual(signed_v2.verify_signed_envelope_v2(wire,ctx,old_guard),exact)
                for mode in accept.MODES:
                    self.assertEqual(
                        accept.verify_same_signed_envelope(wire,ctx,guards[mode],mode),
                        exact)
                    with self.assertRaises((TypeError,ValueError)):
                        accept.verify_same_signed_envelope(wire,ctx,guards[mode],mode)
        self.assertIs(nor46.exact_convolution_prefix,original_nor46)

    def test_all_modes_reject_forged_native_supported_witness_and_keep_nonce(self):
        f=toy(4,41,16)
        source_sha=sha256(b"NOR57-TOY-MALICIOUS-WITNESS").hexdigest()
        good=independent.native_single_inverse(f)
        ctx=context_for(f,good,source_sha)
        wire=signed_v2.make_signed_envelope_v2(good,ctx)
        corrupted=list(good);corrupted[8]+=1
        wrong_signed=signed_v2.make_signed_envelope_v2(corrupted,ctx)
        for mode in accept.MODES:
            guard=v1.ReplayGuard()
            with self.assertRaises((TypeError,ValueError)):
                accept.verify_same_signed_envelope(wrong_signed,ctx,guard,mode)
            self.assertEqual(
                accept.verify_same_signed_envelope(wire,ctx,guard,mode),
                (good,independent.root(good)))
        invalid=list(good);invalid[1]=1
        with self.assertRaises((TypeError,ValueError)):
            signed_v2.make_signed_envelope_v2(invalid,ctx)
        with self.assertRaises((TypeError,ValueError)):
            signed_v2.make_signed_envelope_v2(good[:-1],ctx)

    def test_all_routes_share_authenticity_and_original_cap_failures(self):
        f=toy(4,41,2)
        w=independent.native_single_inverse(f)
        sid=sha256(b"NOR57-TOY-CROSS-CONTEXT").hexdigest()
        ctx=context_for(f,w,sid)
        wire=signed_v2.make_signed_envelope_v2(w,ctx)
        alternative_f=list(f);alternative_f[4]=-3
        wrongcontexts=[
            context_for(f,w,"b"*64),
            context_for(f,w,sid,nonce="b"*64),
            context_for(f,w,sid,session_id="rebound-session"),
            context_for(f,w,sid,provider_id="fake-provider"),
            context_for(f,w,sid,public_key=Ed25519PrivateKey.from_private_bytes(
                b"z"*32).public_key()),
            context_for(tuple(alternative_f),
                        independent.native_single_inverse(tuple(alternative_f)),sid)
        ]
        for mode in accept.MODES:
            with self.subTest(mode=mode):
                for badctx in wrongcontexts:
                    with self.assertRaises((TypeError,ValueError)):
                        accept.verify_same_signed_envelope(
                            wire,badctx,v1.ReplayGuard(),mode)
                for corrupted in (wire[:-1]+bytes([wire[-1]^1]),
                                  wire[:-1],wire+b"\x00",b"NOT!"+wire[4:]):
                    with self.assertRaises((TypeError,ValueError)):
                        accept.verify_same_signed_envelope(
                            corrupted,ctx,v1.ReplayGuard(),mode)
                with self.assertRaises((TypeError,ValueError)):
                    accept.verify_same_signed_envelope(wire,ctx,v1.ReplayGuard(),"any")
        with self.assertRaises((TypeError,ValueError)):
            sparse.checked_full_witness(
                (1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),(1,0),1)
        with self.assertRaises((TypeError,ValueError)):
            sparse.checked_full_witness((1,0),(1,True),1)

    def test_post_first_run_reporting_field_regression(self):
        # The first NOR57 CPU job had an unscored KeyError at this boundary.
        # Future labels stay explicitly follow-up; first failure remains archived.
        from research.nor57_authenticated_router import benchmark_cpu
        arms=benchmark_cpu.ARMS
        row={
            "frozen_public_selector_decision":accept.NATIVE,
            "arm_medians_ms":{arms[0]:2.0,arms[1]:1.0,arms[2]:1.1},
            "always_native_over_always_GMP":0.5,
            "frozen_selector_over_always_GMP":0.55,
            "frozen_selector_over_always_native":1.1
        }
        score=benchmark_cpu.summarize([row])
        self.assertEqual(score["selector_native_case_count"],1)
        self.assertEqual(score["selector_gmp_case_count"],0)
        self.assertEqual(score["sum_source_median_ms_by_arm"][arms[2]],1.1)

    def test_preserved_historical_negative_first_and_stated_60_case_grid(self):
        from research.nor56_batch_codec import original_archive_audit
        assert original_archive_audit.audit()["first_verdict"]=="FAIL (3/5)"
        self.assertEqual(cohort.REGISTERED,
                         "f0a6091f918217138d466456c34fcc12ac8f3907")
        self.assertEqual(cohort.SOURCE_DOMAIN,"NOR57/SOURCE/v1")
        self.assertEqual(len(cohort.historical_ids()),348)
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)

if __name__=="__main__":
    unittest.main()
