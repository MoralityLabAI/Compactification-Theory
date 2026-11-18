"""NOR-46 historical-only exact trace algebra and signature negative controls.

No new NOR46 sources. Signed contexts and input programs are reconstructed
solely from the original NOR42/NOR45 SHA-archived historical source rows.
"""
from __future__ import annotations
import copy
import json
import random
import unittest
from hashlib import sha256
from itertools import product

from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import program_authority as authority
from research.nor45_authorized_program import historical_smoke as history
from research.nor45_authorized_program import independent_execution_verifier as old
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor46_exact_integer_trace import trace_verifier as verifier


class ExactTraceChecks(unittest.TestCase):
    def test_carry_free_convolution_is_exact_with_signed_huge_integers(self):
        rng=random.Random(22623)
        for n in (1,2,8,17,32,79):
            for bound in (4,1<<4097):
                a=tuple(rng.randint(-bound,bound) for _ in range(n))
                b=tuple(rng.randint(-bound,bound) for _ in range(n))
                direct=tuple(sum(a[j]*b[i-j] for j in range(i+1))
                             for i in range(n))
                self.assertEqual(verifier.exact_convolution_prefix(a,b),direct)

    def test_reciprocal_identity_and_tamper(self):
        rng=random.Random(544)
        for n in (5,64,257,769):
            query=(1,)+tuple(rng.randint(-2,2) for _ in range(n-1))
            inv=old.unary(query,"R")
            verifier.validate_inverse(query,inv)
            tampered=list(inv);tampered[-1]+=1
            with self.assertRaises(verifier.TraceError):
                verifier.validate_inverse(query,tuple(tampered))
        with self.assertRaises(verifier.TraceError):
            verifier.validate_inverse((2,1,0),(1,0,0))

    def test_decimal_canonicalization_and_resource_cap(self):
        for value in ("-0","+1","01","00","1.0","1e3","",1):
            with self.assertRaises(verifier.TraceError):
                verifier._integer(value)
        self.assertEqual(verifier._integer(str(-(2**2048))),-(2**2048))
        with self.assertRaises(verifier.TraceError):
            verifier._integer("9"*9867)

    def test_historical_full_signed_hidden_program_and_adversarial_controls(self):
        fixture,grade=history.load_original()
        row=fixture["cases"][0]
        original=grade["source_records"][0]
        self.assertEqual(row["full_public_input_sha256"],original["public_sha256"])
        public=nor42.nor37.public_proposal(row)
        clean=public["pred"]
        query=tuple(row["public_query_input"])
        d=row["degree"]
        _,hidden,beta,transform=nor42.restore_worlds(row,clean)
        spec=history.spec_from_public_proposal(public,d,beta,transform)
        provider_key=trees.test_private_key()
        pinned_provider=trees.test_reference_public_key()
        vkey=session.test_verifier_private_key()
        vpinned=session.public_key_bytes(vkey)
        akey=authority.test_authority_private_key()
        apinned=authority.public_key_of(akey)
        sess=session.VerifierSession(
            source_sha256=row["full_public_input_sha256"],
            observed_degree=d,coefficient_count=len(clean),
            provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=pinned_provider,
            verifier_signing_key=vkey,expires_epoch=100,
            nonce=sha256(b"NOR46 archived-only fixed source session").digest(),
            test_only=True)
        receipt=sess.precommit(
            session.proposal_digest(clean,sess.context),now_epoch=1)
        hidden_tree=trees.OutputTree(hidden)
        reference=session.sign_provider_reference(
            hidden_tree,receipt,sess.context,vpinned,provider_key,now_epoch=2)
        certificate=authority.sign_authorized_program(
            program=spec,query=query,context=sess.context,
            precommit_receipt=receipt,pinned_verifier_public_key=vpinned,
            expected_output_merkle_root=hidden_tree.root.hex(),
            program_authority_signer=akey,now_epoch=2)
        trace=producer.transcript(spec,query,d)
        accepted=verifier.verify_authorized_trace(
            certificate=certificate,trace=trace,query=query,
            session_context=sess.context,precommit_receipt=receipt,
            provider_reference=reference,
            pinned_program_authority_key=apinned,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=vpinned,
            pinned_provider_public_key=pinned_provider,now_epoch=2)
        full=old.verify_authorized_execution(
            certificate=certificate,query=query,
            session_context=sess.context,precommit_receipt=receipt,
            provider_reference=reference,
            pinned_program_authority_key=apinned,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=vpinned,
            pinned_provider_public_key=pinned_provider,now_epoch=2)
        self.assertEqual(accepted["values"],full["values"])
        self.assertEqual(accepted["values"],hidden)
        self.assertEqual(accepted["root_sha256"],full["root_sha256"])
        self.assertEqual(accepted["root_sha256"],hidden_tree.root.hex())
        self.assertTrue(accepted["trace_bytes"]>0)
        args=dict(certificate=certificate,trace=trace,query=query,
            session_context=sess.context,precommit_receipt=receipt,
            provider_reference=reference,
            pinned_program_authority_key=apinned,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=vpinned,
            pinned_provider_public_key=pinned_provider,now_epoch=2)
        corrupted=copy.deepcopy(trace)
        corrupted["output_coefficients"][d+32]=str(
            int(corrupted["output_coefficients"][d+32])+1)
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,"trace":corrupted})
        corrupted=copy.deepcopy(trace)
        op=next(iter(corrupted["unary_transforms"]))
        corrupted["unary_transforms"][op][-1]=str(
            int(corrupted["unary_transforms"][op][-1])+1)
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,"trace":corrupted})
        wrong_cert=copy.deepcopy(certificate)
        wrong_cert["statement"]["program"]["delay"]=112 if spec["delay"]!=112 else 64
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,"certificate":wrong_cert})
        bad_query=list(query);bad_query[100]+=1
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,"query":bad_query})
        bad_key=sha256(b"model self signing key").digest()
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,
                "pinned_program_authority_key":bad_key})
        bad_context=copy.deepcopy(sess.context)
        bad_context["nonce_hex"]="0"*64
        with self.assertRaises(ValueError):
            verifier.verify_authorized_trace(**{**args,
                "session_context":bad_context})
        # A dishonest authorized clean program executes faithfully and
        # still misses externally designated hidden truth.
        clean_spec=history.spec_from_public_proposal(public,d)
        clean_tree=trees.OutputTree(clean)
        false_reference=session.sign_provider_reference(
            clean_tree,receipt,sess.context,vpinned,provider_key,now_epoch=2)
        false_cert=authority.sign_authorized_program(
            program=clean_spec,query=query,context=sess.context,
            precommit_receipt=receipt,pinned_verifier_public_key=vpinned,
            expected_output_merkle_root=clean_tree.root.hex(),
            program_authority_signer=akey,now_epoch=2)
        false_trace=producer.transcript(clean_spec,query,d)
        false_accepted=verifier.verify_authorized_trace(**{**args,
            "certificate":false_cert,"trace":false_trace,
            "provider_reference":false_reference})
        self.assertEqual(false_accepted["values"],clean)
        self.assertNotEqual(false_accepted["values"],hidden)

    def test_full_signed_native_program_with_R_trace(self):
        source=(1,)+tuple((i%5)-2 for i in range(1,769))
        program={
            "base":"R","delay":64,"spacing":24,"amplitude":-1,
            "u":"R","v":"S","hidden_beta":1,"hidden_transform":"N"}
        witness=producer.transcript(program,source,512)
        self.assertEqual(verifier.verified_trace_values(
            witness,source,program,512),
            old.execute_authorized_program(program,source,512))
        corrupted=copy.deepcopy(witness)
        corrupted["unary_transforms"]["R"][500]=str(
            int(corrupted["unary_transforms"]["R"][500])+1)
        with self.assertRaises(verifier.TraceError):
            verifier.verified_trace_values(corrupted,source,program,512)
        damaged=copy.deepcopy(witness)
        damaged["output_coefficients"][-1]=str(
            int(damaged["output_coefficients"][-1])-4)
        with self.assertRaises(verifier.TraceError):
            verifier.verified_trace_values(damaged,source,program,512)


if __name__=="__main__":
    unittest.main()
