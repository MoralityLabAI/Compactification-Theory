"""NOR49 pre-timing tests use only ORIGINAL NOR48 16 timed TRAINING programs.

Never call NOR49 first 48-source benchmark or inspect new holdout timing data.
"""
from __future__ import annotations
import copy,unittest
from collections import Counter
from research.nor48_program_dispatch import test_dispatch as original_tests
from research.nor48_program_dispatch import dispatcher as original
from research.nor46_exact_integer_trace import trace_verifier as previous
from research.nor49_reciprocal_count_dispatch import selector as new


class ProspectiveSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_tests.NOR48Dispatch.setUpClass()
        cls.training=original_tests.NOR48Dispatch.examples
        assert len(cls.training)==16

    def test_frozen_count_rule_on_only_training_16(self):
        groups=Counter()
        for program,query,d,trace,target in self.training:
            count=new.reciprocal_count(program,d,len(query))
            groups[count]+=1
            predicted=new.authorized_backend(program,d,len(query))
            self.assertEqual(predicted,
                "gmp_full_trace" if count>=2 else "nor45_direct")
            for hint in (None,"gmp_full_trace","nor45_direct","attacker-override"):
                result=new.verify_math(program,query,d,trace,hint)
                self.assertEqual(result["route"],predicted)
                self.assertEqual(result["output"],target)
                self.assertEqual(result["merkle_root"],
                                 original.native.independently_computed_root(target))
        self.assertEqual(groups,{0:4,1:5,2:6,3:1})
        self.assertEqual(new.THRESHOLD,2)

    def test_source_only_R_one_remains_direct_even_with_poisoned_trace(self):
        program,query,d,trace,target=next(
            case for case in self.training if
            new.reciprocal_count(case[0],case[2],len(case[1]))==1)
        poisoned=copy.deepcopy(trace)
        poisoned["unary_transforms"]["R"][-1]="0000"
        result=new.verify_math(program,query,d,poisoned,"gmp_full_trace")
        self.assertEqual(result["route"],"nor45_direct")
        self.assertEqual(result["output"],target)
        modified=dict(program)
        modified["spacing"]=1
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(modified,query,d,trace)

    def test_two_or_more_R_reject_malicious_trace_and_preserve_global(self):
        program,query,d,trace,target=next(
            case for case in self.training if
            new.reciprocal_count(case[0],case[2],len(case[1]))>=2)
        saved=previous.exact_convolution_prefix
        self.assertEqual(new.verify_math(program,query,d,trace)["output"],target)
        corrupted=copy.deepcopy(trace)
        corrupted["unary_transforms"]["R"][-1]=str(
            int(corrupted["unary_transforms"]["R"][-1])+1)
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(program,query,d,corrupted)
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(program,query,d,None)
        self.assertIs(previous.exact_convolution_prefix,saved)


    def test_pinned_signed_program_and_provider_bound_to_threshold_route(self):
        from hashlib import sha256
        from research.nor42_paired_program_ood.prospective import experiment as nor42
        from research.nor43_authenticated_witness import signed_witness as trees
        from research.nor44_session_protocol import session_protocol as session
        from research.nor45_authorized_program import program_authority as authority
        from research.nor45_authorized_program import historical_smoke as historical
        fixture,grade=original_tests.old.nor45.load_original()
        rows={r["full_public_input_sha256"]:r for r in fixture["cases"]}
        pin_provider=trees.test_reference_public_key()
        pin_verifier=session.public_key_bytes(session.test_verifier_private_key())
        signer=authority.test_authority_private_key()
        pin_authority=authority.public_key_of(signer)
        for expected_R_count in (0,1,2):
            program,query,d,trace,target=next(
                case for case in self.training if
                new.reciprocal_count(case[0],case[2],len(case[1]))==expected_R_count)
            row=next(r for r in rows.values() if
                     r["public_query_input"]==list(query) and r["degree"]==d)
            sha=row["full_public_input_sha256"]
            clean=nor42.nor37.public_proposal(row)["pred"]
            sess=session.VerifierSession(
                source_sha256=sha,observed_degree=d,
                coefficient_count=len(query),provider_id=trees.PROVIDER_ID,
                pinned_provider_public_key=pin_provider,
                verifier_signing_key=session.test_verifier_private_key(),
                expires_epoch=100,
                nonce=sha256(("NOR49/TEST-SIGNED-SESSION/"+sha).encode()).digest(),
                test_only=True)
            pre=sess.precommit(session.proposal_digest(clean,sess.context),now_epoch=1)
            tree=trees.OutputTree(target)
            ref=session.sign_provider_reference(
                tree,pre,sess.context,pin_verifier,trees.test_private_key(),now_epoch=2)
            certificate=authority.sign_authorized_program(
                program=program,query=query,context=sess.context,
                precommit_receipt=pre,pinned_verifier_public_key=pin_verifier,
                expected_output_merkle_root=tree.root.hex(),
                program_authority_signer=signer,now_epoch=2)
            args=dict(
                certificate=certificate,untrusted_trace=trace,query=query,
                session_context=sess.context,precommit_receipt=pre,
                provider_reference=ref,pinned_program_authority_key=pin_authority,
                expected_program_authority_id=authority.TEST_AUTHORITY_ID,
                pinned_verifier_public_key=pin_verifier,
                pinned_provider_public_key=pin_provider,now_epoch=2)
            verified=new.verify_signed_count_dispatch(**args)
            self.assertEqual(verified["R_count"],expected_R_count)
            expected="gmp_full_trace" if expected_R_count>=2 else "nor45_direct"
            self.assertEqual(verified["backend"],expected)
            self.assertEqual(verified["values"],target)
            self.assertEqual(verified["root_sha256"],tree.root.hex())
            for hint in ("nor45_direct","gmp_full_trace","MODEL-PICKED"):
                self.assertEqual(new.verify_signed_count_dispatch(
                    **args,untrusted_backend_hint=hint)["backend"],expected)
            invalid=copy.deepcopy(certificate)
            invalid["signature_hex"]="00"*64
            with self.assertRaises((ValueError,TypeError)):
                new.verify_signed_count_dispatch(**{**args,"certificate":invalid})
            with self.assertRaises((ValueError,TypeError)):
                new.verify_signed_count_dispatch(**{
                    **args,"expected_program_authority_id":"FORGED-AUTHORITY"})
            fraud=session.sign_provider_reference(
                trees.OutputTree(clean),pre,sess.context,pin_verifier,
                trees.test_private_key(),now_epoch=2)
            if clean!=target:
                with self.assertRaises((ValueError,TypeError)):
                    new.verify_signed_count_dispatch(**{
                        **args,"provider_reference":fraud})
            poisoned=copy.deepcopy(trace)
            if expected_R_count>=2:
                poisoned["unary_transforms"]["R"][-1]=str(
                    int(poisoned["unary_transforms"]["R"][-1])+1)
                with self.assertRaises((ValueError,TypeError)):
                    new.verify_signed_count_dispatch(**{
                        **args,"untrusted_trace":poisoned})
            else:
                poisoned["query_sha256"]="00"*32
                self.assertEqual(new.verify_signed_count_dispatch(
                    **{**args,"untrusted_trace":poisoned})["values"],target)


if __name__=="__main__":
    unittest.main()
