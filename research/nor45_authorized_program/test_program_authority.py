"""NOR45 synthetic authorization, execution and false-program tests.

All signing keys are publicly reproducible TEST fixtures. These are
reference-protocol tests, not a security theorem or prospective source data.
"""
from __future__ import annotations
import copy,hashlib,random,unittest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.rrc_compactification import compound25 as corpus
from research.nor44_session_protocol import session_protocol as session
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor43_authenticated_witness import verifier_reference as openings
from research.nor45_authorized_program import program_authority as authority
from research.nor45_authorized_program import independent_execution_verifier as verifier


class AuthorizedProgramTests(unittest.TestCase):
    def setUp(self):
        self.source=(1,)+tuple((j%7)-3 for j in range(1,769))
        self.source_sha=hashlib.sha256(b"NOR45/SYNTHETIC-ONLY-SOURCE").hexdigest()
        self.clean={
            "base":"I","delay":64,"spacing":24,"amplitude":1,
            "u":"N","v":"S","hidden_beta":0,"hidden_transform":None}
        self.hidden={**self.clean,"hidden_beta":-2,"hidden_transform":"R"}
        self.trustee=authority.test_authority_private_key()
        self.pinned_program=authority.public_key_of(self.trustee)
        self.provider=trees.test_private_key()
        self.provider_pub=trees.test_reference_public_key()
        self.verifier=session.test_verifier_private_key()
        self.clean_output=verifier.execute_authorized_program(
            self.clean,self.source,512)
        self.hidden_output=verifier.execute_authorized_program(
            self.hidden,self.source,512)
        self.clean_tree=trees.OutputTree(self.clean_output)
        self.hidden_tree=trees.OutputTree(self.hidden_output)
        self.session=session.VerifierSession(
            source_sha256=self.source_sha,observed_degree=512,
            coefficient_count=len(self.source),provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=self.provider_pub,
            verifier_signing_key=self.verifier,
            expires_epoch=100,nonce=b"\x19"*32,test_only=True)
        self.context=self.session.context
        self.verifier_pub=session.public_key_bytes(self.verifier)
        self.receipt=self.session.precommit(session.proposal_digest(
            self.clean_output,self.context),now_epoch=1)
        self.reference=self.provider_ref(self.hidden_tree)

    def provider_ref(self,tree):
        return session.sign_provider_reference(
            tree,self.receipt,self.context,self.verifier_pub,
            self.provider,now_epoch=2)

    def certificate(self,program=None,tree=None,signer=None,query=None):
        return authority.sign_authorized_program(
            program=self.hidden if program is None else program,
            query=self.source if query is None else query,
            context=self.context,precommit_receipt=self.receipt,
            pinned_verifier_public_key=self.verifier_pub,
            expected_output_merkle_root=(
                self.hidden_tree if tree is None else tree).root.hex(),
            program_authority_signer=self.trustee if signer is None else signer,
            now_epoch=2)

    def verify(self,cert,reference=None,query=None,session_context=None,
               pinned_authority=None):
        return verifier.verify_authorized_execution(
            certificate=cert,
            query=self.source if query is None else query,
            session_context=self.context if session_context is None
                else session_context,
            precommit_receipt=self.receipt,
            provider_reference=self.reference if reference is None else reference,
            pinned_program_authority_key=(
                self.pinned_program if pinned_authority is None
                else pinned_authority),
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=self.verifier_pub,
            pinned_provider_public_key=self.provider_pub,
            now_epoch=2)

    def test_independent_native_program_and_merkle_root(self):
        program=(
            self.clean["base"],self.clean["delay"],self.clean["spacing"],
            self.clean["amplitude"],self.clean["u"],self.clean["v"])
        actual=corpus.execute(self.source,program)
        self.assertEqual(self.clean_output,actual)
        self.assertEqual(self.hidden_output[:513],actual[:513])
        self.assertEqual(self.hidden_output[544]-actual[544],-2)
        for t,op in ((0,"I"),(1,"N"),(2,"R"),(3,"S")):
            native=corpus.d.operate(self.source,op)
            independent=verifier.unary(self.source,op)
            self.assertEqual(native,independent)
        self.assertEqual(verifier.independently_computed_root(
            self.clean_output),self.clean_tree.root.hex())
        self.assertEqual(verifier.independently_computed_root(
            self.hidden_output),self.hidden_tree.root.hex())

    def test_separate_authority_true_hidden_program_rejected_by_sparse_read(self):
        verified=self.verify(self.certificate())
        self.assertEqual(verified["values"],self.hidden_output)
        self.assertEqual(verified["root_sha256"],self.hidden_tree.root.hex())
        self.session.authorize_reference(self.reference,now_epoch=2)
        selected=(544,)+tuple(range(15))
        self.session.challenge(now_epoch=3,test_indices=selected)
        terminal=self.session.finish(self.clean_output,
            [self.hidden_tree.opening(i) for i in selected],now_epoch=3)
        result=session.verify_terminal_result(terminal,self.context,
            self.verifier_pub)
        self.assertTrue(result["reject"])
        self.assertEqual(self.session.phase,"finalized")

    def test_output_provider_signs_wrong_root_under_honest_program(self):
        cert=self.certificate()
        forged_reference=self.provider_ref(self.clean_tree)
        # NOR44 reference signature is valid under pinned provider.
        self.assertEqual(session.verify_provider_reference(
            forged_reference,self.context,self.receipt,self.provider_pub,
            self.verifier_pub,now_epoch=2)["merkle_root_sha256"],
            self.clean_tree.root.hex())
        # NOR45 independent execution catches the dishonest output provider.
        with self.assertRaises(verifier.ProgramAttestationError):
            self.verify(cert,reference=forged_reference)

    def test_model_self_signed_false_program_rejected_by_pinned_authority(self):
        model_signer=Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"PUBLIC-NOR45-MODEL-KEY").digest())
        wrong=self.certificate(program=self.clean,tree=self.clean_tree,
                               signer=model_signer)
        bad_reference=self.provider_ref(self.clean_tree)
        with self.assertRaises(verifier.ProgramAttestationError):
            self.verify(wrong,reference=bad_reference)
        # A verifier that pins the ATTACKER's key is insecure, and can
        # validate an exact but semantically false clean execution.
        wrong_key=authority.public_key_of(model_signer)
        verified=self.verify(wrong,reference=bad_reference,
                             pinned_authority=wrong_key)
        self.assertEqual(verified["values"],self.clean_output)
        self.assertNotEqual(verified["values"],self.hidden_output)

    def test_dishonest_authorized_program_can_be_faithfully_executed(self):
        wrong_program=self.certificate(program=self.clean,tree=self.clean_tree)
        matching_reference=self.provider_ref(self.clean_tree)
        verified=self.verify(wrong_program,reference=matching_reference)
        self.assertEqual(verified["values"],self.clean_output)
        self.assertNotEqual(verified["values"],self.hidden_output)
        # A valid program signature + full deterministic execution proves
        # consistency with the AUTHORIZED program, not semantic world truth.

    def test_wrong_program_root_query_session_context_and_signature(self):
        cert=self.certificate()
        changed=self.certificate(program=self.clean,tree=self.hidden_tree)
        with self.assertRaises(verifier.ProgramAttestationError):
            self.verify(changed)
        altered=list(self.source);altered[700]+=1
        with self.assertRaises(verifier.ProgramAttestationError):
            self.verify(cert,query=altered)
        context=dict(self.context)
        context["nonce_hex"]="00"*32
        with self.assertRaises((verifier.ProgramAttestationError,session.SessionError)):
            self.verify(cert,session_context=context)
        for k,v in (
            ("semantics","NOR45/unknown/other"),
            ("source_sha256","f"*64),
            ("observed_degree",513),
            ("coefficient_count",1537),
            ("query_sha256","f"*64),
            ("expected_merkle_root_sha256","0"*64),
            ("provider_key_id","9"*64),
            ("precommit_receipt_sha256","0"*64),
            ("authority_id","ANOTHER-AUTHORITY")):
            modified=copy.deepcopy(cert)
            modified["statement"][k]=v
            with self.assertRaises(verifier.ProgramAttestationError):
                self.verify(modified)
        tampered=copy.deepcopy(cert)
        tampered["signature_hex"]="00"*64
        with self.assertRaises(verifier.ProgramAttestationError):
            self.verify(tampered)

    def test_bad_program_grammar_and_integer_encoding_rejected(self):
        for change in (
            {"hidden_beta":3},{"hidden_transform":"BOGUS"},
            {"delay":30},{"spacing":99},{"amplitude":0},
            {"base":"NOT-OP"},{"u":"Q"},{"v":"Q"},
            {"hidden_beta":0},{"delay":True}):
            modified={**self.hidden,**change}
            with self.assertRaises(ValueError):
                authority.validate_program_spec(modified,512)
        with self.assertRaises(TypeError):
            authority.query_digest((1,1.2))
        with self.assertRaises(verifier.ProgramAttestationError):
            verifier.execute_authorized_program(
                self.hidden,(1,)+tuple([1]*767),512)
        with self.assertRaises(TypeError):
            authority.sign_authorized_program(
                program=self.hidden,query=self.source,context=self.context,
                precommit_receipt=self.receipt,
                pinned_verifier_public_key=self.verifier_pub,
                expected_output_merkle_root=self.hidden_tree.root.hex(),
                program_authority_signer=None)


if __name__=="__main__":
    unittest.main()
