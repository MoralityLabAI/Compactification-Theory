"""NOR59 source-free actual NOR45 signed authority/support/root regression tests.

Uses toy public exact inputs only. No prospective NOR59 source identity is
generated here. Original NOR45 and NOR44 signature, precommit, query digest,
provider signature, full independent root and signed grammar are exercised.
All private signing seeds are public deterministic TEST fixtures.
"""
from __future__ import annotations
import copy,hashlib,itertools,random,unittest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor45_authorized_program import program_authority as authority
from research.nor44_session_protocol import session_protocol as session
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor58_static_merkle import template
from research.nor59_authorized_merkle import authorized_root

def program(profile=0,hidden=True):
    entries=(
        {"base":"I","u":"R","v":"N","delay":64,"spacing":24,
         "amplitude":1,"hidden_beta":0,"hidden_transform":None},
        {"base":"N","u":"S","v":"I","delay":80,"spacing":32,
         "amplitude":-2,"hidden_beta":1 if hidden else 0,
         "hidden_transform":"R" if hidden else None},
        {"base":"R","u":"I","v":"S","delay":96,"spacing":24,
         "amplitude":2,"hidden_beta":-1 if hidden else 0,
         "hidden_transform":"S" if hidden else None},
        {"base":"S","u":"N","v":"R","delay":112,"spacing":32,
         "amplitude":-1,"hidden_beta":0,"hidden_transform":None},
        {"base":"R","u":"R","v":"R","delay":80,"spacing":24,
         "amplitude":1,"hidden_beta":2 if hidden else 0,
         "hidden_transform":"N" if hidden else None}
    )
    return dict(entries[profile])

def source(degree:int=512,stride:int=4,bits:int=2)->tuple[int,...]:
    m=3*degree//2+1
    vals=[1]+[0]*(m-1)
    for i in range(stride,m,stride):
        vals[i]=(-1 if (i//stride)%2 else 1)*(1<<(bits-1))
    return tuple(vals)

def signed_bundle(query,profile=1,degree=512,nonce=b"\x1a"*32,
                  certificate_program=None,provider_output=None):
    p=program(profile) if certificate_program is None else certificate_program
    result=nor45.execute_authorized_program(p,query,degree)
    clean=query
    output=result if provider_output is None else provider_output
    provider=trees.test_private_key()
    verifier=session.test_verifier_private_key()
    authority_signer=authority.test_authority_private_key()
    provider_pub=trees.test_reference_public_key()
    verifier_pub=session.public_key_bytes(verifier)
    authority_pub=authority.public_key_of(authority_signer)
    src=hashlib.sha256(
        b"NOR59/TESTONLY/"+bytes([profile])+degree.to_bytes(2,"big")+
        hashlib.sha256(nor45.query_digest(query).encode()).digest()).hexdigest()
    sess=session.VerifierSession(
        source_sha256=src,observed_degree=degree,
        coefficient_count=len(query),provider_id=trees.PROVIDER_ID,
        pinned_provider_public_key=provider_pub,
        verifier_signing_key=verifier,expires_epoch=100,
        nonce=nonce,test_only=True)
    receipt=sess.precommit(
        session.proposal_digest(clean,sess.context),now_epoch=1)
    output_tree=trees.OutputTree(output)
    provider_ref=session.sign_provider_reference(
        output_tree,receipt,sess.context,verifier_pub,provider,now_epoch=2)
    certificate=authority.sign_authorized_program(
        program=p,query=query,context=sess.context,
        precommit_receipt=receipt,pinned_verifier_public_key=verifier_pub,
        expected_output_merkle_root=nor45.independently_computed_root(result),
        program_authority_signer=authority_signer,now_epoch=2)
    kwargs={
        "certificate":certificate,"query":query,
        "session_context":sess.context,
        "precommit_receipt":receipt,
        "provider_reference":provider_ref,
        "pinned_program_authority_key":authority_pub,
        "expected_program_authority_id":authority.TEST_AUTHORITY_ID,
        "pinned_verifier_public_key":verifier_pub,
        "pinned_provider_public_key":provider_pub,
        "now_epoch":2
    }
    return kwargs,result

class GenuineNOR45AuthorizationTests(unittest.TestCase):
    def test_grammar_support_proof_obligations_all_offsets_and_ops(self):
        closure=authorized_root.source_free_closure_explanation()
        self.assertTrue(closure["all_allowed_delay_and_spacing_sums_divisible_by_8"])
        self.assertTrue(closure["all_hidden_suffix_onsets_divisible_by_8"])
        self.assertFalse(closure["formal_Lean_kernel_proof_available"])
        self.assertEqual(closure["NOR45_ops"],["I","N","R","S"])
        self.assertEqual(tuple(closure["support_strides"]),(1,4,8))
        for degree,stride,profile in itertools.product((512,1024),(1,4,8),range(5)):
            f=source(degree,stride,2)
            p=program(profile)
            self.assertEqual(authorized_root.trusted_public_support_stride(f,degree),stride)
            nor45._validate_program(p,degree,len(f))
            self.assertEqual(authorized_root.certify_actual_NOR45_program_support(
                program=p,query=f,observed_degree=degree,
                output=nor45.execute_authorized_program(p,f,degree)),stride)
        modified=program(1)
        modified["delay"]=65
        with self.assertRaises(ValueError):
            authorized_root.certify_actual_NOR45_program_support(
                program=modified,query=source(),observed_degree=512,
                output=source())
        modified=program(1);modified["spacing"]=25
        with self.assertRaises(ValueError):
            authorized_root.certify_actual_NOR45_program_support(
                program=modified,query=source(),observed_degree=512,
                output=source())
        f=source()
        values=list(nor45.execute_authorized_program(program(1),f,512))
        values[1]=1
        with self.assertRaises(ValueError):
            authorized_root.certify_actual_NOR45_program_support(
                program=program(1),query=f,observed_degree=512,
                output=values)

    def test_immutable_templates_are_exact_full_NOR45_roots(self):
        for d,s,profile in ((512,1,0),(512,4,1),(512,8,2),
                            (1024,4,3),(1024,8,4)):
            f=source(d,s,2)
            p=program(profile)
            values=nor45.execute_authorized_program(p,f,d)
            original=nor45.independently_computed_root(values)
            for policy in ("warm","cold"):
                self.assertEqual(authorized_root.authorized_root(
                    program=p,query=f,observed_degree=d,output=values,
                    mode=policy),original)
            t=authorized_root.warm_geometry_only_template(len(f),s)
            self.assertEqual(t.root(values),original)
            self.assertEqual(t.template_sha256,
                             template.build_template(len(f),s).template_sha256)
        with self.assertRaises(ValueError):
            authorized_root.warm_geometry_only_template(3073,4)
        with self.assertRaises(ValueError):
            authorized_root.warm_geometry_only_template(769,True)
        with self.assertRaises(ValueError):
            authorized_root.authorized_root(
                program=program(0),query=source(),observed_degree=512,
                output=source(),mode="unsafe_callback")

    def test_real_signed_NOR45_verifier_default_warm_cold_parity(self):
        for s,p in ((1,0),(4,1),(8,2),(4,3),(8,4)):
            f=source(512,s,2)
            kwargs,reference=signed_bundle(f,profile=p)
            baseline=nor45.verify_authorized_execution(**kwargs)
            self.assertEqual(baseline["values"],reference)
            for mode in ("warm","cold"):
                result=nor45.verify_authorized_execution(
                    **kwargs,_root_policy=mode)
                self.assertEqual(result,baseline)
            with self.assertRaises(nor45.ProgramAttestationError):
                nor45.verify_authorized_execution(
                    **kwargs,_root_policy="untrusted-root-function")

    def test_reject_provider_forgery_authority_key_source_and_session(self):
        f=source(512,8,2)
        kwargs,reference=signed_bundle(f,profile=1)
        baseline=nor45.verify_authorized_execution(**kwargs)
        for mode in ("original","warm","cold"):
            self.assertEqual(
                nor45.verify_authorized_execution(**kwargs,_root_policy=mode),
                baseline)
        badf=list(f);badf[1]=17
        badctx=dict(kwargs)
        badctx["query"]=tuple(badf)
        for mode in ("original","warm","cold"):
            with self.assertRaises((ValueError,TypeError)):
                nor45.verify_authorized_execution(**badctx,_root_policy=mode)
        for field,payload in (
            ("pinned_program_authority_key",b"0"*32),
            ("pinned_provider_public_key",b"1"*32),
            ("pinned_verifier_public_key",b"2"*32),
        ):
            altered=dict(kwargs,**{field:payload})
            for mode in ("original","warm","cold"):
                with self.assertRaises((ValueError,TypeError)):
                    nor45.verify_authorized_execution(
                        **altered,_root_policy=mode)
        bad_certificate=copy.deepcopy(kwargs["certificate"])
        bad_certificate["signature_hex"]="00"*64
        for mode in ("original","warm","cold"):
            with self.assertRaises((ValueError,TypeError)):
                nor45.verify_authorized_execution(
                    **dict(kwargs,certificate=bad_certificate),_root_policy=mode)
        bad_context=copy.deepcopy(kwargs["session_context"])
        bad_context["nonce_hex"]="ff"*32
        for mode in ("original","warm","cold"):
            with self.assertRaises((ValueError,TypeError)):
                nor45.verify_authorized_execution(
                    **dict(kwargs,session_context=bad_context),_root_policy=mode)
        false_out=source(512,8,2)
        falsified,_=signed_bundle(
            f,profile=1,nonce=b"\x1b"*32,provider_output=false_out)
        for mode in ("original","warm","cold"):
            with self.assertRaises((ValueError,TypeError)):
                nor45.verify_authorized_execution(
                    **falsified,_root_policy=mode)

    def test_clean_authenticated_certificate_can_pass_semantically_wrong_authority(self):
        f=source(512,4,2)
        honest,_=signed_bundle(f,profile=1)
        result=nor45.verify_authorized_execution(
            **honest,_root_policy="warm")
        other=signed_bundle(f,profile=0,nonce=b"\x1c"*32)
        separately_authorized=nor45.verify_authorized_execution(
            **other[0],_root_policy="warm")
        self.assertNotEqual(result["values"],separately_authorized["values"])
        # Signature validation gives authority-selected execution consistency,
        # not a proof that the authority selected an honest world program.

if __name__=="__main__":
    unittest.main()
