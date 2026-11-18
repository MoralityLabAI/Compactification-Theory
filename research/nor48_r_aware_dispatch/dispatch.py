"""NOR-48 historical-only signed-program R-aware verifier dispatcher.

Select backend from the independently SIGNED original NOR45 program
statement, after verifying the pinned authority signature. The no-R branch
executes NOR45's independent full program. R branches use the exact NOR47
GMP trace through NOR46's original full signed program/session/root checks.

The GMP path retains a benchmark-only single-thread global override:
NEVER deploy this wrapper concurrently as a production API.
"""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as trace
from research.nor47_gmp_exact_convolution import gmp_kernel


class DispatchError(ValueError):
    pass


def authenticated_requires_reciprocal(*,certificate,pinned_authority_public_key,
                                      expected_authority_id,query,observed_degree):
    """Early signed-statement verification, without executing the program.

    Downstream selected full NOR45/NOR46 verifier still independently
    checks source SHA, session nonce, proposal receipt, provider root,
    source-query digest and every output coefficient.
    """
    if not isinstance(certificate,dict) or set(certificate)!={"statement","signature_hex"}:
        raise DispatchError("untrusted signed-program envelope")
    statement=certificate["statement"]
    keys={"protocol","semantics","authority_id","authority_key_id",
          "context_sha256","session_nonce_hex","source_sha256",
          "observed_degree","coefficient_count","provider_key_id",
          "precommit_receipt_sha256","query_sha256",
          "expected_merkle_root_sha256","program"}
    if not isinstance(statement,dict) or set(statement)!=keys:
        raise DispatchError("signed program has unexpected schema")
    if (statement["protocol"]!=native.PROTOCOL or
        statement["semantics"]!=native.CODE_VERSION or
        statement["authority_id"]!=expected_authority_id):
        raise DispatchError("incorrect signed exact-execution semantics or authority")
    if type(pinned_authority_public_key) is not bytes or len(pinned_authority_public_key)!=32:
        raise DispatchError("source of authority public key is untrusted")
    if statement["authority_key_id"]!=sha256(pinned_authority_public_key).hexdigest():
        raise DispatchError("authority signature key is not pinned")
    try:
        raw_signature=bytes.fromhex(certificate["signature_hex"])
        if len(raw_signature)!=64 or raw_signature.hex()!=certificate["signature_hex"]:
            raise DispatchError("noncanonical signature")
        Ed25519PublicKey.from_public_bytes(pinned_authority_public_key).verify(
            raw_signature,native._canonical(statement))
    except (ValueError,InvalidSignature,TypeError) as exc:
        raise DispatchError("invalid signed native program authority") from exc
    if (statement["observed_degree"]!=observed_degree or
        statement["coefficient_count"]!=len(query) or
        statement["query_sha256"]!=native.query_digest(query)):
        raise DispatchError("signed program does not match query and observation degree")
    program=statement["program"]
    native._validate_program(program,observed_degree,len(query))
    used={program["base"],program["u"],program["v"]}
    if program["hidden_beta"]:
        used.add(program["hidden_transform"])
    return "R" in used


@contextmanager
def temporary_exact_gmp_backend():
    """Single-thread CPU benchmark ONLY; no shared production state."""
    before=trace.exact_convolution_prefix
    trace.exact_convolution_prefix=gmp_kernel.full_gmp
    try:yield
    finally:trace.exact_convolution_prefix=before


def verified_historical_dispatch(*,certificate,trace_witness,query,
    session_context,precommit_receipt,provider_reference,
    pinned_program_authority_key,expected_program_authority_id,
    pinned_verifier_public_key,pinned_provider_public_key,now_epoch=2):
    """Verify signed source context and execute path chosen by signed program.

    The caller has no backend selector. The model-supplied witness is
    irrelevant to the no-R path. Every R proof is checked in its entirety.
    """
    use_gmp=authenticated_requires_reciprocal(
        certificate=certificate,
        pinned_authority_public_key=pinned_program_authority_key,
        expected_authority_id=expected_program_authority_id,
        query=query,
        observed_degree=session_context["observed_degree"])
    shared={
        "certificate":certificate,
        "query":query,
        "session_context":session_context,
        "precommit_receipt":precommit_receipt,
        "provider_reference":provider_reference,
        "pinned_program_authority_key":pinned_program_authority_key,
        "expected_program_authority_id":expected_program_authority_id,
        "pinned_verifier_public_key":pinned_verifier_public_key,
        "pinned_provider_public_key":pinned_provider_public_key,
        "now_epoch":now_epoch
    }
    if use_gmp:
        with temporary_exact_gmp_backend():
            result=trace.verify_authorized_trace(
                trace=trace_witness,**shared)
        backend="GMP exact full trace / authorized program contains R"
    else:
        result=native.verify_authorized_execution(**shared)
        backend="NOR45 exact direct execution / no authorized R"
    return {
        "backend":backend,
        "requires_R":use_gmp,
        "values":result["values"],
        "root_sha256":result["root_sha256"],
        "source_sha256":result["source_sha256"],
        "authority_id":result["authority_id"],
        "signed_program_sha256":sha256(
            native._canonical(certificate["statement"])).hexdigest(),
        "security_scope":"Signed-root/authorized-program fidelity only; "
                         "provider/program semantic honesty external"
    }
