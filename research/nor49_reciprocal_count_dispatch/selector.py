"""NOR-49 prospectively frozen active reciprocal count >=2 dispatcher.

Training source: NOR48 original first 16 timed signed native programs.
This closed threshold is frozen before measuring 48 heldout source programs.
All operations come exclusively from the previously authenticated native
program statement, never the provided trace, model output or backend hint.
"""
from __future__ import annotations
from typing import Sequence
from research.nor48_program_dispatch import dispatcher as prior
from research.nor45_authorized_program import independent_execution_verifier as native

THRESHOLD=2

def reciprocal_count(program:dict,degree:int,m:int)->int:
    native._validate_program(program,degree,m)
    operations=[program["base"],program["u"],program["v"]]
    if program["hidden_beta"]!=0:
        operations.append(program["hidden_transform"])
    return sum(op=="R" for op in operations)

def authorized_backend(program:dict,degree:int,m:int)->str:
    return "gmp_full_trace" if reciprocal_count(program,degree,m)>=THRESHOLD else "nor45_direct"

def verify_math(program:dict,query:Sequence[int],degree:int,
                trace:dict|None=None,untrusted_hint:str|None=None)->dict:
    """Full exact output and root; signed program authentication is upstream."""
    count=reciprocal_count(program,degree,len(query))
    route="gmp_full_trace" if count>=THRESHOLD else "nor45_direct"
    if route=="gmp_full_trace":
        if trace is None:raise prior.DispatchError("GMP branch requires exact trace")
        values=prior.verified_gmp_trace_values(trace,query,program,degree)
    else:
        # A no-R or single-R signed program executes by original NOR45 direct,
        # without inspecting untrusted trace contents.
        values=native.execute_authorized_program(program,query,degree)
    return {
        "authorized_active_R_count":count,
        "route":route,
        "output":values,
        "merkle_root":native.independently_computed_root(values)
    }

def verify_signed_count_dispatch(*,certificate,untrusted_trace,query,
    session_context,precommit_receipt,provider_reference,
    pinned_program_authority_key,expected_program_authority_id,
    pinned_verifier_public_key,pinned_provider_public_key,
    now_epoch=2,untrusted_backend_hint=None):
    """Pin all signatures/session/source/root through existing NOR48 validator.

    This wrapper retains its exact signed authentication and post-execution
    provider-root check while substituting the frozen R-count rule.
    A scoped, per-call delegation does not modify NOR48 or NOR46 globals.
    """
    from hashlib import sha256
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from research.nor44_session_protocol import session_protocol as session
    from research.nor46_exact_integer_trace import trace_verifier as nor46
    if not isinstance(certificate,dict) or set(certificate)!={"statement","signature_hex"}:
        raise prior.DispatchError("invalid signed authority envelope")
    stmt=certificate["statement"]
    required={"protocol","semantics","authority_id","authority_key_id",
      "context_sha256","session_nonce_hex","source_sha256","observed_degree",
      "coefficient_count","provider_key_id","precommit_receipt_sha256",
      "query_sha256","expected_merkle_root_sha256","program"}
    if not isinstance(stmt,dict) or set(stmt)!=required:
        raise prior.DispatchError("noncanonical program statement")
    if (stmt["protocol"]!=native.PROTOCOL or
        stmt["semantics"]!=native.CODE_VERSION or
        stmt["authority_id"]!=expected_program_authority_id or
        type(pinned_program_authority_key) is not bytes or
        len(pinned_program_authority_key)!=32 or
        stmt["authority_key_id"]!=sha256(pinned_program_authority_key).hexdigest()):
        raise prior.DispatchError("unknown signed program authority")
    try:
        sig=bytes.fromhex(certificate["signature_hex"])
        if len(sig)!=64 or sig.hex()!=certificate["signature_hex"]:
            raise prior.DispatchError("noncanonical program signature")
        Ed25519PublicKey.from_public_bytes(pinned_program_authority_key).verify(
            sig,nor46._canonical(stmt))
    except (InvalidSignature,ValueError,TypeError) as exc:
        raise prior.DispatchError("unauthorized signed program") from exc
    session.verify_precommit_receipt(
        precommit_receipt,pinned_verifier_public_key,session_context,now_epoch)
    authorized=session.verify_provider_reference(
        provider_reference,session_context,precommit_receipt,
        pinned_provider_public_key,pinned_verifier_public_key,now_epoch)
    expected={
        "context_sha256":sha256(session.canonical(session_context)).hexdigest(),
        "session_nonce_hex":session_context["nonce_hex"],
        "source_sha256":session_context["source_sha256"],
        "observed_degree":session_context["observed_degree"],
        "coefficient_count":session_context["coefficient_count"],
        "provider_key_id":session_context["provider_key_id"],
        "precommit_receipt_sha256":sha256(
            session.canonical(precommit_receipt)).hexdigest(),
        "query_sha256":native.query_digest(query)}
    if any(stmt[k]!=v for k,v in expected.items()):
        raise prior.DispatchError("signed statement replayed across context")
    selected=verify_math(stmt["program"],query,session_context["observed_degree"],
                         untrusted_trace,untrusted_backend_hint)
    root=selected["merkle_root"]
    if (root!=stmt["expected_merkle_root_sha256"] or
        root!=authorized["merkle_root_sha256"]):
        raise prior.DispatchError("signed provider or program root mismatch")
    return {
        "backend":selected["route"],
        "R_count":selected["authorized_active_R_count"],
        "source_sha256":session_context["source_sha256"],
        "root_sha256":root,
        "values":selected["output"],
        "program":dict(stmt["program"]),
        "authority_id":expected_program_authority_id,
        "mode":"count-R>=2 signed full exact integer verifier, nonsuccinct"
    }
