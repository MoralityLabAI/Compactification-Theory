"""NOR-48 signed-program R-aware exact execution dispatch, arithmetic core.

Backend selection uses the separately authorized native program ONLY.
This reentrant GMP trace verifier passes convolution explicitly, never
reassigns nor46 globals. The signing/session/provider-root validations
are tested separately with the original NOR45/46 authority harness;
this routine alone is not a signed-certificate verifier.
"""
from __future__ import annotations
from typing import Sequence
from hashlib import sha256

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from research.nor44_session_protocol import session_protocol as session

from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as trace
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp

class DispatchError(trace.TraceError):
    pass

R_FIELDS=("base","u","v")

def authorized_routing_key(program:dict,observed_degree:int,m:int)->str:
    """Closed exact program is the sole routing input, never an unsigned hint."""
    native._validate_program(program,observed_degree,m)
    ops=(program[k] for k in R_FIELDS)
    used="R" in ops
    if program["hidden_beta"]:
        used=used or program["hidden_transform"]=="R"
    return "gmp_full_trace" if used else "nor45_direct"

def _checked_transform(source:tuple[int,...],op:str,witness:tuple[int,...],
                       convolve)->None:
    if len(source)!=len(witness) or witness[0]!=1:
        raise DispatchError("wrong transform length or unit")
    if op=="R":
        product=convolve(source,witness)
        if len(product)!=len(source) or product[0]!=1 or any(
                x!=0 for x in product[1:]):
            raise DispatchError("invalid signed exact inverse witness")
    else:
        trace._check_op(source,op,witness)

def verified_gmp_trace_values(witness:dict,query:Sequence[int],program:dict,
                              observed_degree:int,convolve=gmp.full_gmp)->tuple[int,...]:
    """Verify original full NOR46 exact-integer proof with injected backend."""
    source=tuple(query)
    n=len(source)
    native._validate_program(program,observed_degree,n)
    if not source or source[0]!=1 or any(type(v) is not int for v in source):
        raise DispatchError("source must be exact normalized integer series")
    if not isinstance(witness,dict) or set(witness)!={
        "protocol","query_sha256","coefficient_count","observed_degree",
        "unary_transforms","output_coefficients"}:
        raise DispatchError("noncanonical trace schema")
    if witness["protocol"]!=trace.PROTOCOL or witness["query_sha256"]!=native.query_digest(source):
        raise DispatchError("wrong trace protocol or query")
    if (type(witness["coefficient_count"]) is not int or
        witness["coefficient_count"]!=n or
        type(witness["observed_degree"]) is not int or
        witness["observed_degree"]!=observed_degree):
        raise DispatchError("trace degree/length mismatch")
    needed={program["base"],program["u"],program["v"]}
    if program["hidden_beta"]:needed.add(program["hidden_transform"])
    raw=witness["unary_transforms"]
    if not isinstance(raw,dict) or set(raw)!=needed:
        raise DispatchError("missing or excess trace transform")
    transforms={}
    for op in sorted(needed):
        values=raw[op]
        if not isinstance(values,list) or len(values)!=n:
            raise DispatchError("incorrect transform array length")
        coefficients=tuple(trace._integer(v) for v in values)
        _checked_transform(source,op,coefficients,convolve)
        transforms[op]=coefficients
    output=witness["output_coefficients"]
    if not isinstance(output,list) or len(output)!=n:
        raise DispatchError("incorrect output array length")
    claimed=tuple(trace._integer(v) for v in output)
    base=transforms[program["base"]]
    u=transforms[program["u"]]
    v=transforms[program["v"]]
    beta=program["hidden_beta"]
    hidden=transforms[program["hidden_transform"]] if beta else None
    k,d,c=program["delay"],program["spacing"],program["amplitude"]
    onset=observed_degree+32
    for i,have in enumerate(claimed):
        want=base[i]
        if i==k:want+=c
        if i>=k+d:want+=c*u[i-k-d]
        if i>=k+2*d:want+=c*v[i-k-2*d]
        if beta and i>=onset:want+=beta*hidden[i-onset]
        if want!=have:
            raise DispatchError(f"wrong exact output at coefficient {i}")
    return claimed

def dispatch_verified_arithmetic(program:dict,query:Sequence[int],degree:int,
                                 untrusted_trace:dict|None=None,
                                 unsigned_backend_hint:str|None=None):
    """Compute full future and root. The unsigned backend hint is ignored.

    Program must originate from a previously independently authenticated
    signed statement; authentication is outside this core benchmark.
    """
    route=authorized_routing_key(program,degree,len(query))
    if route=="nor45_direct":
        values=native.execute_authorized_program(program,query,degree)
    else:
        if untrusted_trace is None:
            raise DispatchError("R program requires full exact trace witness")
        values=verified_gmp_trace_values(untrusted_trace,query,program,degree)
    root=native.independently_computed_root(values)
    return {"route":route,"output":values,"merkle_root":root}


def verify_authorized_dispatch(*,certificate:dict,untrusted_trace:dict|None,
    query:Sequence[int],session_context:dict,precommit_receipt:dict,
    provider_reference:dict,pinned_program_authority_key:bytes,
    expected_program_authority_id:str,pinned_verifier_public_key:bytes,
    pinned_provider_public_key:bytes,now_epoch:int=2,
    unsigned_backend_hint:str|None=None)->dict:
    """End-to-end signed statement/session/provider/root verification.

    Route ONLY from signed program after independently verifying its
    authority signature and binding to the precommitted verifier session.
    This path performs no global verifier modification, and any untrusted
    trace is ignored for a signed no-R program. Provider honesty remains
    a separate trust premise.
    """
    if not isinstance(certificate,dict) or set(certificate)!={"statement","signature_hex"}:
        raise DispatchError("invalid signed program envelope")
    statement=certificate["statement"]
    fields={"protocol","semantics","authority_id","authority_key_id",
            "context_sha256","session_nonce_hex","source_sha256",
            "observed_degree","coefficient_count","provider_key_id",
            "precommit_receipt_sha256","query_sha256",
            "expected_merkle_root_sha256","program"}
    if not isinstance(statement,dict) or set(statement)!=fields:
        raise DispatchError("incorrect authorized program fields")
    if (statement["protocol"]!=native.PROTOCOL or
        statement["semantics"]!=native.CODE_VERSION or
        statement["authority_id"]!=expected_program_authority_id):
        raise DispatchError("wrong program authority or semantics")
    if (type(pinned_program_authority_key) is not bytes or
        len(pinned_program_authority_key)!=32 or
        statement["authority_key_id"]!=sha256(
            pinned_program_authority_key).hexdigest()):
        raise DispatchError("program authority key not pinned")
    try:
        signature=bytes.fromhex(certificate["signature_hex"])
        if len(signature)!=64 or certificate["signature_hex"]!=signature.hex():
            raise DispatchError("noncanonical program authority signature")
        Ed25519PublicKey.from_public_bytes(
            pinned_program_authority_key).verify(signature,trace._canonical(statement))
    except (InvalidSignature,ValueError,TypeError) as exc:
        raise DispatchError("invalid program authority signature") from exc
    session.verify_precommit_receipt(
        precommit_receipt,pinned_verifier_public_key,session_context,now_epoch)
    reference=session.verify_provider_reference(
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
        "query_sha256":native.query_digest(query)
    }
    if any(statement[name]!=value for name,value in expected.items()):
        raise DispatchError("signed program rebound to another source/session")
    program=statement["program"]
    calculated=dispatch_verified_arithmetic(
        program,query,session_context["observed_degree"],untrusted_trace,
        unsigned_backend_hint=unsigned_backend_hint)
    root=calculated["merkle_root"]
    if (root!=statement["expected_merkle_root_sha256"] or
        root!=reference["merkle_root_sha256"]):
        raise DispatchError("computed output differs from authorized provider root")
    return {
        "source_sha256":session_context["source_sha256"],
        "root_sha256":root,
        "program":dict(program),
        "values":calculated["output"],
        "backend":calculated["route"],
        "authority_id":expected_program_authority_id,
        "mode":"signed-source-bound full exact execution, nonsuccinct"
    }
