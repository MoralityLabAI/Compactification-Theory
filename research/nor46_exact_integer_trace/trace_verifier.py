"""NOR-46 independent transparent exact-BigInt execution trace verifier.

Checks every supplied unary transform and output coefficient. R is verified
using an exact carry-free Kronecker convolution, rather than recomputing the
recursive inverse. Cryptographic signatures/source/session roots are checked
against NOR44/45's independently pinned authorities. Full trace is revealed;
this is deterministic, NON-SUCCINCT, NON-ZK, and not a STARK/SNARK.
"""
from __future__ import annotations
import json
from hashlib import sha256
from typing import Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import independent_execution_verifier as nor45

PROTOCOL="NOR46/full-exact-integer-trace/v1"
MAX_COEFFICIENT_BITS=32768
MAX_PACKED_BITS=120_000_000

class TraceError(ValueError):
    pass


def _canonical(obj)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def _integer(s):
    if type(s) is not str or not s or len(s)>12000:
        raise TraceError("coefficient must be finite canonical decimal text")
    try:
        n=int(s)
    except (ValueError,OverflowError) as exc:
        raise TraceError("invalid integer coefficient") from exc
    if str(n)!=s or abs(n).bit_length()>MAX_COEFFICIENT_BITS:
        raise TraceError("noncanonical or overlarge integer coefficient")
    return n


def exact_convolution_prefix(a:Sequence[int],b:Sequence[int])->tuple[int,...]:
    """One carry-free full-width integer multiply, no field reduction.

    Shift both signed input arrays by offset K=2^maxbits to nonnegative
    digits. Every coefficient of their product is < 4*m*K^2. A base
    2^width strictly exceeding that bound prevents cross-digit carries.
    Subtract cross terms and the offset^2 term to recover exact integers.
    """
    n=len(a)
    if not n or len(b)!=n:
        raise TraceError("convolution vectors need identical nonzero length")
    if any(type(x) is not int for x in a) or any(type(x) is not int for x in b):
        raise TraceError("convolution inputs must be exact integers")
    maxbits=max(max(abs(x).bit_length() for x in a),
                max(abs(x).bit_length() for x in b))
    if maxbits>MAX_COEFFICIENT_BITS:
        raise TraceError("unbounded trace coefficient")
    offset=1<<maxbits
    width=(4*n*offset*offset).bit_length()+1
    if n*width>MAX_PACKED_BITS:
        raise TraceError("packed integer exceeds verifier resource cap")
    whole_a=0
    whole_b=0
    for x,y in zip(reversed(a),reversed(b)):
        whole_a=(whole_a<<width)+(x+offset)
        whole_b=(whole_b<<width)+(y+offset)
    product=whole_a*whole_b
    mask=(1<<width)-1
    prefix_a=prefix_b=0
    result=[]
    for i in range(n):
        prefix_a+=a[i]
        prefix_b+=b[i]
        digit=product&mask
        result.append(digit-offset*(prefix_a+prefix_b)-(i+1)*offset*offset)
        product>>=width
    return tuple(result)


def validate_inverse(source:Sequence[int],claimed:Sequence[int])->None:
    """Certify every degree-<m inverse coefficient, including signed overflow."""
    if len(source)!=len(claimed) or not source or source[0]!=1:
        raise TraceError("inverse requires normalized finite input")
    if claimed[0]!=1:
        raise TraceError("inverse constant coefficient must equal one")
    product=exact_convolution_prefix(source,claimed)
    if product[0]!=1 or any(coef!=0 for coef in product[1:]):
        raise TraceError("inverse trace fails exact integer convolution")


def _check_op(source:tuple[int,...],op:str,transformed:tuple[int,...])->None:
    if len(source)!=len(transformed):
        raise TraceError("transform length mismatch")
    if transformed[0]!=1:
        raise TraceError("transform must preserve unit normalization")
    if op=="R":
        validate_inverse(source,transformed)
    elif op=="I":
        if any(v!=s for v,s in zip(transformed,source)):
            raise TraceError("incorrect identity transform")
    elif op=="N":
        if any(v!=(-s if i%2 else s)
               for i,(s,v) in enumerate(zip(source,transformed))):
            raise TraceError("incorrect sign/substitution transform")
    elif op=="S":
        if any(v!=(source[i//2] if i%2==0 else 0)
               for i,v in enumerate(transformed)):
            raise TraceError("incorrect sparse q-square transform")
    else:
        raise TraceError("unsupported unary transform")


def verified_trace_values(trace:dict,query:Sequence[int],
                          program:dict,observed_degree:int)->tuple[int,...]:
    """Validate the full trace independently of NOR45's native evaluator."""
    source=tuple(query)
    n=len(source)
    nor45._validate_program(program,observed_degree,n)
    if not source or source[0]!=1 or any(type(x) is not int for x in source):
        raise TraceError("source must contain normalized exact integers")
    if not isinstance(trace,dict) or set(trace)!={
        "protocol","query_sha256","coefficient_count","observed_degree",
        "unary_transforms","output_coefficients"}:
        raise TraceError("untrusted proof envelope schema")
    if trace["protocol"]!=PROTOCOL or trace["query_sha256"]!=nor45.query_digest(source):
        raise TraceError("execution proof source/protocol mismatch")
    if (type(trace["coefficient_count"]) is not int or
        trace["coefficient_count"]!=n or
        type(trace["observed_degree"]) is not int or
        trace["observed_degree"]!=observed_degree):
        raise TraceError("trace length or observation mismatch")
    required={program["base"],program["u"],program["v"]}
    if program["hidden_beta"]:
        required.add(program["hidden_transform"])
    transforms=trace["unary_transforms"]
    if not isinstance(transforms,dict) or set(transforms)!=required:
        raise TraceError("missing or extra unary execution trace")
    validated={}
    for op in sorted(required):
        series=transforms[op]
        if not isinstance(series,list) or len(series)!=n:
            raise TraceError("invalid unary witness length")
        witness=tuple(_integer(s) for s in series)
        _check_op(source,op,witness)
        validated[op]=witness
    outputs=trace["output_coefficients"]
    if not isinstance(outputs,list) or len(outputs)!=n:
        raise TraceError("invalid output witness length")
    claimed=tuple(_integer(s) for s in outputs)
    base=validated[program["base"]]
    left=validated[program["u"]]
    right=validated[program["v"]]
    beta=program["hidden_beta"]
    hidden=validated.get(program["hidden_transform"]) if beta else None
    delay=program["delay"]
    spacing=program["spacing"]
    amp=program["amplitude"]
    onset=observed_degree+32
    for i,have in enumerate(claimed):
        want=base[i]
        if i==delay:want+=amp
        if i>=delay+spacing:want+=amp*left[i-delay-spacing]
        if i>=delay+2*spacing:want+=amp*right[i-delay-2*spacing]
        if beta and i>=onset:want+=beta*hidden[i-onset]
        if have!=want:
            raise TraceError(f"incorrect executed output coefficient {i}")
    return claimed


def verify_authorized_trace(*,certificate:dict,trace:dict,query:Sequence[int],
    session_context:dict,precommit_receipt:dict,provider_reference:dict,
    pinned_program_authority_key:bytes,expected_program_authority_id:str,
    pinned_verifier_public_key:bytes,pinned_provider_public_key:bytes,
    now_epoch:int=2)->dict:
    """Authenticate the external program/session/root then verify all trace.

    No call to NOR45 execute_authorized_program or NOR45.unary occurs here.
    Authenticity of the authorized program remains an external premise.
    """
    if not isinstance(certificate,dict) or set(certificate)!={
        "statement","signature_hex"}:
        raise TraceError("invalid program attestation envelope")
    statement=certificate["statement"]
    required={"protocol","semantics","authority_id","authority_key_id",
        "context_sha256","session_nonce_hex","source_sha256",
        "observed_degree","coefficient_count","provider_key_id",
        "precommit_receipt_sha256","query_sha256",
        "expected_merkle_root_sha256","program"}
    if not isinstance(statement,dict) or set(statement)!=required:
        raise TraceError("noncanonical program statement")
    if (statement["protocol"]!=nor45.PROTOCOL or
        statement["semantics"]!=nor45.CODE_VERSION or
        statement["authority_id"]!=expected_program_authority_id):
        raise TraceError("wrong authority or exact-execution semantics")
    if (type(pinned_program_authority_key) is not bytes or
        len(pinned_program_authority_key)!=32 or
        statement["authority_key_id"]!=sha256(
            pinned_program_authority_key).hexdigest()):
        raise TraceError("program authority key not independently pinned")
    try:
        sig=bytes.fromhex(certificate["signature_hex"])
        if len(sig)!=64 or certificate["signature_hex"]!=sig.hex():
            raise TraceError("noncanonical authority signature")
        Ed25519PublicKey.from_public_bytes(
            pinned_program_authority_key).verify(sig,_canonical(statement))
    except (ValueError,TypeError,InvalidSignature) as exc:
        raise TraceError("invalid independent program authority signature") from exc
    session.verify_precommit_receipt(
        precommit_receipt,pinned_verifier_public_key,session_context,now_epoch)
    authorized_provider=session.verify_provider_reference(
        provider_reference,session_context,precommit_receipt,
        pinned_provider_public_key,pinned_verifier_public_key,now_epoch)
    required_context={
        "context_sha256":sha256(session.canonical(session_context)).hexdigest(),
        "session_nonce_hex":session_context["nonce_hex"],
        "source_sha256":session_context["source_sha256"],
        "observed_degree":session_context["observed_degree"],
        "coefficient_count":session_context["coefficient_count"],
        "provider_key_id":session_context["provider_key_id"],
        "precommit_receipt_sha256":sha256(
            session.canonical(precommit_receipt)).hexdigest(),
        "query_sha256":nor45.query_digest(query)
    }
    if any(statement[name]!=value for name,value in required_context.items()):
        raise TraceError("signed program was rebound to another source/session")
    program=statement["program"]
    values=verified_trace_values(
        trace,query,program,session_context["observed_degree"])
    output_root=nor45.independently_computed_root(values)
    if (statement["expected_merkle_root_sha256"]!=output_root or
        authorized_provider["merkle_root_sha256"]!=output_root):
        raise TraceError("trace output disagrees with signed program/provider root")
    return {
        "source_sha256":session_context["source_sha256"],
        "root_sha256":output_root,
        "program":dict(program),
        "values":values,
        "trace_sha256":sha256(_canonical(trace)).hexdigest(),
        "trace_bytes":len(_canonical(trace)),
        "authority_id":expected_program_authority_id,
        "mode":"full exact-integer trace, non-succinct"
    }
