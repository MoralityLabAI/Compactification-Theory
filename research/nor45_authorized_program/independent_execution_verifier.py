"""NOR45 separately implemented authorized program execution verifier.

Verifies a pinned program-authority Ed25519 signature and the original
verifier-signed NOR44 precommit, then recomputes ALL integral coefficients
without importing either the NOR45 producer or original RRC25 evaluator.

Proof scope: faithful execution of the separately AUTHORIZED program, under
external collision-resistance, signature and honest program-authority premises.
It cannot infer that this program describes actual semantic ground truth.
"""
from __future__ import annotations
import json
from hashlib import sha256
from struct import pack
from typing import Sequence
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from research.nor44_session_protocol import session_protocol as session

PROTOCOL="NOR45/authorized-executable-output/v1"
CODE_VERSION="NOR45/RRC25-exact-series-plus-optional-D32-suffix/v1"
OPS=("I","N","R","S")
DELAYS=(64,80,96,112)
SPACINGS=(24,32)
AMPLITUDES=(-2,-1,1,2)
LEAF=b"NOR43/leaf/v1\x00"
NODE=b"NOR43/node/v1\x00"
PAD=b"NOR43/padding/v1\x00"


class ProgramAttestationError(ValueError):
    pass


def _word(n:int)->bytes:
    if type(n) is not int or not 0<=n<2**32:
        raise ProgramAttestationError("invalid index/length framing")
    return pack(">I",n)


def _integer_bytes(value:int)->bytes:
    if type(value) is not int:
        raise ProgramAttestationError("source or output contains noninteger")
    unsigned=abs(value)
    count=max(1,(unsigned.bit_length()+7)//8)
    return bytes([int(value<0)])+_word(count)+unsigned.to_bytes(count,"big")


def query_digest(query:Sequence[int])->str:
    if not 1<=len(query)<2**32 or query[0]!=1:
        raise ProgramAttestationError("invalid normalized query source")
    h=sha256(b"NOR45/exact-query-vector/v1\x00"+_word(len(query)))
    for i,val in enumerate(query):
        c=_integer_bytes(val)
        h.update(_word(i)+_word(len(c))+c)
    return h.hexdigest()


def _canonical(statement:dict)->bytes:
    return json.dumps(statement,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def _public_key_id(pinned:bytes)->str:
    if not isinstance(pinned,bytes) or len(pinned)!=32:
        raise ProgramAttestationError("program authority key not pinned")
    return sha256(pinned).hexdigest()


def _validate_program(p:dict,d:int,m:int):
    if not isinstance(p,dict) or set(p)!={"base","delay","spacing","amplitude",
                                          "u","v","hidden_beta","hidden_transform"}:
        raise ProgramAttestationError("noncanonical executable program")
    if (type(p["base"]) is not str or p["base"] not in OPS or
        type(p["u"]) is not str or p["u"] not in OPS or
        type(p["v"]) is not str or p["v"] not in OPS):
        raise ProgramAttestationError("unsupported RRC25 operation")
    if any(type(p[k]) is not int for k in (
            "delay","spacing","amplitude","hidden_beta")):
        raise ProgramAttestationError("invalid exact program integer")
    if (p["delay"] not in DELAYS or p["spacing"] not in SPACINGS or
        p["amplitude"] not in AMPLITUDES or d not in (512,1024) or
        m!=3*d//2+1):
        raise ProgramAttestationError("outside registered native grammar")
    beta=p["hidden_beta"]
    if beta==0:
        if p["hidden_transform"] is not None:
            raise ProgramAttestationError("clean suffix transform must be null")
    elif beta not in AMPLITUDES or p["hidden_transform"] not in OPS:
        raise ProgramAttestationError("invalid hidden program extension")
    return p


def unary(source:tuple[int,...],op:str)->tuple[int,...]:
    """Standalone exact I/N/R/S semantics, independent of corpus.execute."""
    if not source or source[0]!=1:
        raise ProgramAttestationError("source is not an integral unit series")
    n=len(source)
    if op=="I":
        return source
    if op=="N":
        return tuple((-v if i&1 else v) for i,v in enumerate(source))
    if op=="S":
        return tuple(source[i//2] if (i&1)==0 else 0 for i in range(n))
    if op=="R":
        inverse=[1]
        for i in range(1,n):
            coefficient=0
            for k in range(1,i+1):
                coefficient-=source[k]*inverse[i-k]
            inverse.append(coefficient)
        return tuple(inverse)
    raise ProgramAttestationError("unsupported operation")


def execute_authorized_program(program:dict,query:Sequence[int],
                               observed_degree:int)->tuple[int,...]:
    source=tuple(query)
    m=len(source)
    _validate_program(program,observed_degree,m)
    base=unary(source,program["base"])
    left=unary(source,program["u"])
    right=unary(source,program["v"])
    beta=program["hidden_beta"]
    hidden=unary(source,program["hidden_transform"]) if beta else None
    k,space,c=program["delay"],program["spacing"],program["amplitude"]
    onset=observed_degree+32
    values=[]
    for index in range(m):
        value=base[index]
        if index==k:value+=c
        if index>=k+space:value+=c*left[index-k-space]
        if index>=k+2*space:value+=c*right[index-k-2*space]
        if beta and index>=onset:value+=beta*hidden[index-onset]
        values.append(value)
    return tuple(values)


def independently_computed_root(values:Sequence[int])->str:
    """Independent Merkle construction, not calling NOR43 producer code."""
    m=len(values)
    if not 1<=m<2**32:
        raise ProgramAttestationError("invalid output size")
    size=1<<(m-1).bit_length()
    layer=[]
    for i in range(size):
        if i<m:
            leaf=sha256(LEAF+_word(m)+_word(i)+_integer_bytes(values[i])).digest()
        else:
            leaf=sha256(PAD+_word(m)+_word(i)).digest()
        layer.append(leaf)
    while len(layer)>1:
        layer=[
            sha256(NODE+layer[j]+layer[j+1]).digest()
            for j in range(0,len(layer),2)
        ]
    return layer[0].hex()


def verify_authorized_execution(*,certificate:dict,query:Sequence[int],
    session_context:dict,precommit_receipt:dict,provider_reference:dict,
    pinned_program_authority_key:bytes,expected_program_authority_id:str,
    pinned_verifier_public_key:bytes,pinned_provider_public_key:bytes,
    now_epoch:int=2, _root_policy:str="original")->dict:
    """Authenticate an executable statement and fully verify provider root.

    Returns the independently computed vector only to the trusted verifier
    AFTER the model precommitted. Never expose its contents to the model
    before receipt/challenge state has been frozen.
    """
    if not isinstance(certificate,dict) or set(certificate)!={
        "statement","signature_hex"}:
        raise ProgramAttestationError("invalid program authority envelope")
    statement=certificate["statement"]
    fields={"protocol","semantics","authority_id","authority_key_id",
            "context_sha256","session_nonce_hex","source_sha256",
            "observed_degree","coefficient_count","provider_key_id",
            "precommit_receipt_sha256","query_sha256",
            "expected_merkle_root_sha256","program"}
    if not isinstance(statement,dict) or set(statement)!=fields:
        raise ProgramAttestationError("invalid closed attestation schema")
    if statement["protocol"]!=PROTOCOL or statement["semantics"]!=CODE_VERSION:
        raise ProgramAttestationError("untrusted execution semantics")
    if statement["authority_id"]!=expected_program_authority_id:
        raise ProgramAttestationError("untrusted program-authority identity")
    if statement["authority_key_id"]!=_public_key_id(
            pinned_program_authority_key):
        raise ProgramAttestationError("program authority not pinned")
    try:
        signature=bytes.fromhex(certificate["signature_hex"])
        if len(signature)!=64 or certificate["signature_hex"]!=signature.hex():
            raise ProgramAttestationError("noncanonical authority signature")
        Ed25519PublicKey.from_public_bytes(
            pinned_program_authority_key).verify(signature,_canonical(statement))
    except (InvalidSignature,ValueError,TypeError) as exc:
        raise ProgramAttestationError("invalid program-authority signature") from exc
    session.verify_precommit_receipt(
        precommit_receipt,pinned_verifier_public_key,session_context,now_epoch)
    root_reference=session.verify_provider_reference(
        provider_reference,session_context,precommit_receipt,
        pinned_provider_public_key,pinned_verifier_public_key,now_epoch)
    expected_context={
        "context_sha256":sha256(session.canonical(session_context)).hexdigest(),
        "session_nonce_hex":session_context["nonce_hex"],
        "source_sha256":session_context["source_sha256"],
        "observed_degree":session_context["observed_degree"],
        "coefficient_count":session_context["coefficient_count"],
        "provider_key_id":session_context["provider_key_id"],
        "precommit_receipt_sha256":sha256(
            session.canonical(precommit_receipt)).hexdigest(),
        "query_sha256":query_digest(query)
    }
    for field,want in expected_context.items():
        if statement[field]!=want:
            raise ProgramAttestationError(f"cross-session or source replay: {field}")
    program=statement["program"]
    values=execute_authorized_program(
        program,query,session_context["observed_degree"])
    if _root_policy=="original":
        independent_root=independently_computed_root(values)
    elif _root_policy in ("warm","cold"):
        # The ONLY optional policies are NOR59's exact source/program-certified
        # static NOR45 Merkle implementations. There is no root callback,
        # arbitrary template, untrusted stride or bypass of signed checks.
        # Every original authority, precommit, provider, source and executable
        # program check above remains identical and runs before this branch.
        from research.nor59_authorized_merkle import authorized_root
        independent_root=authorized_root.authorized_root(
            program=program,query=query,
            observed_degree=session_context["observed_degree"],
            output=values,mode=_root_policy)
    else:
        raise ProgramAttestationError("unregistered internal NOR45 root policy")
    if (statement["expected_merkle_root_sha256"]!=independent_root or
        root_reference["merkle_root_sha256"]!=independent_root):
        raise ProgramAttestationError(
            "provider output root differs from authorized program execution")
    return {"root_sha256":independent_root,"values":values,
            "program":dict(program),"authority_id":expected_program_authority_id,
            "source_sha256":session_context["source_sha256"]}
