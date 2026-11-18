"""NOR45 historical-only signed program-authority statement producer.

Caller-supplied signer is mandatory. Public deterministic test keys are
fixtures and cannot establish actual provider/program truth or deployment
authorization. The independent execution verifier never imports this file.
"""
from __future__ import annotations
import json
from hashlib import sha256
from struct import pack
from typing import Sequence
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.nor44_session_protocol import session_protocol as session

ATTEST_PROTOCOL="NOR45/authorized-executable-output/v1"
SEMANTICS="NOR45/RRC25-exact-series-plus-optional-D32-suffix/v1"
TEST_AUTHORITY_ID="NOR45-PUBLIC-TEST-PROGRAM-AUTHORITY"
TEST_AUTHORITY_SEED=sha256(
    b"NOR45/UNTRUSTED-PUBLIC-TEST-ONLY-AUTHORITY-PRIVATE-KEY/v1").digest()
OPS=("I","N","R","S")
DELAYS=(64,80,96,112)
SPACINGS=(24,32)
AMPLITUDES=(-2,-1,1,2)


def canonical(value:dict)->bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def key_id(public_key:bytes)->str:
    if not isinstance(public_key,bytes) or len(public_key)!=32:
        raise ValueError("invalid Ed25519 program authority public key")
    return sha256(public_key).hexdigest()


def public_key_of(private_key:Ed25519PrivateKey)->bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def test_authority_private_key()->Ed25519PrivateKey:
    """Publicly known test fixture. Never use for real authentication."""
    return Ed25519PrivateKey.from_private_bytes(TEST_AUTHORITY_SEED)


def _framed_signed_integer(x:int)->bytes:
    if type(x) is not int:raise TypeError("query coefficient not an exact integer")
    magnitude=abs(x)
    n=max(1,(magnitude.bit_length()+7)//8)
    return bytes([int(x<0)])+pack(">I",n)+magnitude.to_bytes(n,"big")


def query_digest(values:Sequence[int])->str:
    """Exact indexed source-query commitment, independent of proposal hash."""
    if not 1<=len(values)<2**32 or values[0]!=1:
        raise ValueError("expected normalized finite integral source")
    h=sha256(b"NOR45/exact-query-vector/v1\x00"+pack(">I",len(values)))
    for i,c in enumerate(values):
        encoded=_framed_signed_integer(c)
        h.update(pack(">I",i)+pack(">I",len(encoded))+encoded)
    return h.hexdigest()


def validate_program_spec(program:dict,observation_degree:int)->dict:
    required={"base","delay","spacing","amplitude","u","v",
              "hidden_beta","hidden_transform"}
    if type(program) is not dict or set(program)!=required:
        raise ValueError("invalid closed program specification")
    if (program["base"] not in OPS or program["u"] not in OPS or
        program["v"] not in OPS):
        raise ValueError("unsupported native transform")
    if (type(program["delay"]) is not int or
        program["delay"] not in DELAYS or
        type(program["spacing"]) is not int or
        program["spacing"] not in SPACINGS or
        type(program["amplitude"]) is not int or
        program["amplitude"] not in AMPLITUDES):
        raise ValueError("unsupported RRC25 finite program")
    if type(program["hidden_beta"]) is not int:
        raise ValueError("hidden suffix amplitude must be exact integer")
    if program["hidden_beta"]==0:
        if program["hidden_transform"] is not None:
            raise ValueError("clean program must have null hidden transform")
    elif (program["hidden_beta"] not in AMPLITUDES or
          program["hidden_transform"] not in OPS):
        raise ValueError("hidden program is outside registered suffix law")
    if type(observation_degree) is not int or observation_degree not in (512,1024):
        raise ValueError("unsupported NOR42 visible degree")
    return program


def sign_authorized_program(*,program:dict,query:Sequence[int],
                            context:dict,precommit_receipt:dict,
                            pinned_verifier_public_key:bytes,
                            expected_output_merkle_root:str,
                            program_authority_signer:Ed25519PrivateKey,
                            authority_id:str=TEST_AUTHORITY_ID,
                            now_epoch:int=2)->dict:
    """Sign a REFERENCE statement after independent verifier precommit.

    Calling this method still assumes the authority knows which program
    represents true world. The method does not derive that premise.
    """
    session.verify_precommit_receipt(
        precommit_receipt,pinned_verifier_public_key,context,now_epoch)
    validate_program_spec(program,context["observed_degree"])
    if len(query)!=context["coefficient_count"]:
        raise ValueError("query/output vector length mismatch")
    if type(authority_id) is not str or not authority_id:
        raise ValueError("missing program authority ID")
    if not isinstance(expected_output_merkle_root,str) or len(
        expected_output_merkle_root)!=64 or any(
        c not in "0123456789abcdef" for c in expected_output_merkle_root):
        raise ValueError("noncanonical expected root digest")
    if not isinstance(program_authority_signer,Ed25519PrivateKey):
        raise TypeError("an explicitly provisioned program signer is required")
    statement={
        "protocol":ATTEST_PROTOCOL,
        "semantics":SEMANTICS,
        "authority_id":authority_id,
        "authority_key_id":key_id(public_key_of(program_authority_signer)),
        "context_sha256":sha256(session.canonical(context)).hexdigest(),
        "session_nonce_hex":context["nonce_hex"],
        "source_sha256":context["source_sha256"],
        "observed_degree":context["observed_degree"],
        "coefficient_count":context["coefficient_count"],
        "provider_key_id":context["provider_key_id"],
        "precommit_receipt_sha256":sha256(session.canonical(
            precommit_receipt)).hexdigest(),
        "query_sha256":query_digest(query),
        "expected_merkle_root_sha256":expected_output_merkle_root,
        "program":dict(program),
    }
    sig=program_authority_signer.sign(canonical(statement))
    return {"statement":statement,"signature_hex":sig.hex()}
