"""NOR56 version-bound Ed25519 compact-v2 exact inverse verification.

Reuses NOR55's *trusted source/session context* and canonical header parser
without modifying its original first-run v1/JSON implementation.
The canonical header explicitly binds protocol=v2 and format=compact-v2;
a distinct signature domain rejects cross-version signed-wire substitution.
"""
from __future__ import annotations
from dataclasses import replace
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from research.nor53_native_pack import fast_kernel
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import codec_v2

FORMAT_V2="compact-v2"
PROTOCOL_V2="nor56.signed_exact_integer_inverse.v2"
SIGN_DOMAIN_V2=b"NOR56/ED25519-RESEARCH-COMPACT-V2\x00"

def make_context_v2(*,source,source_sha256,stride,expected_root_sha256,
                    session_id,sequence,nonce=None,provider_id=v1.PROVIDER_ID,
                    public_key=None):
    """All NOR55 context/cap/source checks, then only format version changes."""
    original=v1.make_context(
        source=source,source_sha256=source_sha256,stride=stride,
        expected_root_sha256=expected_root_sha256,payload_format="compact-v1",
        session_id=session_id,sequence=sequence,nonce=nonce,
        provider_id=provider_id,public_key=public_key)
    return replace(original,payload_format=FORMAT_V2)

def expected_header(context:v1.TrustedSession,payload:bytes)->dict:
    if not isinstance(context,v1.TrustedSession) or context.payload_format!=FORMAT_V2:
        raise v1.SignedWitnessError("trusted context is not compact-v2")
    header=v1._expected_header(context,payload)
    header["protocol"]=PROTOCOL_V2
    return header

def make_signed_envelope_v2(witness,context:v1.TrustedSession,
                            signer:Ed25519PrivateKey|None=None)->bytes:
    signer=v1.research_private_key() if signer is None else signer
    if not isinstance(signer,Ed25519PrivateKey):
        raise v1.SignedWitnessError("v2 test signer must be Ed25519")
    from research.nor52_sparse_inverse import algebra as support
    # Producer must send an exact canonical supported signed full witness;
    # original signed bounds are independently enforced by the receiver.
    support.checked_full_witness(context.source,witness,context.stride)
    payload=codec_v2.encode_full_supported(witness,context.stride)
    head=v1.canonical(expected_header(context,payload))
    if len(head)>v1.MAX_HEADER_BYTES:
        raise v1.SignedWitnessError("compact-v2 signed header oversized")
    unsigned=v1.WIRE_MAGIC+len(head).to_bytes(4,"big")+head+payload
    wire=unsigned+signer.sign(SIGN_DOMAIN_V2+unsigned)
    if len(wire)>v1.MAX_WIRE_BYTES:
        raise v1.SignedWitnessError("compact-v2 signed envelope oversized")
    return wire

def verify_signed_envelope_v2(wire:bytes,context:v1.TrustedSession,
                              guard:v1.ReplayGuard):
    if not isinstance(guard,v1.ReplayGuard):
        raise v1.SignedWitnessError("trusted replay guard missing")
    head,payload,signature,unsigned=v1.inspect_header(wire)
    key=guard.check(context)
    if head!=expected_header(context,payload):
        raise v1.SignedWitnessError("compact-v2 signed header/source/session mismatch")
    try:
        context.public_key.verify(signature,SIGN_DOMAIN_V2+unsigned)
    except InvalidSignature as exc:
        raise v1.SignedWitnessError("compact-v2 invalid pinned Ed25519 signature") from exc
    values=codec_v2.decode_full_supported(
        payload,len(context.source),context.stride)
    # Recheck all original full source/witness caps, support zeros and exact
    # signed GMP identity. Recompute ORIGINAL full Merkle root independently.
    result,root=fast_kernel.compressed_verified_root(
        context.source,values,context.stride)
    if root!=context.expected_root_sha256:
        raise v1.SignedWitnessError("compact-v2 verified full root mismatch")
    guard.commit(key,context)
    return result,root
