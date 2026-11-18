"""NOR57 same-envelope Ed25519 source-bound exact witness acceptance.

All three arms authenticate and decode identical NOR56 compact-v2 bytes,
validate the original FULL source/witness/packing caps, check exact integer
identity, independently build the FULL NOR45 Merkle root and commit replay.
Only the algebraic identity step may differ. Source selector is frozen from
historical first panels, never from witness, nonce, provider or timing.
"""
from __future__ import annotations
from cryptography.exceptions import InvalidSignature
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import signed_v2,codec_v2

RULE_ID="NOR57-FROZEN-PUBLIC-SELECTOR-v1"
MODES=("always_gmp","always_native","frozen_selector")
GMP="GMP"
NATIVE="NATIVE"

class NOR57WitnessError(ValueError):
    pass

def select_from_public_source(source,stride:int)->str:
    """Only independently trusted FULL public source controls routing.

    Exact frozen rule:
       NATIVE iff supported stride=4, length in (769,1537), and actual
       maximum source coefficient abs(bit_length) == 16.
    """
    f=support.checked_source(source,stride)
    bit_width=max(abs(x).bit_length() for x in f)
    if stride==4 and len(f) in (769,1537) and bit_width==16:
        return NATIVE
    return GMP

def recomputed_native_identity(source_reduced,witness_reduced,
                               original_full_witness,full_length,stride):
    """Independent native reduced reciprocal must exactly equal signed witness."""
    actual=native.unary(source_reduced,"R")
    if (len(actual)!=len(witness_reduced) or
        any(a!=b for a,b in zip(actual,witness_reduced))):
        raise NOR57WitnessError("signed witness differs from independent exact native reciprocal")
    expanded=support.expand(actual,full_length,stride)
    if expanded!=original_full_witness:
        raise NOR57WitnessError("native recomputed full output differs from signed full witness")

def checked_gmp_identity(source_reduced,witness_reduced):
    result=fast_kernel.exact_convolution_prefix_fast(
        source_reduced,witness_reduced)
    if result!=(1,)+(0,)*(len(source_reduced)-1):
        raise NOR57WitnessError("GMP signed exact inverse identity failed")

def verify_same_signed_envelope(wire:bytes,context:v1.TrustedSession,
                                guard:v1.ReplayGuard,mode:str):
    """One common signed verification pipeline for all original NOR57 arms."""
    if mode not in MODES:
        raise NOR57WitnessError("unknown registered signed acceptance backend")
    if not isinstance(context,v1.TrustedSession) or not isinstance(guard,v1.ReplayGuard):
        raise NOR57WitnessError("missing independently trusted context/guard")
    # Routing decision is made BEFORE reading untrusted envelope bytes.
    f=support.checked_source(context.source,context.stride)
    decision=select_from_public_source(f,context.stride) if mode=="frozen_selector" else (
        NATIVE if mode=="always_native" else GMP)
    if context.payload_format!=signed_v2.FORMAT_V2:
        raise NOR57WitnessError("NOR57 requires exact signed compact-v2 source format")
    # Shared parsing/header, unique nonce/session, provider key and signature.
    header,payload,signature,unsigned=v1.inspect_header(wire)
    key=guard.check(context)
    if header!=signed_v2.expected_header(context,payload):
        raise NOR57WitnessError("source/nonce/session/expected-root mismatch")
    try:
        context.public_key.verify(
            signature,signed_v2.SIGN_DOMAIN_V2+unsigned)
    except InvalidSignature as exc:
        raise NOR57WitnessError("invalid pinned Ed25519 research signature") from exc
    witness=codec_v2.decode_full_supported(
        payload,len(f),context.stride)
    # Original full-space built-in signed integer, source support, coefficient
    # and packed resource bounds checked identically before either identity.
    full_f,full_w=support.checked_full_witness(f,witness,context.stride)
    reduced_f=support.compress(full_f,context.stride)
    reduced_w=support.compress(full_w,context.stride)
    if decision==NATIVE:
        recomputed_native_identity(
            reduced_f,reduced_w,full_w,len(full_f),context.stride)
    elif decision==GMP:
        checked_gmp_identity(reduced_f,reduced_w)
    else:
        raise NOR57WitnessError("unregistered exact verifier backend")
    root=native.independently_computed_root(full_w)
    if root!=context.expected_root_sha256:
        raise NOR57WitnessError("signed witness full original Merkle root mismatch")
    guard.commit(key,context)
    return full_w,root
