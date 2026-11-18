"""NOR58 common authenticated signed-v2 exact GMP witness acceptance.

All arms consume byte-identical signed envelope and independently trusted
source/nonce/provider/root context, check the same full original signed caps
and exact GMP identity. Only the original NOR45 full Merkle implementation
differs: original root, indexed static warm template, or cold per-call build.
"""
from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import signed_v2,codec_v2
from research.nor58_static_merkle import template

ORIGINAL="original"
WARM="warm"
COLD="cold"
MODES=(ORIGINAL,WARM,COLD)

class NOR58SignedRootError(ValueError):
    pass

def verify_same_signed_witness(wire:bytes,context:v1.TrustedSession,
                               guard:v1.ReplayGuard,mode:str,
                               warm_template:template.StaticGeometryTemplate|None=None):
    if mode not in MODES:
        raise NOR58SignedRootError("unregistered signed exact Merkle root mode")
    if not isinstance(context,v1.TrustedSession) or not isinstance(guard,v1.ReplayGuard):
        raise NOR58SignedRootError("independent trusted signed source/session context required")
    if context.payload_format!=signed_v2.FORMAT_V2:
        raise NOR58SignedRootError("signed compact-v2 source version required")
    # The only template key is the source geometry independently trusted by
    # verifier. Refuse even a prebuilt mismatched template before decoding.
    if mode==WARM and (not isinstance(warm_template,template.StaticGeometryTemplate)
        or warm_template.m!=len(context.source)
        or warm_template.stride!=context.stride):
        raise NOR58SignedRootError("warm template trusted geometry mismatch")
    f=support.checked_source(context.source,context.stride)
    header,payload,signature,unsigned=v1.inspect_header(wire)
    guard_key=guard.check(context)
    if header!=signed_v2.expected_header(context,payload):
        raise NOR58SignedRootError("signed source/nonce/provider/full root mismatch")
    try:
        context.public_key.verify(signature,signed_v2.SIGN_DOMAIN_V2+unsigned)
    except InvalidSignature as exc:
        raise NOR58SignedRootError("invalid pinned research Ed25519 signature") from exc
    witness=codec_v2.decode_full_supported(payload,len(f),context.stride)
    full_f,full_w=support.checked_full_witness(f,witness,context.stride)
    reduced_f=support.compress(full_f,context.stride)
    reduced_w=support.compress(full_w,context.stride)
    product=fast_kernel.exact_convolution_prefix_fast(reduced_f,reduced_w)
    if product!=(1,)+(0,)*(len(reduced_f)-1):
        raise NOR58SignedRootError("signed exact GMP reduced inverse identity mismatch")
    if mode==ORIGINAL:
        root=nor45.independently_computed_root(full_w)
    elif mode==WARM:
        root=warm_template.root(full_w)
    else:
        cold=template.build_template(len(full_f),context.stride)
        root=cold.root(full_w)
    if root!=context.expected_root_sha256:
        raise NOR58SignedRootError("signed witness full original Merkle root mismatch")
    guard.commit(guard_key,context)
    return full_w,root
