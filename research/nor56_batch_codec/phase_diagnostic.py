"""NOR56 separate, UNTITMED diagnostic verifier phase replay.

Runs AFTER the registered first whole-operation CPU arms. Splits exact
header/replay, signature, canonical decode, original full support/caps,
native-packed exact GMP convolution, and full NOR45 Merkle root.
The phase ledger can never change the prospectively registered five gates.
"""
from __future__ import annotations
import time
from cryptography.exceptions import InvalidSignature
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import codec as v1codec,envelope as v1
from research.nor56_batch_codec import codec_v2,signed_v2

PHASES=(
    "header_and_replay_ns","ed25519_signature_ns",
    "canonical_payload_decode_ns","full_source_witness_support_caps_ns",
    "signed_exact_GMP_convolution_ns","full_original_Merkle_root_ns"
)

def profile_one(wire:bytes,context:v1.TrustedSession,
                format_name:str,reference,expected_root:str)->dict:
    assert format_name in ("compact-v1","json-decimal-v1",signed_v2.FORMAT_V2)
    guard=v1.ReplayGuard()
    t0=time.perf_counter_ns()
    header,payload,signature,unsigned=v1.inspect_header(wire)
    key=guard.check(context)
    expected=(signed_v2.expected_header(context,payload)
              if format_name==signed_v2.FORMAT_V2 else
              v1._expected_header(context,payload))
    if header!=expected or context.payload_format!=format_name:
        raise v1.SignedWitnessError("phase replay trusted signed source mismatch")
    t1=time.perf_counter_ns()
    context.public_key.verify(
        signature,
        (signed_v2.SIGN_DOMAIN_V2 if format_name==signed_v2.FORMAT_V2
         else v1.SIGN_DOMAIN)+unsigned)
    t2=time.perf_counter_ns()
    if format_name=="compact-v1":
        witness=v1codec.decode_full_supported(
            payload,len(context.source),context.stride)
    elif format_name==signed_v2.FORMAT_V2:
        witness=codec_v2.decode_full_supported(
            payload,len(context.source),context.stride)
    else:
        witness=producer.parse_witness_bytes(payload,len(context.source))
    t3=time.perf_counter_ns()
    f,w=support.checked_full_witness(
        context.source,witness,context.stride)
    t4=time.perf_counter_ns()
    small_f=support.compress(f,context.stride)
    small_w=support.compress(w,context.stride)
    product=fast_kernel.exact_convolution_prefix_fast(small_f,small_w)
    if product!=(1,)+(0,)*(len(small_f)-1):
        raise ValueError("diagnostic exact signed GMP inverse identity fails")
    t5=time.perf_counter_ns()
    root=native.independently_computed_root(w)
    if root!=context.expected_root_sha256 or root!=expected_root or w!=tuple(reference):
        raise ValueError("diagnostic full original Merkle/reference parity fails")
    guard.commit(key,context)
    t6=time.perf_counter_ns()
    phases={
        "header_and_replay_ns":t1-t0,
        "ed25519_signature_ns":t2-t1,
        "canonical_payload_decode_ns":t3-t2,
        "full_source_witness_support_caps_ns":t4-t3,
        "signed_exact_GMP_convolution_ns":t5-t4,
        "full_original_Merkle_root_ns":t6-t5
    }
    if any(type(x) is not int or x<0 for x in phases.values()):
        raise ValueError("negative diagnostic time duration")
    other_guard=v1.ReplayGuard()
    other=(signed_v2.verify_signed_envelope_v2(wire,context,other_guard)
           if format_name==signed_v2.FORMAT_V2
           else v1.verify_signed_envelope(wire,context,other_guard))
    if other!=(w,root):
        raise ValueError("phase replay not equal to unchanged signed verifier")
    return {
        **phases,
        "phase_total_ns":sum(phases.values()),
        "format":format_name,
        "decoded_reduced_length":support.compressed_length(len(f),context.stride),
        "original_full_length":len(f),
        "full_output_and_root_equal":True,
        "signature_and_exact_identity_pass":True,
        "payload_bytes":len(payload),
        "signed_wire_bytes":len(wire)
    }
