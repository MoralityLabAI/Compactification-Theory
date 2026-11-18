"""NOR54: local end-to-end exact compressed producer + fast verifier arm.

On each invocation produce a fresh complete witness, canonically serialize it,
parse as builtin ints and independently validate via NOR53 fast compressed GMP.
This arm measures local arithmetic + codec and Merkle cost, not a remote
provider, signature, or network transfer.
"""
from __future__ import annotations
import json
from collections.abc import Sequence
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel

class LocalWitnessCodecError(ValueError):
    pass

def canonical_witness_bytes(values:Sequence[int])->bytes:
    if not isinstance(values,(list,tuple)):
        raise LocalWitnessCodecError("exact witness container must be list/tuple")
    if any(type(v) is not int for v in values):
        raise LocalWitnessCodecError("witness must contain builtin exact integers")
    return json.dumps([str(x) for x in values],
                      separators=(",",":"),ensure_ascii=True).encode("ascii")

def parse_witness_bytes(raw:bytes,expected_len:int)->tuple[int,...]:
    if type(raw) is not bytes:
        raise LocalWitnessCodecError("wire witness must be bytes")
    if type(expected_len) is not int or expected_len<1:
        raise LocalWitnessCodecError("invalid expected full witness length")
    try:
        items=json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError,ValueError) as exc:
        raise LocalWitnessCodecError("invalid exact witness JSON") from exc
    if type(items) is not list or len(items)!=expected_len:
        raise LocalWitnessCodecError("non-list or wrong length witness")
    for item in items:
        if type(item) is not str or not item:
            raise LocalWitnessCodecError("witness coefficient must be decimal text")
        if item[0]=="-":
            if len(item)==1 or not item[1:].isdigit():
                raise LocalWitnessCodecError("malformed signed coefficient")
        elif not item.isdigit():
            raise LocalWitnessCodecError("malformed unsigned coefficient")
    # Parse and verify a single canonical representation, preventing alternate
    # textual encodings or noncanonical negative zero.
    try:
        values=tuple(int(s) for s in items)
    except ValueError as exc:
        raise LocalWitnessCodecError("witness integer conversion failed") from exc
    if canonical_witness_bytes(values)!=raw:
        raise LocalWitnessCodecError("noncanonical exact decimal witness")
    return values

def produce_compressed_full_witness(source:Sequence[int],stride:int)->tuple[int,...]:
    f=support.checked_source(source,stride)
    reduced=support.compress(f,stride)
    inverse_reduced=native.unary(reduced,"R")
    full=support.expand(inverse_reduced,len(f),stride)
    # Same original full-space bounds as NOR52 original checked execution.
    gmp._parameters(f,full)
    return full

def compressed_produce_serialize_fast_verify_root(source:Sequence[int],stride:int):
    """End-to-end local production+serialization+parse+verification+full root."""
    f=support.checked_source(source,stride)
    produced=produce_compressed_full_witness(f,stride)
    raw=canonical_witness_bytes(produced)
    decoded=parse_witness_bytes(raw,len(f))
    return fast_kernel.compressed_verified_root(f,decoded,stride)
