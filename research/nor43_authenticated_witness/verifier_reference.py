"""NOR-43 independent reference-verifier for pinned signed coefficient roots.

Imports none of the producer code or its internal helpers. An authorized
reference root is NOT evidence that the publisher computed the true output:
the honest target provider and pinned key distribution are external premises.
"""
from __future__ import annotations

import json
from hashlib import sha256
from struct import pack

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PROTOCOL="NOR43/source-bound-merkle-output/v1"
LEAF=b"NOR43/leaf/v1\x00"
NODE=b"NOR43/node/v1\x00"


class VerificationError(ValueError):
    pass


def _word(n):
    if type(n) is not int or not 0<=n<(1<<32):
        raise VerificationError("invalid u32 index or length")
    return pack(">I",n)


def _signed_value(value):
    if type(value) is not str or not value:
        raise VerificationError("coefficient must be canonical decimal str")
    try:
        n=int(value)
    except (ValueError,OverflowError) as exc:
        raise VerificationError("invalid coefficient") from exc
    if str(n)!=value:
        raise VerificationError("noncanonical integer representation")
    length=max(1,(abs(n).bit_length()+7)//8)
    return bytes([int(n<0)])+_word(length)+abs(n).to_bytes(length,"big")


def _expected_depth(n):
    if type(n) is not int or not 1<=n<2**32:
        raise VerificationError("invalid committed output length")
    return (n-1).bit_length()


def verify_signed_manifest(bundle:dict,pinned_public_key:bytes,
                           expected_source_sha256:str,
                           expected_observed_degree:int,
                           expected_coefficient_count:int,
                           expected_publisher_id:str)->dict:
    """Require all context and signature before accepting any opening."""
    if not isinstance(bundle,dict) or set(bundle)!={
        "manifest","ed25519_signature_hex","manifest_sha256"}:
        raise VerificationError("unexpected signed manifest envelope")
    manifest=bundle["manifest"]
    required={
        "protocol","publisher_id","source_sha256",
        "observed_degree","coefficient_count","merkle_root_sha256"}
    if not isinstance(manifest,dict) or set(manifest)!=required:
        raise VerificationError("unexpected signed manifest schema")
    if manifest["protocol"]!=PROTOCOL:
        raise VerificationError("wrong protocol domain")
    if manifest["publisher_id"]!=expected_publisher_id:
        raise VerificationError("untrusted provider")
    if manifest["source_sha256"]!=expected_source_sha256:
        raise VerificationError("public source mismatch")
    if manifest["observed_degree"]!=expected_observed_degree:
        raise VerificationError("observed prefix degree mismatch")
    if manifest["coefficient_count"]!=expected_coefficient_count:
        raise VerificationError("future length mismatch")
    m=manifest["coefficient_count"]
    if type(m) is not int or type(manifest["observed_degree"]) is not int:
        raise VerificationError("noninteger index/degree")
    if not 0<=manifest["observed_degree"]<m:
        raise VerificationError("invalid observed interval")
    _expected_depth(m)
    try:
        root=bytes.fromhex(manifest["merkle_root_sha256"])
        signature=bytes.fromhex(bundle["ed25519_signature_hex"])
        public_key=bytes(pinned_public_key)
    except (ValueError,TypeError) as exc:
        raise VerificationError("malformed root, signature or key") from exc
    if (len(root)!=32 or len(signature)!=64 or len(public_key)!=32 or
        root.hex()!=manifest["merkle_root_sha256"]):
        raise VerificationError("noncanonical cryptographic bytes")
    raw=json.dumps(manifest,sort_keys=True,separators=(",",":"),
                   ensure_ascii=True).encode("ascii")
    if sha256(raw).hexdigest()!=bundle["manifest_sha256"]:
        raise VerificationError("manifest digest mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature,raw)
    except (ValueError,InvalidSignature) as exc:
        raise VerificationError("signature failed against pinned reference key") from exc
    return manifest


def verify_index_opening(verified_manifest:dict,opening:dict)->bool:
    """Merkle inclusion against already authenticated reference manifest."""
    if not isinstance(opening,dict) or set(opening)!={
        "index","coefficient","siblings"}:
        raise VerificationError("invalid opening schema")
    index=opening["index"]
    n=verified_manifest["coefficient_count"]
    if type(index) is not int or not 0<=index<n:
        raise VerificationError("opened coefficient index outside future")
    siblings=opening["siblings"]
    if not isinstance(siblings,list) or len(siblings)!=_expected_depth(n):
        raise VerificationError("incomplete or oversized Merkle path")
    leaf=sha256(LEAF+_word(n)+_word(index)+_signed_value(
        opening["coefficient"])).digest()
    position=index
    for sibling_hex in siblings:
        try:
            sibling=bytes.fromhex(sibling_hex)
        except (TypeError,ValueError) as exc:
            raise VerificationError("invalid Merkle sibling") from exc
        if len(sibling)!=32 or sibling.hex()!=sibling_hex:
            raise VerificationError("noncanonical Merkle sibling")
        if position%2:
            leaf=sha256(NODE+sibling+leaf).digest()
        else:
            leaf=sha256(NODE+leaf+sibling).digest()
        position//=2
    if leaf.hex()!=verified_manifest["merkle_root_sha256"]:
        raise VerificationError("Merkle path or coefficient mismatch")
    return True


def verified_sample_rejection(proposal:list[int]|tuple[int,...],
                              verified_manifest:dict,
                              openings:list[dict],expected_query_count:int=16)->bool:
    """Reject public-model proposal iff a valid authenticated checked value differs."""
    if len(proposal)!=verified_manifest["coefficient_count"]:
        raise VerificationError("proposal length mismatch")
    if len(openings)!=expected_query_count:
        raise VerificationError("unexpected audit query budget")
    indices=[entry.get("index") for entry in openings]
    if len(indices)!=len(set(indices)):
        raise VerificationError("repeated checked indices")
    rejected=False
    for entry in openings:
        verify_index_opening(verified_manifest,entry)
        if proposal[entry["index"]]!=int(entry["coefficient"]):
            rejected=True
    return rejected
