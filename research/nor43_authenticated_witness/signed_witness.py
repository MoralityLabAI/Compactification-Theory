"""NOR-43 test-only signed Merkle coefficient commitment producer.

This module is intentionally a PRODUCER only. The independent verifier lives
in verifier_reference.py, implements all digest framing separately, and pins
a public key external to the claimed signed bundle. The deterministic test
key in this repository is PUBLIC KNOWLEDGE and is never a deployment key.

The separate honest-reference-provider assumption is required for every
claim about true output correctness.
"""
from __future__ import annotations
import json
from hashlib import sha256
from struct import pack
from typing import Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PROTOCOL="NOR43/source-bound-merkle-output/v1"
PROVIDER_ID="NOR43-SYNTHETIC-REFERENCE-ONLY"
LEAF_DOMAIN=b"NOR43/leaf/v1\x00"
NODE_DOMAIN=b"NOR43/node/v1\x00"
PAD_DOMAIN=b"NOR43/padding/v1\x00"
TEST_KEY_SEED=sha256(b"NOR43/REPRODUCIBLE-UNTRUSTED-TEST-PRIVATE-KEY/v1").digest()


def _u32(n:int)->bytes:
    if not isinstance(n,int) or not 0<=n<(1<<32):
        raise ValueError("integer outside u32 framing")
    return pack(">I",n)


def _integer_encoding(value:int)->bytes:
    if type(value) is not int:
        raise TypeError("coefficient must be a Python int")
    sign=1 if value<0 else 0
    magnitude=abs(value)
    count=max(1,(magnitude.bit_length()+7)//8)
    data=magnitude.to_bytes(count,"big")
    assert data[0]!=0 or (len(data)==1 and magnitude==0)
    return bytes((sign,))+_u32(len(data))+data


def _leaf(index:int,total:int,value:int)->bytes:
    if not 0<=index<total:
        raise ValueError("index outside committed output")
    payload=LEAF_DOMAIN+_u32(total)+_u32(index)+_integer_encoding(value)
    return sha256(payload).digest()


def _pad(index:int,total:int)->bytes:
    if not index>=total:raise ValueError("padding index overlaps output")
    return sha256(PAD_DOMAIN+_u32(total)+_u32(index)).digest()


def _node(left:bytes,right:bytes)->bytes:
    assert len(left)==len(right)==32
    return sha256(NODE_DOMAIN+left+right).digest()


def _next_power_two(n:int)->int:
    if not 1<=n<(1<<32):raise ValueError("empty or oversized output")
    return 1<<(n-1).bit_length()


def canonical_manifest(value:dict)->bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def test_private_key()->Ed25519PrivateKey:
    """Public fixture. Not a credential."""
    return Ed25519PrivateKey.from_private_bytes(TEST_KEY_SEED)


def test_reference_public_key()->bytes:
    return test_private_key().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def sha_of_proposed_output(values:Sequence[int])->str:
    """Unforgeable-by-assumption order binding for the model proposal.

    To preserve the NOR42 common-public-view premise, freeze this SHA
    BEFORE disclosing the signed oracle reference root to the proposer.
    """
    total=len(values)
    h=sha256()
    h.update(b"NOR43/proposal/v1\x00"+_u32(total))
    for i,value in enumerate(values):
        encoding=_integer_encoding(value)
        h.update(_u32(i)+_u32(len(encoding))+encoding)
    return h.hexdigest()


class OutputTree:
    """Complete coefficient Merkle tree, with deterministic per-index padding."""
    def __init__(self,values:Sequence[int]):
        coefficients=tuple(values)
        if not coefficients:raise ValueError("empty output")
        self.values=coefficients
        self.m=len(coefficients)
        size=_next_power_two(self.m)
        bottom=[
            _leaf(i,self.m,coefficients[i]) if i<self.m else _pad(i,self.m)
            for i in range(size)]
        levels=[tuple(bottom)]
        while len(levels[-1])>1:
            parent=levels[-1]
            levels.append(tuple(_node(parent[i],parent[i+1])
                                for i in range(0,len(parent),2)))
        self.levels=tuple(levels)
        self.root=self.levels[-1][0]
        self.depth=len(self.levels)-1

    def opening(self,index:int)->dict:
        if type(index) is not int or not 0<=index<self.m:
            raise ValueError("invalid opening index")
        position=index
        siblings=[]
        for level in self.levels[:-1]:
            siblings.append(level[position^1].hex())
            position//=2
        assert len(siblings)==self.depth
        return {"index":index,"coefficient":str(self.values[index]),
                "siblings":siblings}


def signed_reference_manifest(tree:OutputTree,source_sha256:str,
                              observed_degree:int,
                              publisher_id:str=PROVIDER_ID,
                              signing_key:Ed25519PrivateKey|None=None)->dict:
    """Reference-provider role: authority is NOT inferred from this bundle."""
    if (len(source_sha256)!=64 or
        any(ch not in "0123456789abcdef" for ch in source_sha256)):
        raise ValueError("public source must use lowercase SHA-256 hex")
    if not 0<=observed_degree<tree.m:
        raise ValueError("invalid observation degree")
    if not publisher_id or len(publisher_id)>256:
        raise ValueError("publisher ID must be pinned externally")
    key=test_private_key() if signing_key is None else signing_key
    manifest={
        "protocol":PROTOCOL,
        "publisher_id":publisher_id,
        "source_sha256":source_sha256,
        "observed_degree":observed_degree,
        "coefficient_count":tree.m,
        "merkle_root_sha256":tree.root.hex(),
    }
    encoded=canonical_manifest(manifest)
    signature=key.sign(encoded)
    assert len(signature)==64
    return {
        "manifest":manifest,
        "ed25519_signature_hex":signature.hex(),
        "manifest_sha256":sha256(encoded).hexdigest()
    }
