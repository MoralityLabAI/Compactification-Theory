"""NOR55: source-bound Ed25519-fixture signed compact/full witness envelope.

Security: independently pinned verifier key, source input SHA and trusted
reference root, exact canonical signed bytes and unique nonce/sequence.
The fixture private key is public-by-construction and ONLY for research tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json,secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from cryptography.exceptions import InvalidSignature

from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import codec

WIRE_MAGIC=b"N55E"
SIGN_DOMAIN=b"NOR55/ED25519-RESEARCH-V1\x00"
MAX_HEADER_BYTES=4096
MAX_WIRE_BYTES=8_500_000
SIG_LENGTH=64
PROVIDER_ID="nor55-only-research-fixture-v1"
KEY_SEED=sha256(b"NOR55-ED25519-TEST-KEY-NOT-FOR-DEPLOYMENT/v1").digest()
PROTOCOL="nor55.signed_exact_integer_inverse.v1"
FORMATS=("compact-v1","json-decimal-v1")
HEADER_KEYS={
    "protocol","payload_format","payload_len","payload_sha256",
    "source_sha256","source_input_sha256","expected_root_sha256",
    "full_length","stride","provider_id","provider_key_sha256",
    "session_id","sequence","nonce"
}

class SignedWitnessError(ValueError):
    pass

def canonical(x)->bytes:
    return json.dumps(x,sort_keys=True,ensure_ascii=True,
                      separators=(",",":")).encode("ascii")

def research_private_key()->Ed25519PrivateKey:
    """Reproducible *insecure* test fixture private key, never a deployed key."""
    return Ed25519PrivateKey.from_private_bytes(KEY_SEED)

def research_public_key()->Ed25519PublicKey:
    return research_private_key().public_key()

def public_bytes(key:Ed25519PublicKey)->bytes:
    return key.public_bytes(Encoding.Raw,PublicFormat.Raw)

@dataclass(frozen=True)
class TrustedSession:
    source_sha256:str
    source:tuple[int,...]
    stride:int
    expected_root_sha256:str
    payload_format:str
    session_id:str
    sequence:int
    nonce:str
    provider_id:str
    public_key:Ed25519PublicKey
    source_input_sha256:str

def make_context(*,source,source_sha256,stride,expected_root_sha256,
                 payload_format,session_id,sequence,nonce=None,
                 provider_id=PROVIDER_ID,public_key=None)->TrustedSession:
    f=support.checked_source(source,stride)
    if payload_format not in FORMATS:
        raise SignedWitnessError("unsupported trusted payload format")
    if type(sequence) is not int or not 1<=sequence<=1_000_000:
        raise SignedWitnessError("invalid verifier-issued sequence")
    if not isinstance(session_id,str) or not 1<=len(session_id)<=128:
        raise SignedWitnessError("invalid verifier session identifier")
    if not isinstance(source_sha256,str) or len(source_sha256)!=64:
        raise SignedWitnessError("invalid trusted source SHA")
    if not isinstance(expected_root_sha256,str) or len(expected_root_sha256)!=64:
        raise SignedWitnessError("invalid trusted expected full Merkle root")
    n=secrets.token_hex(32) if nonce is None else nonce
    if (not isinstance(n,str) or len(n)!=64 or
        any(ch not in "0123456789abcdef" for ch in n)):
        raise SignedWitnessError("nonce must be 256-bit canonical hex")
    key=research_public_key() if public_key is None else public_key
    if not isinstance(key,Ed25519PublicKey):
        raise SignedWitnessError("trusted public key must be Ed25519")
    if not isinstance(provider_id,str) or not provider_id:
        raise SignedWitnessError("invalid pinned provider identifier")
    digest=sha256(producer.canonical_witness_bytes(f)).hexdigest()
    return TrustedSession(
        source_sha256=source_sha256,source=f,stride=stride,
        expected_root_sha256=expected_root_sha256,
        payload_format=payload_format,session_id=session_id,
        sequence=sequence,nonce=n,provider_id=provider_id,
        public_key=key,source_input_sha256=digest
    )

def _expected_header(context:TrustedSession,payload:bytes)->dict:
    return {
        "protocol":PROTOCOL,
        "payload_format":context.payload_format,
        "payload_len":len(payload),
        "payload_sha256":sha256(payload).hexdigest(),
        "source_sha256":context.source_sha256,
        "source_input_sha256":context.source_input_sha256,
        "expected_root_sha256":context.expected_root_sha256,
        "full_length":len(context.source),
        "stride":context.stride,
        "provider_id":context.provider_id,
        "provider_key_sha256":sha256(public_bytes(context.public_key)).hexdigest(),
        "session_id":context.session_id,
        "sequence":context.sequence,
        "nonce":context.nonce
    }

def _payload(witness,context:TrustedSession)->bytes:
    if context.payload_format=="compact-v1":
        return codec.encode_full_supported(witness,context.stride)
    if context.payload_format=="json-decimal-v1":
        w=tuple(witness)
        support.checked_full_witness(context.source,w,context.stride)
        return producer.canonical_witness_bytes(w)
    raise SignedWitnessError("unknown trusted payload codec")

def make_signed_envelope(witness,context:TrustedSession,
                         signer:Ed25519PrivateKey|None=None)->bytes:
    signer=research_private_key() if signer is None else signer
    if not isinstance(signer,Ed25519PrivateKey):
        raise SignedWitnessError("fixture signer must be Ed25519")
    payload=_payload(witness,context)
    header=canonical(_expected_header(context,payload))
    if len(header)>MAX_HEADER_BYTES:
        raise SignedWitnessError("signed source header too large")
    unsigned=(WIRE_MAGIC+len(header).to_bytes(4,"big")+header+payload)
    result=unsigned+signer.sign(SIGN_DOMAIN+unsigned)
    if len(result)>MAX_WIRE_BYTES:
        raise SignedWitnessError("signed source envelope exceeds byte cap")
    return result

class ReplayGuard:
    def __init__(self):
        self.last={}
        self.nonces=set()

    def check(self,context:TrustedSession):
        key=(context.source_sha256,context.session_id,context.provider_id)
        if context.nonce in self.nonces:
            raise SignedWitnessError("replayed verifier nonce")
        if context.sequence!=self.last.get(key,0)+1:
            raise SignedWitnessError("nonmonotone signer session sequence")
        return key

    def commit(self,key,context:TrustedSession):
        self.last[key]=context.sequence
        self.nonces.add(context.nonce)

def inspect_header(wire:bytes)->tuple[dict,bytes,bytes,bytes]:
    if type(wire) is not bytes or not 8+SIG_LENGTH<=len(wire)<=MAX_WIRE_BYTES:
        raise SignedWitnessError("non-byte or oversized signed envelope")
    if not wire.startswith(WIRE_MAGIC):
        raise SignedWitnessError("invalid signed envelope domain")
    header_length=int.from_bytes(wire[4:8],"big")
    if not 1<=header_length<=MAX_HEADER_BYTES:
        raise SignedWitnessError("invalid signed header length")
    head_end=8+header_length
    if head_end+SIG_LENGTH>len(wire):
        raise SignedWitnessError("truncated signed header/payload")
    head_raw=wire[8:head_end]
    try:
        header=json.loads(head_raw.decode("ascii"))
    except (UnicodeDecodeError,ValueError) as exc:
        raise SignedWitnessError("malformed signed header") from exc
    if type(header) is not dict or set(header)!=HEADER_KEYS or canonical(header)!=head_raw:
        raise SignedWitnessError("noncanonical signed header schema")
    unsigned=wire[:-SIG_LENGTH]
    payload=wire[head_end:-SIG_LENGTH]
    signature=wire[-SIG_LENGTH:]
    if (type(header["payload_len"]) is not int or
        header["payload_len"]!=len(payload)):
        raise SignedWitnessError("untrusted signed payload byte length mismatch")
    if header["payload_sha256"]!=sha256(payload).hexdigest():
        raise SignedWitnessError("signed payload digest mismatch")
    return header,payload,signature,unsigned

def verify_signed_envelope(wire:bytes,context:TrustedSession,guard:ReplayGuard):
    if not isinstance(guard,ReplayGuard):
        raise SignedWitnessError("missing trusted replay guard")
    header,payload,signature,unsigned=inspect_header(wire)
    key=guard.check(context)
    if header!=_expected_header(context,payload):
        raise SignedWitnessError("signature header disagrees with trusted source/session/root")
    try:
        context.public_key.verify(signature,SIGN_DOMAIN+unsigned)
    except InvalidSignature as exc:
        raise SignedWitnessError("invalid pinned provider Ed25519 signature") from exc
    if context.payload_format=="compact-v1":
        witness=codec.decode_full_supported(payload,len(context.source),context.stride)
    elif context.payload_format=="json-decimal-v1":
        witness=producer.parse_witness_bytes(payload,len(context.source))
    else:
        raise SignedWitnessError("untrusted source format")
    # Original full-source/full-witness resource caps, support positions,
    # exact signed identity and ORIGINAL full m-coefficient Merkle root.
    values,root=fast_kernel.compressed_verified_root(
        context.source,witness,context.stride)
    if root!=context.expected_root_sha256:
        raise SignedWitnessError("verified exact witness root disagrees with trusted root")
    guard.commit(key,context)
    return values,root
