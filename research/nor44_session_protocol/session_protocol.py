"""NOR-44 historical-only reference session with signed precommit and challenge.

Security assumptions, all EXTERNAL: honest authorized target provider, correct
pinned provider key, trustworthy verifier state and CSPRNG, SHA256 binding and
Ed25519 unforgeability. Deterministic keys and injected random values are
explicitly TEST ONLY. This implementation makes no claim of deployment trust.

The runtime cannot verify semantic truth from a signed reference root alone.
"""
from __future__ import annotations

import json
import secrets
from hashlib import sha256
from typing import Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from research.nor43_authenticated_witness import signed_witness as nor43_producer
from research.nor43_authenticated_witness import verifier_reference as nor43_verifier

CONTEXT_PROTOCOL="NOR44/session-context/v1"
PROPOSAL_PROTOCOL="NOR44/source-and-session-proposal/v1"
RECEIPT_PROTOCOL="NOR44/verifier-precommit-receipt/v1"
REFERENCE_PROTOCOL="NOR44/provider-session-reference/v1"
CHALLENGE_PROTOCOL="NOR44/postcommit-challenge/v1"
FINISH_PROTOCOL="NOR44/terminal-verifier-result/v1"
QUERY_COUNT=16
CONTEXT_FIELDS=frozenset({
    "protocol","nonce_hex","source_sha256","observed_degree",
    "coefficient_count","provider_id","provider_key_id",
    "verifier_key_id","expires_epoch","test_only"
})
TEST_VERIFIER_KEY_SEED=sha256(
    b"NOR44/REPRODUCIBLE-PUBLIC-TEST-VERIFIER-SIGNER/v1").digest()


class SessionError(ValueError):
    """Protocol/context/order/authenticity failure. No adaptive retry."""


def _hex_digest(value, name):
    if type(value) is not str or len(value)!=64 or any(
            c not in "0123456789abcdef" for c in value):
        raise SessionError(f"invalid {name} SHA-256 hex")
    return value


def _u32(value, field):
    if type(value) is not int or not 0<=value<(1<<32):
        raise SessionError(f"invalid {field}")
    return value


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def public_key_bytes(private:Ed25519PrivateKey):
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def public_key_id(pinned:bytes):
    if not isinstance(pinned,bytes) or len(pinned)!=32:
        raise SessionError("invalid externally pinned Ed25519 public key")
    return sha256(pinned).hexdigest()


def test_verifier_private_key():
    """PUBLIC reproducible fixture, unsuitable for real authorization."""
    return Ed25519PrivateKey.from_private_bytes(TEST_VERIFIER_KEY_SEED)


def _validate_context(context):
    if not isinstance(context,dict) or set(context)!=CONTEXT_FIELDS:
        raise SessionError("invalid source/session context schema")
    if context["protocol"]!=CONTEXT_PROTOCOL:
        raise SessionError("wrong session protocol")
    for field in ("nonce_hex","source_sha256","provider_key_id","verifier_key_id"):
        _hex_digest(context[field],field)
    degree=_u32(context["observed_degree"],"observation degree")
    n=_u32(context["coefficient_count"],"output length")
    if not 0<=degree<n or n<QUERY_COUNT:
        raise SessionError("invalid observed output interval")
    _u32(context["expires_epoch"],"expiration epoch")
    if (type(context["provider_id"]) is not str or
        not context["provider_id"] or len(context["provider_id"])>256):
        raise SessionError("invalid provider identity")
    if type(context["test_only"]) is not bool:
        raise SessionError("explicit test-mode Boolean required")
    return context


def _require_not_expired(context,now_epoch):
    _u32(now_epoch,"current authenticated epoch")
    if now_epoch>context["expires_epoch"]:
        raise SessionError("expired reference session")


def proposal_digest(values:Sequence[int], context:dict):
    """Hash proposal + nonce + SOURCE before reference root/challenge reveal."""
    _validate_context(context)
    if len(values)!=context["coefficient_count"]:
        raise SessionError("proposal output length mismatch")
    if any(type(v) is not int for v in values):
        raise SessionError("proposal must use exact signed integer coefficients")
    inner=nor43_producer.sha_of_proposed_output(values)
    return sha256(b"NOR44/source-session-proposal/v1\x00"+
                  canonical(context)+bytes.fromhex(inner)).hexdigest()


def _sign(label,payload,key):
    return {"payload":payload,
            "signature_hex":key.sign(canonical(payload)).hex()}


def _verify_signature(bundle, public_key, label):
    if not isinstance(bundle,dict) or set(bundle)!={"payload","signature_hex"}:
        raise SessionError(f"malformed signed {label}")
    try:
        sig=bytes.fromhex(bundle["signature_hex"])
        key=bytes(public_key)
        if len(sig)!=64 or len(key)!=32 or bundle["signature_hex"]!=sig.hex():
            raise SessionError(f"noncanonical signed {label}")
        Ed25519PublicKey.from_public_bytes(key).verify(
            sig,canonical(bundle["payload"]))
    except (TypeError,ValueError,InvalidSignature) as exc:
        raise SessionError(f"unauthorized or malformed {label}") from exc
    return bundle["payload"]


def verify_precommit_receipt(receipt, pinned_verifier_public_key,
                             expected_context,now_epoch):
    _validate_context(expected_context)
    _require_not_expired(expected_context,now_epoch)
    data=_verify_signature(receipt,pinned_verifier_public_key,
                           "verifier proposal receipt")
    if set(data)!={"protocol","context","proposal_sha256","phase"}:
        raise SessionError("wrong precommit receipt schema")
    if data["protocol"]!=RECEIPT_PROTOCOL or data["phase"]!="proposal-frozen":
        raise SessionError("wrong precommit stage")
    if data["context"]!=expected_context:
        raise SessionError("proposal context/session replay")
    _hex_digest(data["proposal_sha256"],"proposal")
    if public_key_id(pinned_verifier_public_key)!=expected_context["verifier_key_id"]:
        raise SessionError("verifier receipt key ID mismatch")
    return data


def sign_provider_reference(tree, receipt, expected_context,
                            pinned_verifier_public_key,
                            provider_signing_key:Ed25519PrivateKey,now_epoch):
    """Provider verifies prior verifier receipt before signing a true root.

    The caller is responsible for whether the output tree is actually true.
    A dishonest authorized provider can sign a wrong vector and pass auth.
    """
    data=verify_precommit_receipt(
        receipt,pinned_verifier_public_key,expected_context,now_epoch)
    if tree.m!=expected_context["coefficient_count"]:
        raise SessionError("provider tree output length mismatch")
    if public_key_id(public_key_bytes(provider_signing_key))!=(
            expected_context["provider_key_id"]):
        raise SessionError("signer is not the pinned reference provider")
    payload={
        "protocol":REFERENCE_PROTOCOL,
        "context":dict(expected_context),
        "proposal_sha256":data["proposal_sha256"],
        "precommit_receipt_sha256":sha256(canonical(receipt)).hexdigest(),
        "merkle_root_sha256":tree.root.hex()
    }
    return _sign("provider-reference",payload,provider_signing_key)


def verify_provider_reference(bundle,expected_context,receipt,
                              pinned_provider_public_key,
                              pinned_verifier_public_key,now_epoch):
    data=verify_precommit_receipt(
        receipt,pinned_verifier_public_key,expected_context,now_epoch)
    payload=_verify_signature(
        bundle,pinned_provider_public_key,"provider reference")
    if set(payload)!={
        "protocol","context","proposal_sha256",
        "precommit_receipt_sha256","merkle_root_sha256"}:
        raise SessionError("wrong reference manifest schema")
    if payload["protocol"]!=REFERENCE_PROTOCOL:
        raise SessionError("wrong reference protocol domain")
    if payload["context"]!=expected_context:
        raise SessionError("reference context/session replay")
    if public_key_id(pinned_provider_public_key)!=(
            expected_context["provider_key_id"]):
        raise SessionError("reference key not the pinned provider")
    if payload["proposal_sha256"]!=data["proposal_sha256"]:
        raise SessionError("reference rebinds original proposal")
    if payload["precommit_receipt_sha256"]!=sha256(
            canonical(receipt)).hexdigest():
        raise SessionError("reference replays another verifier receipt")
    _hex_digest(payload["merkle_root_sha256"],"Merkle reference root")
    return payload


class VerifierSession:
    """One-use verifier-held session and challenge, with terminal signed receipt."""
    def __init__(self, *, source_sha256,observed_degree,coefficient_count,
                 provider_id,pinned_provider_public_key,
                 verifier_signing_key:Ed25519PrivateKey,
                 expires_epoch,nonce=None,test_only=False):
        if type(test_only) is not bool:
            raise SessionError("test_only must be a Boolean")
        if nonce is not None and not test_only:
            raise SessionError("injected nonce only permitted in test mode")
        nonce=secrets.token_bytes(32) if nonce is None else nonce
        if not isinstance(nonce,bytes) or len(nonce)!=32:
            raise SessionError("expected 32-byte session nonce")
        self.pinned_provider_public_key=bytes(pinned_provider_public_key)
        self.verifier_key=verifier_signing_key
        self.pinned_verifier_public_key=public_key_bytes(verifier_signing_key)
        self.context=_validate_context({
            "protocol":CONTEXT_PROTOCOL,
            "nonce_hex":nonce.hex(),
            "source_sha256":source_sha256,
            "observed_degree":observed_degree,
            "coefficient_count":coefficient_count,
            "provider_id":provider_id,
            "provider_key_id":public_key_id(self.pinned_provider_public_key),
            "verifier_key_id":public_key_id(self.pinned_verifier_public_key),
            "expires_epoch":expires_epoch,
            "test_only":test_only
        })
        self.phase="created"
        self.receipt=None
        self.reference=None
        self.requested_indices=None
        self.challenge_envelope=None

    def _require(self,phase,now_epoch):
        if self.phase!=phase:
            raise SessionError(f"invalid phase: expected {phase}, found {self.phase}")
        _require_not_expired(self.context,now_epoch)

    def precommit(self,model_proposal_sha256, *, now_epoch):
        self._require("created",now_epoch)
        digest=_hex_digest(model_proposal_sha256,"model proposal")
        receipt=_sign("precommit",{
            "protocol":RECEIPT_PROTOCOL,
            "context":dict(self.context),
            "proposal_sha256":digest,
            "phase":"proposal-frozen"
        },self.verifier_key)
        self.receipt=receipt
        self.phase="proposed"
        return receipt

    def authorize_reference(self,provider_bundle, *, now_epoch):
        self._require("proposed",now_epoch)
        try:
            reference=verify_provider_reference(
                provider_bundle,self.context,self.receipt,
                self.pinned_provider_public_key,
                self.pinned_verifier_public_key,now_epoch)
        except Exception:
            self.phase="failed"
            raise
        self.reference=provider_bundle
        self.phase="reference_authenticated"
        return reference

    def challenge(self, *, now_epoch,test_indices=None):
        self._require("reference_authenticated",now_epoch)
        if test_indices is not None and not self.context["test_only"]:
            self.phase="failed"
            raise SessionError("challenge injection forbidden outside test mode")
        if test_indices is None:
            indices=tuple(secrets.SystemRandom().sample(
                range(self.context["coefficient_count"]),QUERY_COUNT))
        else:
            indices=tuple(test_indices)
        if (len(indices)!=QUERY_COUNT or
            any(type(i) is not int or i<0 or
                i>=self.context["coefficient_count"] for i in indices) or
            len(set(indices))!=QUERY_COUNT):
            self.phase="failed"
            raise SessionError("invalid verifier-selected challenge positions")
        payload={
            "protocol":CHALLENGE_PROTOCOL,
            "context":dict(self.context),
            "reference_envelope_sha256":sha256(
                canonical(self.reference)).hexdigest(),
            "requested_indices":list(indices)
        }
        self.requested_indices=indices
        self.challenge_envelope=_sign("challenge",payload,self.verifier_key)
        self.phase="challenged"
        return self.challenge_envelope

    def finish(self,revealed_proposal,openings, *, now_epoch):
        self._require("challenged",now_epoch)
        try:
            if proposal_digest(revealed_proposal,self.context)!=(
                    self.receipt["payload"]["proposal_sha256"]):
                raise SessionError("model revealed a different proposal after true root")
            reference=self.reference["payload"]
            verified_manifest={
                "coefficient_count":self.context["coefficient_count"],
                "merkle_root_sha256":reference["merkle_root_sha256"]
            }
            reject=nor43_verifier.verified_sample_rejection(
                tuple(revealed_proposal),verified_manifest,openings,
                challenged_indices=self.requested_indices,
                expected_query_count=QUERY_COUNT)
        except Exception:
            self.phase="failed"
            raise
        result={
            "protocol":FINISH_PROTOCOL,
            "context":dict(self.context),
            "proposal_sha256":self.receipt["payload"]["proposal_sha256"],
            "precommit_receipt_sha256":sha256(
                canonical(self.receipt)).hexdigest(),
            "provider_reference_sha256":sha256(
                canonical(self.reference)).hexdigest(),
            "challenge_envelope_sha256":sha256(
                canonical(self.challenge_envelope)).hexdigest(),
            "requested_indices":list(self.requested_indices),
            "reject":bool(reject),
            "phase":"finalized"
        }
        self.phase="finalized"
        return _sign("result",result,self.verifier_key)


def verify_terminal_result(bundle,expected_context,pinned_verifier_public_key):
    data=_verify_signature(bundle,pinned_verifier_public_key,"terminal result")
    if set(data)!={"protocol","context","proposal_sha256",
                   "precommit_receipt_sha256","provider_reference_sha256",
                   "challenge_envelope_sha256","requested_indices",
                   "reject","phase"}:
        raise SessionError("invalid terminal receipt schema")
    if (data["protocol"]!=FINISH_PROTOCOL or data["phase"]!="finalized"
            or data["context"]!=expected_context or type(data["reject"]) is not bool):
        raise SessionError("wrong or replayed terminal context")
    for name in ("proposal_sha256","precommit_receipt_sha256",
                 "provider_reference_sha256","challenge_envelope_sha256"):
        _hex_digest(data[name],name)
    indices=data["requested_indices"]
    if not isinstance(indices,list) or len(indices)!=QUERY_COUNT or len(set(
            indices))!=QUERY_COUNT:
        raise SessionError("terminal receipt challenged indices invalid")
    return data
