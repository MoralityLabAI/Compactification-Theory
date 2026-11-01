"""G1 offline council: authenticated mock identities, quorum, revocation,
constitutional versioning and deterministic, hash-chained replay.

Synthetic and non-kinetic. Keys are HMAC secrets derived from a public seed: they
authenticate messages inside the simulation and protect nothing real. The engine
has no clock, randomness, networking or model calls, so a log replays exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import hmac
import json
from typing import Mapping

OPERATOR = "operator"   # opens requests, grants and revokes the mandate
REGISTRY = "registry"   # validates and invalidates evidence
RATIFIER = "ratifier"   # external (human) ratification of constitutional amendments
NETWORK = "network"     # transport layer: reports member links going down or up
AUTHORITIES = (OPERATOR, REGISTRY, RATIFIER, NETWORK)

ROLE_KINDS = {
    OPERATOR: {"open_request", "grant_mandate", "revoke_mandate"},
    REGISTRY: {"validate_evidence", "invalidate_evidence"},
    RATIFIER: {"ratify_amendment"},
    NETWORK: {"link_down", "link_up"},
}
MEMBER_KINDS = {"vote", "propose_amendment"}


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


@dataclass(frozen=True)
class Message:
    kind: str
    sender: str
    body: Mapping
    sig: str

    def to_json(self) -> dict:
        return {"kind": self.kind, "sender": self.sender, "body": dict(self.body), "sig": self.sig}

    @staticmethod
    def from_json(obj: Mapping) -> "Message":
        return Message(obj["kind"], obj["sender"], obj["body"], obj["sig"])


def _payload(kind: str, sender: str, body: Mapping) -> bytes:
    return canonical({"kind": kind, "sender": sender, "body": dict(body)})


class Keyring:
    """Mock identities: one HMAC key per principal, derived from a public seed."""

    def __init__(self, principals, seed: str = "6gw-g1"):
        self._keys = {
            p: hashlib.sha256(f"6gw-mock-key|{seed}|{p}".encode()).digest() for p in principals
        }

    def sign(self, sender: str, kind: str, body: Mapping) -> Message:
        sig = hmac.new(self._keys[sender], _payload(kind, sender, body), hashlib.sha256).hexdigest()
        return Message(kind, sender, dict(body), sig)

    def forge(self, signer: str, claimed_sender: str, kind: str, body: Mapping) -> Message:
        """A message claiming `claimed_sender` but signed with `signer`'s key."""
        sig = hmac.new(self._keys[signer], _payload(kind, claimed_sender, body), hashlib.sha256).hexdigest()
        return Message(kind, claimed_sender, dict(body), sig)

    def verify(self, msg: Message) -> bool:
        key = self._keys.get(msg.sender)
        if key is None:
            return False
        expected = hmac.new(key, _payload(msg.kind, msg.sender, msg.body), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, msg.sig)


@dataclass(frozen=True)
class Constitution:
    version: int
    members: tuple[str, ...]
    quorum: int
    scopes: tuple[str, ...]

    def validate(self) -> None:
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("members must be a nonempty set")
        if set(self.members) & set(AUTHORITIES):
            raise ValueError("members cannot hold an authority identity")
        if not 2 * self.quorum > len(self.members) or self.quorum > len(self.members):
            raise ValueError("quorum must be a strict majority of members")
        if not self.scopes:
            raise ValueError("constitution must permit at least one scope")

    def to_json(self) -> dict:
        return {"version": self.version, "members": list(self.members),
                "quorum": self.quorum, "scopes": list(self.scopes)}

    @staticmethod
    def from_json(obj: Mapping) -> "Constitution":
        return Constitution(int(obj["version"]), tuple(obj["members"]), int(obj["quorum"]), tuple(obj["scopes"]))

    @cached_property
    def hash(self) -> str:
        return digest(self.to_json())


GENESIS = "0" * 64


class Council:
    """Deterministic authority state machine with an append-only, hash-chained log.

    A request is approved exactly when the mandate is active, the request's scope is
    permitted by the ratified constitution, its evidence is currently validated, and
    at least `quorum` currently linked members hold an approving vote bound to the
    current mandate epoch and constitution hash. Every other case is a safe hold.
    """

    def __init__(self, constitution: Constitution, keyring: Keyring, record: bool = True):
        constitution.validate()
        self.genesis = constitution
        self.constitution = constitution
        self.ratified = {constitution.hash: constitution}
        self.keyring = keyring
        self.record = record
        self.mandate_active = False
        self.mandate_epoch = 0
        self.mandate_seq: int | None = None
        self.valid_evidence: dict[str, int] = {}          # evidence id -> seq of validation
        self.present = set(constitution.members)
        self.requests: dict[str, dict] = {}
        self.votes: dict[str, dict[str, dict]] = {}       # request -> member -> vote record
        self.pending: dict[str, Constitution] = {}
        self.log: list[dict] = []
        self.seq = 0
        self.verifications = 0

    # -- log ---------------------------------------------------------------
    def _append(self, entry: dict) -> dict:
        entry = {"seq": self.seq, **entry}
        self.seq += 1
        if self.record:
            entry["prev"] = self.log[-1]["hash"] if self.log else GENESIS
            entry["hash"] = digest(entry)
            self.log.append(entry)
        return entry

    # -- message processing -----------------------------------------------
    def process(self, msg: Message) -> tuple[bool, str | None]:
        reason = self._apply(msg)
        self._append({"type": "message", "message": msg.to_json(),
                      "accepted": reason is None, "reason": reason})
        return reason is None, reason

    def _apply(self, msg: Message) -> str | None:
        members = self.constitution.members
        if msg.sender not in AUTHORITIES and msg.sender not in members:
            return "unknown_sender"
        self.verifications += 1
        if not self.keyring.verify(msg):
            return "bad_signature"
        allowed = ROLE_KINDS.get(msg.sender, MEMBER_KINDS)
        if msg.kind not in allowed:
            return "role"
        if msg.sender in members and msg.sender not in self.present:
            return "partitioned"
        try:
            return getattr(self, "_on_" + msg.kind)(msg)
        except (KeyError, TypeError, ValueError):
            return "malformed"

    def _clear_votes(self) -> None:
        self.votes = {rid: {} for rid in self.votes}

    def _on_open_request(self, msg):
        rid = str(msg.body["request"])
        if rid in self.requests:
            return "duplicate_request"
        self.requests[rid] = {"scope": str(msg.body["scope"]), "evidence": str(msg.body["evidence"])}
        self.votes[rid] = {}

    def _on_grant_mandate(self, msg):
        if not self.mandate_active:
            self.mandate_active = True
            self.mandate_epoch += 1
            self.mandate_seq = self.seq

    def _on_revoke_mandate(self, msg):
        self.mandate_active = False
        self.mandate_seq = None
        self._clear_votes()

    def _on_validate_evidence(self, msg):
        eid = str(msg.body["evidence"])
        self.valid_evidence.setdefault(eid, self.seq)

    def _on_invalidate_evidence(self, msg):
        self.valid_evidence.pop(str(msg.body["evidence"]), None)

    def _on_link_down(self, msg):
        member = msg.body["member"]
        if member not in self.constitution.members:
            return "not_member"
        self.present.discard(member)

    def _on_link_up(self, msg):
        member = msg.body["member"]
        if member not in self.constitution.members:
            return "not_member"
        self.present.add(member)

    def _on_vote(self, msg):
        rid = str(msg.body["request"])
        if rid not in self.requests:
            return "unknown_request"
        if not self.mandate_active or msg.body["epoch"] != self.mandate_epoch:
            return "stale_epoch"
        if msg.body["constitution"] != self.constitution.hash:
            return "stale_constitution"
        if not isinstance(msg.body["approve"], bool):
            return "malformed"
        self.votes[rid][msg.sender] = {"approve": msg.body["approve"],
                                      "reason": str(msg.body.get("reason", "")), "seq": self.seq}

    def _on_propose_amendment(self, msg):
        proposal = Constitution.from_json(msg.body["constitution"])
        proposal.validate()
        if proposal.version != self.constitution.version + 1:
            return "bad_version"
        self.pending[proposal.hash] = proposal

    def _on_ratify_amendment(self, msg):
        proposal = self.pending.pop(msg.body["constitution"], None)
        if proposal is None:
            return "not_pending"
        if proposal.version != self.constitution.version + 1:
            return "bad_version"
        old_members = set(self.constitution.members)
        self.constitution = proposal
        self.ratified[proposal.hash] = proposal
        # Continuing members keep their link state; newly seated members start linked.
        self.present = {m for m in proposal.members if m in self.present or m not in old_members}
        self.pending.clear()
        self._clear_votes()

    # -- decisions --------------------------------------------------------
    def evaluate(self, rid: str) -> dict:
        req = self.requests.get(rid)
        votes = self.votes.get(rid, {})
        approvers = sorted(m for m, v in votes.items() if v["approve"] and m in self.present)
        checks = {
            "request": req is not None,
            "mandate": self.mandate_active,
            "scope": req is not None and req["scope"] in self.constitution.scopes,
            "evidence": req is not None and req["evidence"] in self.valid_evidence,
            "quorum": len(approvers) >= self.constitution.quorum,
        }
        approve = all(checks.values())
        return {
            "request": rid,
            "outcome": "approve" if approve else "hold",
            "checks": checks,
            "approvers": approvers,
            "dissent": sorted([m, v["reason"]] for m, v in votes.items() if not v["approve"]),
            "vote_seqs": {m: votes[m]["seq"] for m in approvers},
            "constitution": self.constitution.hash,
            "version": self.constitution.version,
            "mandate_epoch": self.mandate_epoch if self.mandate_active else None,
            "mandate_seq": self.mandate_seq,
            "evidence_seq": self.valid_evidence.get(req["evidence"]) if req else None,
            "present": sorted(self.present),
        }

    def decide(self, rid: str) -> dict:
        decision = self.evaluate(rid)
        self._append({"type": "decision", "decision": decision})
        return decision

    # -- public state an honest agent needs to address its vote -------------
    def ballot(self) -> dict:
        return {"epoch": self.mandate_epoch, "constitution": self.constitution.hash}


def verify_chain(log: list[dict]) -> int | None:
    """Index of the first entry whose hash link is broken, or None."""
    prev = GENESIS
    for i, entry in enumerate(log):
        body = {k: v for k, v in entry.items() if k != "hash"}
        if entry.get("prev") != prev or digest(body) != entry.get("hash") or entry.get("seq") != i:
            return i
        prev = entry["hash"]
    return None


def replay(genesis: Constitution, keyring: Keyring, log: list[dict]) -> dict:
    """Re-execute a log from the genesis constitution and compare every outcome."""
    broken = verify_chain(log)
    if broken is not None:
        return {"ok": False, "index": broken, "reason": "hash_chain"}
    council = Council(genesis, keyring)
    for i, entry in enumerate(log):
        if entry["type"] == "message":
            accepted, reason = council.process(Message.from_json(entry["message"]))
            if (accepted, reason) != (entry["accepted"], entry["reason"]):
                return {"ok": False, "index": i, "reason": "message_outcome"}
        else:
            if council.decide(entry["decision"]["request"]) != entry["decision"]:
                return {"ok": False, "index": i, "reason": "decision"}
    return {"ok": True, "index": None, "reason": None, "final_hash": council.log[-1]["hash"] if log else GENESIS}


def audit_chain(genesis: Constitution, log: list[dict], index: int) -> dict:
    """Independently reconstruct, from the log alone, the authority behind one approval.

    Returns the missing links; an approval has closed audit when the list is empty.
    """
    entry = log[index]
    decision = entry["decision"]
    missing = []
    accepted = [e for e in log[:index] if e["type"] == "message" and e["accepted"]]
    by_seq = {e["seq"]: e for e in accepted}

    constitutions = {genesis.hash: genesis}
    current = genesis
    for e in accepted:
        m = e["message"]
        if m["kind"] == "propose_amendment":
            c = Constitution.from_json(m["body"]["constitution"])
            constitutions.setdefault(c.hash, c)
        if m["kind"] == "ratify_amendment":
            current = constitutions[m["body"]["constitution"]]
    if decision["constitution"] != current.hash:
        missing.append("constitution")

    mandate = by_seq.get(decision["mandate_seq"])
    if mandate is None or mandate["message"]["kind"] != "grant_mandate":
        missing.append("mandate")
    elif any(e["message"]["kind"] == "revoke_mandate" for e in accepted if e["seq"] > mandate["seq"]):
        missing.append("mandate_revoked")

    request = next((e["message"]["body"] for e in accepted
                    if e["message"]["kind"] == "open_request"
                    and e["message"]["body"]["request"] == decision["request"]), None)
    if request is None:
        missing.append("request")
    else:
        if request["scope"] not in current.scopes:
            missing.append("scope")
        ev = by_seq.get(decision["evidence_seq"])
        if ev is None or ev["message"]["kind"] != "validate_evidence" or ev["message"]["body"]["evidence"] != request["evidence"]:
            missing.append("evidence")
        elif any(e["message"]["kind"] == "invalidate_evidence" and e["message"]["body"]["evidence"] == request["evidence"]
                 for e in accepted if e["seq"] > ev["seq"]):
            missing.append("evidence_invalidated")

    for member in decision["approvers"]:
        vote = by_seq.get(decision["vote_seqs"].get(member))
        if (vote is None or vote["message"]["sender"] != member or vote["message"]["kind"] != "vote"
                or vote["message"]["body"]["approve"] is not True
                or vote["message"]["body"]["constitution"] != decision["constitution"]):
            missing.append(f"vote:{member}")
            continue
        later = [e["message"] for e in accepted if e["seq"] > vote["seq"]]
        if any(m["kind"] in ("revoke_mandate", "ratify_amendment") for m in later):
            missing.append(f"vote_stale:{member}")
        if any(m["kind"] == "vote" and m["sender"] == member and m["body"]["request"] == decision["request"] for m in later):
            missing.append(f"vote_superseded:{member}")
        links = [m["kind"] for m in later if m["kind"] in ("link_down", "link_up") and m["body"]["member"] == member]
        if links and links[-1] == "link_down":
            missing.append(f"partitioned:{member}")
    if len(decision["approvers"]) < current.quorum:
        missing.append("quorum")
    return {"missing": missing, "closed": not missing}
