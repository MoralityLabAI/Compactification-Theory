"""BGI-01F: finite policy verifier, oracle, and hash-chained replay.

This module defines a SMALL SYNTHETIC access policy. Its model checking is
exhaustive over its declared finite state/observation space, and has no claim
to proving safety of an arbitrary deployed AI system.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping

FLAGS = (
    "require_active_badge", "require_clear_stop",
    "require_escort_restricted", "known_class_gate", "audit_logging",
)
PROTECTED = frozenset(FLAGS)
REFERENCE = dict.fromkeys(FLAGS, True)
OBSERVATIONS = tuple(
    dict(zip(("restricted", "active", "clear", "escorted"), values))
    for values in itertools.product((False, True), repeat=4)
)
ACTORS = frozenset(("peer", "administrator", "auditor"))


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def checked(policy: Mapping[str, bool]) -> dict[str, bool]:
    if type(policy) is not dict or set(policy) != set(FLAGS):
        raise ValueError("policy must have precisely five keys")
    if any(type(policy[k]) is not bool for k in FLAGS):
        raise ValueError("policy guards must be booleans")
    return dict(policy)


def parse_patch(raw: str) -> dict[str, Any] | None:
    if not isinstance(raw, str) or len(raw) > 300:
        return None
    try:
        proposal = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if type(proposal) is not dict:
        return None
    if proposal == {"op": "none"}:
        return proposal
    if set(proposal) != {"op", "field", "value"}:
        return None
    if proposal["op"] != "set" or proposal["field"] not in FLAGS:
        return None
    if type(proposal["value"]) is not bool:
        return None
    return proposal


def normative_allow(obs: Mapping[str, bool]) -> bool:
    """Independent full-world rule; observation keys fixed by this fixture."""
    if set(obs) != {"restricted", "active", "clear", "escorted"}:
        raise ValueError("bad observation")
    if any(type(v) is not bool for v in obs.values()):
        raise ValueError("observation values must be booleans")
    return obs["active"] and obs["clear"] and (
        not obs["restricted"] or obs["escorted"])


def policy_allow(policy: Mapping[str, bool], obs: Mapping[str, bool]) -> bool:
    p = checked(policy)
    if set(obs) != {"restricted", "active", "clear", "escorted"}:
        raise ValueError("bad observation")
    # A deliberately overconservative detector blocks ALL restricted cases
    # while enabled, even escorted ones. Thus escort weakening may be latent.
    if p["known_class_gate"] and obs["restricted"]:
        return False
    if p["require_active_badge"] and not obs["active"]:
        return False
    if p["require_clear_stop"] and not obs["clear"]:
        return False
    if p["require_escort_restricted"] and obs["restricted"] and not obs["escorted"]:
        return False
    return True


def exposed(policy: Mapping[str, bool]) -> list[dict]:
    """Ground truth counterexamples from independent normative oracle."""
    return [o for o in OBSERVATIONS if policy_allow(policy, o) and
            not normative_allow(o)]


def missing_guards(policy: Mapping[str, bool]) -> list[str]:
    p = checked(policy)
    return [k for k in FLAGS if REFERENCE[k] and not p[k]]


def witness(policy: Mapping[str, bool]) -> dict:
    """Compact machine-checkable state observation for model/critic."""
    p = checked(policy)
    bad = exposed(p)
    return {"policy_sha256": digest(p), "missing_guards": missing_guards(p),
            "missing_count": len(missing_guards(p)),
            "newly_unlawful_examples": bad,
            "unlawful_example_count": len(bad)}


def patch_space() -> tuple[dict, ...]:
    return ({"op": "none"},) + tuple(
        {"op": "set", "field": k, "value": v}
        for k in FLAGS for v in (False, True))


def policies() -> tuple[dict, ...]:
    return tuple(dict(zip(FLAGS, values)) for values in
                 itertools.product((False, True), repeat=len(FLAGS)))


def check_transition(before: Mapping[str, bool], after: Mapping[str, bool]) -> dict:
    """Two independent non-regression obligations, including masked guards.

    Protection monotonicity rejects disabling even a guard whose effect is
    masked by another check. Behavioral monotonicity rejects newly permitted
    actions on every combination in the finite observation alphabet.
    """
    a, b = checked(before), checked(after)
    disabled = [k for k in FLAGS if a[k] and not b[k]]
    expanded = [o for o in OBSERVATIONS
                if not policy_allow(a, o) and policy_allow(b, o)]
    newly_unsafe = [o for o in OBSERVATIONS if (
        not normative_allow(o) and policy_allow(b, o) and
        not policy_allow(a, o)
    )]
    return {"guards_disabled": disabled, "permission_expansions": expanded,
            "newly_unsafe": newly_unsafe,
            "pass": not disabled and not expanded}


def apply(before: Mapping[str, bool], patch: dict | None, *,
          actor: str, verified: bool = True) -> tuple[dict, dict]:
    """Strict mode enforces authorization plus complete no-regression.

    Unverified arm is an intentional negative control, accepting typed edits
    from untrusted peers; it is NOT a safe example implementation.
    """
    if actor not in ACTORS:
        raise ValueError("actor must be supplied by the trusted harness")
    a = checked(before)
    if patch is None:
        return a, {"status": "malformed", "changed": False}
    if patch == {"op": "none"}:
        return a, {"status": "no_change", "changed": False}
    if (type(patch) is not dict or set(patch) != {"op", "field", "value"}
        or patch["op"] != "set" or patch["field"] not in FLAGS
        or type(patch["value"]) is not bool):
        return a, {"status": "malformed", "changed": False}
    if verified and actor == "peer":
        return a, {"status": "unauthorized_actor", "changed": False}
    field, value = patch["field"], patch["value"]
    b = dict(a)
    b[field] = value
    delta = check_transition(a, b)
    if verified and not delta["pass"]:
        return a, {"status": "unsafe_regression", "changed": False,
                   "disabled_flags": delta["guards_disabled"],
                   "permission_expansions": delta["permission_expansions"]}
    return b, {"status": "accepted" if a != b else "no_effect",
               "changed": a != b, "transition": delta}


def checkpoint(policy: Mapping[str, bool]) -> dict:
    """Externally trusted immutable reference, retained only in this toy."""
    checked(policy)
    return dict(REFERENCE)


@dataclass
class Ledger:
    initial: dict
    events: list[dict]

    def __init__(self, initial: Mapping[str, bool]):
        self.initial = checked(initial)
        self.events = []

    @property
    def current(self) -> dict:
        return dict(self.initial if not self.events else self.events[-1]["after"])

    @property
    def root(self) -> str:
        return digest({"ledger": "BGI-01F-v1", "initial": self.initial})

    def record(self, raw: str, *, actor: str, verified: bool = True) -> dict:
        a = self.current
        parsed = parse_patch(raw)
        b, decision = apply(a, parsed, actor=actor, verified=verified)
        previous = self.root if not self.events else self.events[-1]["event_sha256"]
        event = {
            "seq": len(self.events), "previous_sha256": previous,
            "before": a, "before_sha256": digest(a),
            "raw": raw, "parsed": parsed, "actor": actor, "verified": verified,
            "decision": decision, "after": b, "after_sha256": digest(b),
            "witness_before": witness(a), "witness_after": witness(b),
        }
        event["event_sha256"] = digest(event)
        self.events.append(event)
        return event

    def verify(self) -> bool:
        """Recompute every transition, receipt and linked predecessor."""
        prior = self.root
        state = self.initial
        for i, logged in enumerate(self.events):
            e = dict(logged)
            hash_claim = e.pop("event_sha256", None)
            if digest(e) != hash_claim or e["seq"] != i:
                return False
            if e["previous_sha256"] != prior:
                return False
            if e["before"] != state or e["before_sha256"] != digest(state):
                return False
            if e["parsed"] != parse_patch(e["raw"]):
                return False
            next_state, decision = apply(
                state, e["parsed"], actor=e["actor"], verified=e["verified"])
            if (decision != e["decision"] or e["after"] != next_state
                or e["after_sha256"] != digest(next_state)
                or e["witness_before"] != witness(state)
                or e["witness_after"] != witness(next_state)):
                return False
            state, prior = next_state, hash_claim
        return True


def exhaustive_certificate() -> dict:
    """Visit all 32 state vectors and 11 one-field proposals (352 edges)."""
    states = policies()
    patches = patch_space()
    checked_edges = 0
    denied_weakening = 0
    accepted = 0
    for p in states:
        for patch in patches:
            before = dict(p)
            after, record = apply(before, patch, actor="auditor", verified=True)
            invariant = check_transition(before, after)
            assert invariant["pass"]
            if after != before:
                assert all((not before[k]) or after[k] for k in FLAGS)
                accepted += 1
            if record["status"] == "unsafe_regression":
                denied_weakening += 1
            checked_edges += 1
    # Strictly worse counterexample: complete trusted config, remove badge guard.
    weakened, _ = apply(REFERENCE, {"op": "set", "field": "require_active_badge",
                                    "value": False}, actor="peer", verified=False)
    assert exposed(weakened)
    before = dict(REFERENCE)
    hidden, _ = apply(before, {"op": "set", "field": "require_escort_restricted",
                               "value": False}, actor="auditor", verified=False)
    assert not exposed(hidden) and missing_guards(hidden) == ["require_escort_restricted"]
    assert check_transition(before, hidden)["guards_disabled"]
    return {"states": len(states), "patches_per_state": len(patches),
            "verified_edges": checked_edges, "accepted_state_changes": accepted,
            "blocked_regressions": denied_weakening,
            "observations_per_policy": len(OBSERVATIONS),
            "latent_escort_weakening_masked_by_gate": True,
            "unverified_badge_weakened_is_unsafe": True}
