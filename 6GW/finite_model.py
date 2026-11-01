"""Exact finite-state control quotient reference for 6GW research.

Deliberately synthetic and non-kinetic. No networking, model calls or actuators.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from typing import Hashable, Mapping


@dataclass(frozen=True)
class GovernanceState:
    mandate: bool
    evidence: bool
    quorum: bool


@dataclass(frozen=True)
class FiniteSystem:
    states: tuple[Hashable, ...]
    actions: tuple[str, ...]
    observations: Mapping[Hashable, str]
    transitions: Mapping[tuple[Hashable, str], Hashable]

    def validate(self) -> None:
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("states must be a nonempty set")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be a nonempty set")
        if set(self.observations) != set(self.states):
            raise ValueError("each state needs exactly one observation")
        expected = {(s, a) for s in self.states for a in self.actions}
        if set(self.transitions) != expected:
            raise ValueError("transitions must be total on states x actions")
        if not set(self.transitions.values()).issubset(set(self.states)):
            raise ValueError("transition target lies outside declared domain")


def step(state: GovernanceState, action: str) -> GovernanceState:
    """Allowed interventions only edit one registered governance field."""
    if action == "grant_mandate":
        return GovernanceState(True, state.evidence, state.quorum)
    if action == "revoke_mandate":
        return GovernanceState(False, state.evidence, state.quorum)
    if action == "validate_evidence":
        return GovernanceState(state.mandate, True, state.quorum)
    if action == "invalidate_evidence":
        return GovernanceState(state.mandate, False, state.quorum)
    if action == "assemble_quorum":
        return GovernanceState(state.mandate, state.evidence, True)
    if action == "lose_quorum":
        return GovernanceState(state.mandate, state.evidence, False)
    raise ValueError(f"unregistered action: {action}")


ACTIONS = (
    "grant_mandate", "revoke_mandate", "validate_evidence",
    "invalidate_evidence", "assemble_quorum", "lose_quorum",
)


def governance_system(actions: tuple[str, ...] = ACTIONS) -> FiniteSystem:
    states = tuple(GovernanceState(*values) for values in product((False, True), repeat=3))
    return FiniteSystem(
        states=states,
        actions=actions,
        observations={s: "approve" if s.mandate and s.evidence and s.quorum else "hold" for s in states},
        transitions={(s, a): step(s, a) for s in states for a in actions},
    )


def _canonical(signatures: list[Hashable]) -> tuple[int, ...]:
    labels: dict[Hashable, int] = {}
    return tuple(labels.setdefault(signature, len(labels)) for signature in signatures)


def minimal_partition(system: FiniteSystem) -> tuple[int, ...]:
    """Coarsest exact deterministic quotient preserving observations and all allowed future actions."""
    system.validate()
    position = {state: i for i, state in enumerate(system.states)}
    classes = _canonical([system.observations[s] for s in system.states])
    while True:
        signatures = [
            (classes[i], tuple(classes[position[system.transitions[(s, a)]]] for a in system.actions))
            for i, s in enumerate(system.states)
        ]
        refined = _canonical(signatures)
        if refined == classes:
            return classes
        classes = refined


def quotient_witness(system: FiniteSystem, encoding: Mapping[Hashable, Hashable]):
    """First exact one-step obstruction, or None if the proposed quotient factors."""
    system.validate()
    if set(encoding) != set(system.states):
        raise ValueError("encoding must cover exactly the declared states")
    for i, left in enumerate(system.states):
        for right in system.states[i + 1:]:
            if encoding[left] != encoding[right]:
                continue
            if system.observations[left] != system.observations[right]:
                return ("readout", left, right, None)
            for action in system.actions:
                if encoding[system.transitions[(left, action)]] != encoding[system.transitions[(right, action)]]:
                    return ("successor", left, right, action)
    return None


def demonstrate() -> dict:
    full = governance_system()
    revoke_only = governance_system(("revoke_mandate",))
    labels_full = minimal_partition(full)
    labels_limited = minimal_partition(revoke_only)
    approval_only = {s: full.observations[s] for s in full.states}
    witness = quotient_witness(full, approval_only)
    return {
        "status": "verified_finite_reference_only",
        "states": len(full.states),
        "all_interventions": len(full.actions),
        "exact_classes_all_interventions": len(set(labels_full)),
        "exact_classes_revoke_only": len(set(labels_limited)),
        "approval_only_fails_quotient": witness is not None,
        "witness_action": witness[3] if witness else None,
        "no_model_calls_or_war_simulation": True,
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), sort_keys=True, indent=2))
