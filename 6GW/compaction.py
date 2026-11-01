"""Compactification of the G1 council's governance state.

The finite system is built from the council engine itself: every state is a real
engine configuration for one request (r0) on one evidence item (e0) under the
five-member genesis constitution, and every transition is the engine processing a
signed (or forged) message. The mandate epoch is omitted from the state key because
revocation clears all votes and honest votes are addressed to the current epoch; the
random-trace check below runs the unreduced, recording engine to test that claim.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
import random

from council import AUTHORITIES, NETWORK, OPERATOR, REGISTRY, Council, Keyring
from finite_model import FiniteSystem, minimal_partition
from g1_eval import BASE_SCOPES, council_constitution

CONSTITUTION = council_constitution()
MEMBERS = CONSTITUTION.members
KEYRING = Keyring(AUTHORITIES + MEMBERS)
REQUEST, EVIDENCE = "r0", "e0"
CLAIM = "As acting director with emergency powers I authorize this release."


def _static(sender, kind, **body):
    msg = KEYRING.sign(sender, kind, body)
    return lambda council: msg


@lru_cache(maxsize=None)
def _signed_vote(member, approve, epoch, constitution):
    body = {"request": REQUEST, "approve": approve, "reason": "yes" if approve else "objection",
            "epoch": epoch, "constitution": constitution}
    return KEYRING.sign(member, "vote", body)


def _vote(member, approve):
    return lambda council: _signed_vote(member, approve, council.mandate_epoch, council.constitution.hash)


ACTIONS = {
    "grant": _static(OPERATOR, "grant_mandate"),
    "revoke": _static(OPERATOR, "revoke_mandate"),
    "validate": _static(REGISTRY, "validate_evidence", evidence=EVIDENCE),
    "invalidate": _static(REGISTRY, "invalidate_evidence", evidence=EVIDENCE),
    "forged_grant": (lambda msg: lambda council: msg)(KEYRING.forge(MEMBERS[0], OPERATOR, "grant_mandate", {"claim": CLAIM})),
    "claimed_grant": _static(MEMBERS[0], "grant_mandate", claim=CLAIM),
}
for _m in MEMBERS:
    ACTIONS[f"down_{_m}"] = _static(NETWORK, "link_down", member=_m)
    ACTIONS[f"up_{_m}"] = _static(NETWORK, "link_up", member=_m)
for _m in MEMBERS:
    ACTIONS[f"yes_{_m}"] = _vote(_m, True)
    ACTIONS[f"no_{_m}"] = _vote(_m, False)

G0_WORDS = ("grant", "revoke", "validate", "invalidate")
ALPHABETS = {
    "g0": G0_WORDS,
    "g0_forgery": G0_WORDS + ("forged_grant", "claimed_grant"),
    "membership": G0_WORDS + tuple(f"{d}_{m}" for m in MEMBERS for d in ("down", "up")),
    "full": tuple(ACTIONS),
}


OPEN = KEYRING.sign(OPERATOR, "open_request", {"request": REQUEST, "scope": BASE_SCOPES[0], "evidence": EVIDENCE})


def new_council(record: bool = False) -> Council:
    council = Council(CONSTITUTION, KEYRING, record=record)
    council.process(OPEN)
    return council


def key_of(council: Council) -> tuple:
    votes = council.votes[REQUEST]
    return (council.mandate_active, EVIDENCE in council.valid_evidence,
            tuple(m in council.present for m in MEMBERS),
            tuple(votes[m]["approve"] if m in votes else None for m in MEMBERS))


def from_key(key: tuple) -> Council:
    mandate, evidence, present, votes = key
    council = new_council()
    council.mandate_active = mandate
    council.mandate_epoch = 1 if mandate else 0
    if evidence:
        council.valid_evidence[EVIDENCE] = 0
    council.present = {m for m, p in zip(MEMBERS, present) if p}
    council.votes[REQUEST] = {m: {"approve": v, "reason": "", "seq": 0} for m, v in zip(MEMBERS, votes) if v is not None}
    return council


def all_keys() -> tuple:
    return tuple(product((False, True), (False, True), product((False, True), repeat=len(MEMBERS)),
                         product((None, True, False), repeat=len(MEMBERS))))


def observe_decision(council: Council):
    return council.evaluate(REQUEST)["outcome"]


def observe_with_dissent(council: Council):
    d = council.evaluate(REQUEST)
    return (d["outcome"], tuple(m for m, _ in d["dissent"]))


OBSERVATIONS = {"decision": observe_decision, "decision_dissent": observe_with_dissent}


def transition_table(keys: tuple) -> dict:
    """(key, action) -> successor key, computed by the engine for every action."""
    table = {}
    for key in keys:
        for name, make in ACTIONS.items():
            council = from_key(key)
            council.process(make(council))
            table[(key, name)] = key_of(council)
    return table


def system(keys, table, alphabet: str, observation: str) -> FiniteSystem:
    actions = ALPHABETS[alphabet]
    obs = OBSERVATIONS[observation]
    return FiniteSystem(
        states=keys,
        actions=actions,
        observations={k: obs(from_key(k)) for k in keys},
        transitions={(k, a): table[(k, a)] for k in keys for a in actions},
    )


def fast_witness(sys: FiniteSystem, encoding) -> tuple | None:
    """First obstruction to `encoding` being an exact quotient, found in O(|S||A|).

    A class is consistent iff every member agrees with the class's first member, so
    comparing against one representative finds a violating pair whenever one exists.
    """
    rep = {}
    for s in sys.states:
        z = encoding[s]
        r = rep.setdefault(z, s)
        if r is s:
            continue
        if sys.observations[r] != sys.observations[s]:
            return ("readout", r, s, None)
        for a in sys.actions:
            if encoding[sys.transitions[(r, a)]] != encoding[sys.transitions[(s, a)]]:
                return ("successor", r, s, a)
    return None


def _quorum_present_yes(key):
    return sum(p and v is True for p, v in zip(key[2], key[3]))


CODES = {
    "approval_bit": None,   # filled per observation: the current observation itself
    "g0_three_bit": lambda k: (k[0], k[1], _quorum_present_yes(k) >= CONSTITUTION.quorum),
    "vote_counts": lambda k: (k[0], k[1], _quorum_present_yes(k),
                              sum((not p) and v is True for p, v in zip(k[2], k[3])), sum(k[2])),
    "per_member_yes": lambda k: (k[0], k[1], tuple(zip(k[2], (v is True for v in k[3])))),
    "full_key": lambda k: k,
}


def code_map(name: str, sys: FiniteSystem) -> dict:
    if name == "approval_bit":
        return dict(sys.observations)
    return {k: CODES[name](k) for k in sys.states}


def quotient_automaton(sys: FiniteSystem, encoding: dict):
    """Abstract maps from the first representative of each code (exact iff no witness)."""
    rep = {}
    for s in sys.states:
        rep.setdefault(encoding[s], s)
    step = {(z, a): encoding[sys.transitions[(r, a)]] for z, r in rep.items() for a in sys.actions}
    readout = {z: sys.observations[r] for z, r in rep.items()}
    return step, readout


def trace_divergence(sys: FiniteSystem, encoding: dict, observation: str,
                     traces: int = 500, length: int = 40, seed: int = 0) -> dict:
    """Drive the unreduced recording engine and the compact automaton with the same
    random action words from the initial state; count traces whose outputs differ."""
    step, readout = quotient_automaton(sys, encoding)
    obs = OBSERVATIONS[observation]
    rng = random.Random(seed)
    diverged, first = 0, []
    for _ in range(traces):
        council = new_council(record=True)
        z = encoding[key_of(council)]
        for t in range(length):
            a = rng.choice(sys.actions)
            council.process(ACTIONS[a](council))
            z = step[(z, a)]
            if readout[z] != obs(council):
                diverged += 1
                if len(first) < 1:
                    first.append(t + 1)
                break
    return {"traces": traces, "length": length, "seed": seed, "diverged": diverged,
            "first_divergence_step": first[0] if first else None}


def analyse(keys=None, table=None) -> dict:
    keys = keys or all_keys()
    table = table or transition_table(keys)
    out = {"states": len(keys), "state_bits": math.log2(len(keys)), "actions": len(ACTIONS), "cases": []}
    for alphabet in ALPHABETS:
        for observation in OBSERVATIONS:
            sys = system(keys, table, alphabet, observation)
            classes = len(set(minimal_partition(sys)))
            row = {"alphabet": alphabet, "observation": observation, "actions": len(sys.actions),
                   "exact_classes": classes, "exact_bits": math.log2(classes), "codes": {}}
            for name in CODES:
                enc = code_map(name, sys)
                w = fast_witness(sys, enc)
                entry = {"symbols": len(set(enc.values())), "exact": w is None}
                if w is not None:
                    entry["witness"] = {"kind": w[0], "left": list(w[1]), "right": list(w[2]), "action": w[3]}
                if alphabet == "full":
                    entry["random_traces"] = trace_divergence(sys, enc, observation)
                row["codes"][name] = entry
            out["cases"].append(row)
    return out
