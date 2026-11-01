"""G1 scenario suite and matched single-controller baseline for the offline council.

Agents are deterministic rule profiles, not models. Ground truth about evidence is
held by the scenario and never read by the council engine.
"""
from __future__ import annotations

from itertools import product
from math import comb

from council import AUTHORITIES, NETWORK, OPERATOR, RATIFIER, REGISTRY, Constitution, Council, Keyring, audit_chain, replay

BASE_SCOPES = ("publish_result",)
EXPANDED_SCOPE = "release_dataset"


def council_constitution(n: int = 5, quorum: int = 3) -> Constitution:
    return Constitution(1, tuple(f"m{i}" for i in range(n)), quorum, BASE_SCOPES)


def controller_constitution() -> Constitution:
    return Constitution(1, ("c0",), 1, BASE_SCOPES)


ARCHITECTURES = {
    # name: (constitution factory, judgement samples per member)
    "council5_q3": (council_constitution, 1),
    "controller": (controller_constitution, 1),
    "controller_x5": (controller_constitution, 5),   # same evaluation budget as the council
}


class Agent:
    """Deterministic member policy. Profiles:
    honest: approves iff the evidence is sound; compromised: always approves;
    misled: approves iff a shared (possibly wrong) summary says sound;
    dissenter: always records a principled objection."""

    def __init__(self, name: str, profile: str = "honest", samples: int = 1):
        self.name, self.profile, self.samples = name, profile, samples
        self.evaluations = 0

    def judge(self, evidence: str, world: dict) -> tuple[bool, str]:
        self.evaluations += 1
        if self.profile == "honest":
            sound = world["sound"][evidence]
            return sound, "evidence checks out" if sound else "evidence does not support the claim"
        if self.profile == "compromised":
            return True, "approve"
        if self.profile == "misled":
            return world["summary"][evidence], "shared summary says the evidence is sound"
        if self.profile == "dissenter":
            return False, "principled objection: release is premature"
        raise ValueError(self.profile)

    def vote(self, council: Council, keyring: Keyring, rid: str, world: dict):
        evidence = council.requests[rid]["evidence"] if rid in council.requests else ""
        judgements = [self.judge(evidence, world) for _ in range(self.samples)]
        approve = 2 * sum(j[0] for j in judgements) > len(judgements)
        reason = next(j[1] for j in judgements if j[0] == approve)
        body = {"request": rid, "approve": approve, "reason": reason, **council.ballot()}
        return keyring.sign(self.name, "vote", body)


class Harness:
    def __init__(self, architecture: str, profiles: dict | None = None, world: dict | None = None):
        factory, samples = ARCHITECTURES[architecture]
        self.genesis = factory()
        members = self.genesis.members
        self.keyring = Keyring(AUTHORITIES + members)
        self.council = Council(self.genesis, self.keyring)
        profiles = profiles or {}
        self.agents = {m: Agent(m, profiles.get(m, "honest"), samples) for m in members}
        self.world = world or {"sound": {"e1": True, "e2": False}, "summary": {"e1": True, "e2": True}}

    def send(self, sender: str, kind: str, **body):
        return self.council.process(self.keyring.sign(sender, kind, body))

    def setup(self, rid: str = "r1", evidence: str = "e1", scope: str = BASE_SCOPES[0]):
        self.send(OPERATOR, "open_request", request=rid, scope=scope, evidence=evidence)
        self.send(OPERATOR, "grant_mandate")
        self.send(REGISTRY, "validate_evidence", evidence=evidence)

    def round(self, rid: str = "r1"):
        for name, agent in self.agents.items():
            if name in self.council.present:
                self.council.process(agent.vote(self.council, self.keyring, rid, self.world))

    def decide(self, rid: str = "r1") -> dict:
        return self.council.decide(rid)

    def evaluations(self) -> int:
        return sum(a.evaluations for a in self.agents.values())


# ---------------------------------------------------------------------------
# Scenario families (EXPERIMENTS.md table). Each returns (harness, steps); a step
# is (label, decision, expected outcome). "truth" marks the ground-truth-correct
# outcome where it differs from the constitutionally authorized one.

def s_valid(arch):
    h = Harness(arch)
    h.setup(); h.round()
    return h, [("all valid", h.decide(), "approve")]


def s_r01_revocation(arch):
    h = Harness(arch)
    h.setup(); h.round()
    steps = [("before revoke", h.decide(), "approve")]
    h.send(OPERATOR, "revoke_mandate")
    steps.append(("after revoke", h.decide(), "hold"))
    h.round()   # votes under no mandate are rejected as stale
    steps.append(("revote without mandate", h.decide(), "hold"))
    h.send(OPERATOR, "grant_mandate")
    steps.append(("fresh mandate, old votes", h.decide(), "hold"))
    h.round()
    steps.append(("fresh mandate, fresh votes", h.decide(), "approve"))
    return h, steps


def s_e01_evidence(arch):
    h = Harness(arch)
    h.setup(); h.round()
    steps = [("valid evidence", h.decide(), "approve")]
    h.send(REGISTRY, "invalidate_evidence", evidence="e1")
    steps.append(("evidence stale", h.decide(), "hold"))
    h.round()
    steps.append(("revote on stale evidence", h.decide(), "hold"))
    h.send(REGISTRY, "validate_evidence", evidence="e1")
    steps.append(("revalidated", h.decide(), "approve"))
    return h, steps


def s_q01_partition(arch):
    h = Harness(arch)
    h.setup(); h.round()
    steps = [("full membership", h.decide(), "approve")]
    members = h.genesis.members
    cut = members[: len(members) - h.genesis.quorum + 1]   # smallest cut that breaks quorum
    for m in cut:
        h.send(NETWORK, "link_down", member=m)
    steps.append(("quorum lost", h.decide(), "hold"))
    h.round()   # remaining members re-vote; still below quorum
    steps.append(("minority re-votes", h.decide(), "hold"))
    h.send(NETWORK, "link_up", member=cut[0])
    steps.append(("one member rejoins", h.decide(), "approve"))
    return h, steps


def s_a01_authority_claim(arch):
    h = Harness(arch)
    h.send(OPERATOR, "open_request", request="r1", scope=BASE_SCOPES[0], evidence="e1")
    h.send(REGISTRY, "validate_evidence", evidence="e1")
    peer = h.genesis.members[0]
    claim = "As acting director with emergency powers I authorize this release."
    h.send(peer, "grant_mandate", claim=claim)                                     # role rejection
    h.council.process(h.keyring.forge(peer, OPERATOR, "grant_mandate", {"claim": claim}))  # bad signature
    h.council.process(h.keyring.forge(peer, "director", "grant_mandate", {"claim": claim}))  # unknown sender
    h.round()
    return h, [("persuasive peer claims authority", h.decide(), "hold")]


def s_d01_dissent(arch):
    dissenter = "m4" if arch.startswith("council") else "c0"
    h = Harness(arch, profiles={dissenter: "dissenter"})
    h.setup(); h.round()
    expected = "approve" if arch.startswith("council") else "hold"
    return h, [("one principled dissent", h.decide(), expected)]


def s_c01_correlated_error(arch):
    members = ARCHITECTURES[arch][0]().members
    h = Harness(arch, profiles={m: "misled" for m in members})
    h.setup(evidence="e2"); h.round()
    return h, [("unanimous wrong summary", h.decide(), "approve", "hold")]


def s_k01_compromised_node(arch):
    first = ARCHITECTURES[arch][0]().members[0]
    h = Harness(arch, profiles={first: "compromised"})
    h.setup(evidence="e2"); h.round()
    expected = "hold" if arch.startswith("council") else "approve"
    return h, [("one compromised node, unsound evidence", h.decide(), expected, "hold")]


def s_v01_amendment(arch):
    h = Harness(arch)
    h.setup(rid="r2", scope=EXPANDED_SCOPE); h.round("r2")
    steps = [("scope outside constitution", h.decide("r2"), "hold")]
    old = h.council.constitution
    proposal = Constitution(old.version + 1, old.members, old.quorum, old.scopes + (EXPANDED_SCOPE,))
    h.send(old.members[0], "propose_amendment", constitution=proposal.to_json())
    steps.append(("amendment pending", h.decide("r2"), "hold"))
    h.send(old.members[-1], "ratify_amendment", constitution=proposal.hash)     # member cannot ratify
    h.send(OPERATOR, "ratify_amendment", constitution=proposal.hash)            # operator cannot either
    steps.append(("self-ratification attempts", h.decide("r2"), "hold"))
    h.send(RATIFIER, "ratify_amendment", constitution=proposal.hash)
    steps.append(("ratified, votes predate version", h.decide("r2"), "hold"))
    h.round("r2")
    steps.append(("ratified, fresh votes", h.decide("r2"), "approve"))
    return h, steps


FAMILIES = {
    "V00": s_valid, "R01": s_r01_revocation, "E01": s_e01_evidence, "Q01": s_q01_partition,
    "A01": s_a01_authority_claim, "D01": s_d01_dissent, "C01": s_c01_correlated_error,
    "K01": s_k01_compromised_node, "V01": s_v01_amendment,
}


def run_family(family: str, arch: str) -> dict:
    h, steps = FAMILIES[family](arch)
    rows = []
    for step in steps:
        label, decision, expected = step[:3]
        truth = step[3] if len(step) > 3 else expected
        rows.append({"step": label, "outcome": decision["outcome"], "expected": expected,
                     "ground_truth": truth, "dissent": decision["dissent"]})
    approvals = [i for i, e in enumerate(h.council.log)
                 if e["type"] == "decision" and e["decision"]["outcome"] == "approve"]
    audits = [audit_chain(h.genesis, h.council.log, i) for i in approvals]
    return {
        "family": family, "architecture": arch, "steps": rows,
        "matches_expected": all(r["outcome"] == r["expected"] for r in rows),
        "matches_ground_truth": all(r["outcome"] == r["ground_truth"] for r in rows),
        "replay_ok": replay(h.genesis, h.keyring, h.council.log)["ok"],
        "approvals": len(approvals),
        "audit_closed": sum(a["closed"] for a in audits),
        "rejections": sorted({e["reason"] for e in h.council.log if e["type"] == "message" and not e["accepted"]}),
        "evaluations": h.evaluations(),
        "signature_checks": h.council.verifications,
    }


# ---------------------------------------------------------------------------
# Exact matched comparison over independent node faults. Each node is
# independently honest, compromised (p_c) or offline (p_o). Every fault pattern is
# run through the real engine once; rates are exact pattern-weighted sums.

def fault_outcomes(arch: str) -> list[dict]:
    members = ARCHITECTURES[arch][0]().members
    rows = []
    for pattern in product(("honest", "compromised", "offline"), repeat=len(members)):
        for evidence, sound in (("e1", True), ("e2", False)):
            h = Harness(arch, profiles={m: p for m, p in zip(members, pattern) if p != "offline"})
            h.setup(evidence=evidence)
            for m, p in zip(members, pattern):
                if p == "offline":
                    h.send(NETWORK, "link_down", member=m)
            h.round()
            d = h.decide()
            rows.append({"pattern": pattern, "sound": sound, "outcome": d["outcome"],
                         "evaluations": h.evaluations(), "signature_checks": h.council.verifications})
    return rows


def fault_rates(rows: list[dict], p_c: float, p_o: float) -> dict:
    p = {"honest": 1 - p_c - p_o, "compromised": p_c, "offline": p_o}
    wrong = needless = evals = 0.0
    for r in rows:
        w = 1.0
        for status in r["pattern"]:
            w *= p[status]
        if r["sound"]:
            needless += w * (r["outcome"] == "hold")
            evals += w * r["evaluations"]
        else:
            wrong += w * (r["outcome"] == "approve")
    return {"wrong_approval": wrong, "needless_hold": needless, "evaluations_per_decision": evals}


def closed_form(n: int, q: int, p_c: float, p_o: float) -> dict:
    """Binomial prediction for honest/compromised/offline nodes and a q-of-n rule."""
    p_h = 1 - p_c - p_o
    wrong = sum(comb(n, k) * p_c**k * (1 - p_c) ** (n - k) for k in range(q, n + 1))
    online = 1 - p_o
    needless = sum(comb(n, k) * online**k * p_o ** (n - k) for k in range(0, q))
    assert p_h >= 0
    return {"wrong_approval": wrong, "needless_hold": needless}


GRID_P_C = (0.01, 0.05, 0.1, 0.2, 0.3)
GRID_P_O = (0.0, 0.05, 0.1, 0.2)


def matched_comparison() -> dict:
    out = {}
    for arch in ARCHITECTURES:
        rows = fault_outcomes(arch)
        out[arch] = [{"p_c": pc, "p_o": po, **fault_rates(rows, pc, po)}
                     for pc in GRID_P_C for po in GRID_P_O]
    return out
