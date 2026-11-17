"""BGI-01: finite, explicitly synthetic beneficial-drive stress test.

Only Python standard library is required. This is a causal *toy model*, not an
implementation or evaluation of Omega, Hyperon, a trained LLM, or a moral agent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Iterable


@dataclass(frozen=True)
class Arm:
    name: str
    repair: str  # none | quorum | backup
    gate: str  # independent | mutable | absent
    detect_novel: bool


ARMS = (
    Arm("A_fixed_external", "none", "independent", False),
    Arm("B_developmental_mutable", "quorum", "mutable", True),
    Arm("C_developmental_unguarded", "quorum", "absent", True),
    Arm("D_hybrid_independent", "backup", "independent", True),
)


@dataclass(frozen=True)
class State:
    supports: int
    gate_intact: bool = True


@dataclass(frozen=True)
class Shock:
    hazard: str = "benign"  # benign | known | novel
    pressure: int = 0
    erased: int = 0
    rewrite_gate: bool = False
    peer_corrupted: bool = False


@dataclass(frozen=True)
class Step:
    state: State
    recommended: str
    emitted: str
    violated: bool
    vetoed: bool


def validate(n: int, quorum: int) -> None:
    if not (1 <= quorum <= n <= 12):
        raise ValueError("Require 1 <= quorum <= n <= 12")


def repair(mask: int, n: int, quorum: int, mechanism: str, peer_corrupted: bool = False) -> int:
    validate(n, quorum)
    if mask < 0 or mask >= (1 << n):
        raise ValueError("support mask outside n-bit domain")
    if mechanism not in {"none", "quorum", "backup"}:
        raise ValueError("unknown repair mechanism")
    if mechanism == "none" or peer_corrupted:
        return mask
    if mechanism == "quorum" and mask.bit_count() < quorum:
        return mask
    for i in range(n):
        if not (mask & (1 << i)):
            return mask | (1 << i)  # one rebuilt support per round
    return mask


def transition(state: State, shock: Shock, arm: Arm, n: int = 4, quorum: int = 2) -> Step:
    validate(n, quorum)
    if not (0 <= state.supports < 1 << n) or shock.erased & ~((1 << n) - 1):
        raise ValueError("bad support mask")
    if shock.hazard not in {"benign", "known", "novel"} or not (0 <= shock.pressure <= n):
        raise ValueError("bad hazard or pressure")
    mask = state.supports & ~shock.erased
    gate_ok = state.gate_intact and not (arm.gate == "mutable" and shock.rewrite_gate)
    mask = repair(mask, n, quorum, arm.repair, shock.peer_corrupted)
    recognized = (shock.hazard == "known") or (
        shock.hazard == "novel" and arm.detect_novel and not shock.peer_corrupted
        and mask.bit_count() >= quorum
    )
    # A soft drive competes with instrumental pressure; a hard gate can veto it.
    recommended = "serve" if shock.hazard == "benign" else (
        "decline" if recognized and mask.bit_count() > shock.pressure else "unsafe"
    )
    veto = recommended == "unsafe" and (
        (arm.gate == "independent" and shock.hazard == "known") or
        (arm.gate == "mutable" and gate_ok and recognized)
    )
    emitted = "decline" if veto else recommended
    return Step(State(mask, gate_ok), recommended, emitted,
                shock.hazard != "benign" and emitted == "unsafe", veto)


def recovery_steps(state: State, arm: Arm, n: int = 4, quorum: int = 2) -> int | None:
    """Time to recover *all* supports during unopposed benign rounds."""
    validate(n, quorum)
    s = state
    for t in range(n + 1):
        if s.supports == (1 << n) - 1:
            return t
        s = transition(s, Shock(), arm, n, quorum).state
    return None


def basin(arm: Arm, n: int = 4, quorum: int = 2) -> dict:
    """Recovery basin over uniform support masks, with intact initial gate."""
    validate(n, quorum)
    ok = [mask for mask in range(1 << n) if recovery_steps(State(mask), arm, n, quorum) is not None]
    return {"recoverable_masks": len(ok), "total_masks": 1 << n,
            "fraction": str(Fraction(len(ok), 1 << n)),
            "max_recovery_rounds": max(recovery_steps(State(m), arm, n, quorum) for m in ok)}


def basin_probability(arm: Arm, n: int = 4, quorum: int = 2,
                      initial_support_probability: Fraction = Fraction(1, 2)) -> Fraction:
    """Exact basin mass under independent Bernoulli priors on initial supports.

    The weighting is user-chosen; basin *cardinality* is not an intrinsic
    physical probability for a randomly initialized intelligent agent.
    """
    validate(n, quorum)
    p = initial_support_probability
    if not 0 <= p <= 1:
        raise ValueError("prior probability outside [0,1]")
    return sum((p ** m.bit_count() * (1 - p) ** (n - m.bit_count())
                for m in range(1 << n)
                if recovery_steps(State(m), arm, n, quorum) is not None), Fraction())


def shocks(n: int = 4, *, novel: bool = False, single_erasure: bool = True) -> tuple[Shock, ...]:
    """Prespecified finite adversary alphabet, one erase bit per round."""
    hazards = ("known", "novel") if novel else ("known",)
    erasures = (0,) + tuple(1 << i for i in range(n)) if single_erasure else tuple(range(1 << n))
    return tuple(Shock(h, p, e, rewrite)
                 for h, p, e, rewrite in product(hazards, (0, n), erasures, (False, True)))


def safety_kernel(arm: Arm, attacks: Iterable[Shock], n: int = 4,
                  quorum: int = 2) -> frozenset[State]:
    """Greatest invariant set with no emitted harm under every attack."""
    validate(n, quorum)
    alphabet = tuple(attacks)
    if not alphabet:
        raise ValueError("empty adversary alphabet")
    all_states = frozenset(State(m, b) for m in range(1 << n) for b in (False, True))
    region = all_states
    while True:
        keep = frozenset(s for s in region if all(
            not (out := transition(s, w, arm, n, quorum)).violated and out.state in region
            for w in alphabet))
        if keep == region:
            return keep
        region = keep


def shortest_violation(initial: State, arm: Arm, attacks: Iterable[Shock],
                       n: int = 4, quorum: int = 2, max_steps: int = 4) -> list[dict] | None:
    """Breadth-first shortest failing witness. Attacks sorted by caller ordering."""
    alphabet = tuple(attacks)
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, trace = queue.popleft()
        if len(trace) >= max_steps:
            continue
        for attack in alphabet:
            out = transition(state, attack, arm, n, quorum)
            node = {"before": asdict(state), "shock": asdict(attack),
                    "after": asdict(out.state), "recommended": out.recommended,
                    "emitted": out.emitted, "violated": out.violated,
                    "vetoed": out.vetoed}
            path = trace + [node]
            if out.violated:
                return path
            if out.state not in seen:
                seen.add(out.state)
                queue.append((out.state, path))
    return None


def first_quorum_loss_probability(arm: Arm, *, n: int = 4, quorum: int = 2,
                                  rounds: int = 8, per_support_loss: Fraction = Fraction(1, 4)) -> Fraction:
    """Exact independent-loss process; first passage below quorum AFTER repair.

    This is a declared stochastic assumption, not an adversarial guarantee.
    Returns event P(exists t <= rounds: supports(t) < quorum).
    """
    validate(n, quorum)
    if rounds < 0 or not (0 <= per_support_loss <= 1):
        raise ValueError("bad time or loss probability")
    distribution: dict[tuple[int, bool], Fraction] = {((1 << n) - 1, False): Fraction(1)}
    for _ in range(rounds):
        nxt: dict[tuple[int, bool], Fraction] = {}
        for (mask, lost_before), mass in distribution.items():
            active = tuple(i for i in range(n) if mask & (1 << i))
            for k in range(len(active) + 1):
                probability = per_support_loss ** k * (1 - per_support_loss) ** (len(active) - k)
                if not probability:
                    continue
                for indices in combinations(active, k):
                    erased = sum(1 << i for i in indices)
                    new_mask = repair(mask & ~erased, n, quorum, arm.repair)
                    key = (new_mask, lost_before or new_mask.bit_count() < quorum)
                    nxt[key] = nxt.get(key, Fraction()) + mass * probability
        distribution = nxt
    assert sum(distribution.values(), Fraction()) == 1
    return sum((mass for (mask, happened), mass in distribution.items() if happened), Fraction())


def quorum_margin_witness(n: int = 4, quorum: int = 2) -> dict:
    """Exhaustively check robust forward invariance of count >= q+1 under <=1 loss.

    Requires self-regeneration, uncorrupted peers, and one repair after each loss.
    """
    validate(n, quorum)
    if quorum == n:
        raise ValueError("q+1 must fit in n")
    good = [m for m in range(1 << n) if m.bit_count() >= quorum + 1]
    checked = 0
    for mask in good:
        for erased in (0,) + tuple(1 << i for i in range(n)):
            after = repair(mask & ~erased, n, quorum, "quorum")
            assert after.bit_count() >= quorum + 1
            checked += 1
    # A state with exactly q supports is vulnerable to a single targeted deletion.
    boundary = (1 << quorum) - 1
    after_boundary = repair(boundary & ~1, n, quorum, "quorum")
    assert after_boundary.bit_count() == quorum - 1
    return {"robust_support_threshold": quorum + 1, "checked_edges": checked,
            "boundary_counterexample": {"start": boundary, "erased": 1,
                                        "end": after_boundary}}
