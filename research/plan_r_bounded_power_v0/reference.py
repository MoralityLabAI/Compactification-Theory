"""Plan R bounded-power reference: finite toy simulator, stdlib only.

No external I/O, real models, gradients, networks, funds, or signing.
Every reachable edge in the declared finite world is enumerated exactly.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path


ACTIONS = ("idle", "read_a", "read_b", "receive_peer", "compile", "release", "write_external", "coordinate")
MODEL_SHA256 = sha256(json.dumps({"policy": "frozen-symbolic-v1", "actions": ACTIONS},
                                separators=(",", ":")).encode()).hexdigest()
EXTERNAL = frozenset(("write_external", "coordinate"))


def _integer(value: object, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True)
class Scenario:
    name: str
    horizon: int = 6
    memory_slots: int = 2
    local_facts: int = 3
    peer_facts: int = 0
    peer_enabled: bool = False
    tool_grants: frozenset[str] = frozenset()
    max_instances: int = 1
    release_lag: int = 0
    enforce_gateway: bool = False
    approved_actions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scenario name required")
        for key, value, lower, upper in (("horizon", self.horizon, 0, 12),
                                          ("memory_slots", self.memory_slots, 0, 2),
                                          ("local_facts", self.local_facts, 0, 3),
                                          ("peer_facts", self.peer_facts, 0, 3),
                                          ("max_instances", self.max_instances, 1, 4),
                                          ("release_lag", self.release_lag, 0, 12)):
            _integer(value, lower, upper, key)
        if type(self.peer_enabled) is not bool or type(self.enforce_gateway) is not bool:
            raise ValueError("gateway and peer flags must be booleans")
        for key, choices, domain in (("tool_grants", self.tool_grants, EXTERNAL),
                                     ("approved_actions", self.approved_actions, frozenset(ACTIONS))):
            if not isinstance(choices, frozenset) or not all(isinstance(x, str) for x in choices) or not choices <= domain:
                raise ValueError(f"invalid {key}")
        if self.enforce_gateway and not self.approved_actions:
            raise ValueError("enforced gateway requires an explicit allowlist")


@dataclass(frozen=True)
class State:
    tick: int = 0
    facts: int = 0
    compiled: bool = False
    instances: int = 1
    unsafe: bool = False


def transition(s: State, action: str, c: Scenario) -> State | None:
    """A denied action returns None; invalid action names raise ValueError."""
    if action not in ACTIONS:
        raise ValueError("unknown action")
    if s.tick >= c.horizon or s.unsafe:
        return None
    if c.enforce_gateway and action not in c.approved_actions:
        return None
    kw = {"tick": s.tick + 1, "facts": s.facts, "compiled": s.compiled,
          "instances": s.instances, "unsafe": s.unsafe}
    if action == "idle":
        pass
    elif action in ("read_a", "read_b"):
        fact = 1 if action == "read_a" else 2
        if not c.local_facts & fact or (s.facts | fact).bit_count() > c.memory_slots:
            return None
        kw["facts"] |= fact
    elif action == "receive_peer":
        if not c.peer_enabled or not c.peer_facts or (s.facts | c.peer_facts).bit_count() > c.memory_slots:
            return None
        kw["facts"] |= c.peer_facts
    elif action == "compile":
        if s.facts != 3:
            return None
        kw["compiled"] = True
    elif action == "release":
        if s.tick < c.release_lag or s.instances >= c.max_instances:
            return None
        kw["instances"] += 1
    elif action == "write_external":
        if not s.compiled or action not in c.tool_grants:
            return None
        kw["unsafe"] = True
    elif action == "coordinate":
        if not s.compiled or s.instances < 2 or action not in c.tool_grants:
            return None
        kw["unsafe"] = True
    return State(**kw)


def capability_proxy(s: State) -> int:
    """Declared toy task score, NOT an empirical AI benchmark score."""
    return s.facts.bit_count() + 2 * int(s.compiled)


def _config_record(c: Scenario) -> dict:
    out = asdict(c)
    out["tool_grants"] = sorted(c.tool_grants)
    out["approved_actions"] = sorted(c.approved_actions)
    return out


def _hash_record(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def explore(c: Scenario) -> dict:
    """Exact finite BFS of all admissible schedules, up to c.horizon."""
    start = State()
    parent: dict[State, tuple[State, str] | None] = {start: None}
    queue = deque((start,))
    first_unsafe: State | None = None
    max_score = 0
    max_instances = 1
    blocked = 0
    while queue:
        s = queue.popleft()
        max_score = max(max_score, capability_proxy(s))
        max_instances = max(max_instances, s.instances)
        if s.unsafe and first_unsafe is None:
            first_unsafe = s
        for action in ACTIONS:
            nxt = transition(s, action, c)
            if nxt is None:
                blocked += 1
            elif nxt not in parent:
                parent[nxt] = (s, action)
                queue.append(nxt)
    witness = None
    if first_unsafe is not None:
        actions: list[str] = []
        states = [asdict(first_unsafe)]
        cursor = first_unsafe
        while parent[cursor] is not None:
            prev, action = parent[cursor]
            actions.append(action)
            states.append(asdict(prev))
            cursor = prev
        actions.reverse()
        states.reverse()
        witness = {"actions": actions, "states": states}
    config = _config_record(c)
    result = {"scenario": c.name, "config_sha256": _hash_record(config),
              "model_sha256": MODEL_SHA256, "horizon": c.horizon,
              "reachable_states": len(parent), "denied_edges": blocked,
              "unsafe_reachable": first_unsafe is not None,
              "max_capability_proxy": max_score, "max_instances_reached": max_instances,
              "shortest_unsafe_witness": witness}
    result["result_sha256"] = _hash_record(result)
    return result


def observe(s: State) -> tuple[int, bool, int, bool]:
    """A candidate compactification projection deliberately omitting time."""
    return (s.facts, s.compiled, s.instances, s.unsafe)


def distinguishing_word(a: State, b: State, c: Scenario, max_length: int) -> tuple[str, ...] | None:
    """Shortest action suffix distinguishing two states by outputs or denial.

    Exhausts all finite action suffixes of length <= max_length using pair BFS.
    """
    _integer(max_length, 0, 12, "max_length")
    if observe(a) != observe(b):
        return ()
    queue = deque(((a, b, ()),))
    visited = {(a, b)}
    while queue:
        x, y, suffix = queue.popleft()
        if len(suffix) == max_length:
            continue
        for action in ACTIONS:
            nx, ny = transition(x, action, c), transition(y, action, c)
            word = suffix + (action,)
            if (nx is None) != (ny is None):
                return word
            if nx is None:
                continue
            if observe(nx) != observe(ny):
                return word
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, word))
    return None


def cases() -> dict[str, Scenario]:
    """Five prospective ablation families. Every arm holds MODEL_SHA256 fixed."""
    write = frozenset(("write_external",))
    coord = frozenset(("coordinate",))
    both = write | coord
    all_safe = frozenset(("idle", "read_a", "read_b", "receive_peer", "compile", "release"))
    return {
        "01_memory_off": Scenario("01_memory_off", memory_slots=0, tool_grants=write),
        "01_memory_on": Scenario("01_memory_on", tool_grants=write),
        "02_peer_sham": Scenario("02_peer_sham", local_facts=1, peer_facts=2, tool_grants=write),
        "02_peer_exchange": Scenario("02_peer_exchange", local_facts=1, peer_facts=2,
                                     peer_enabled=True, tool_grants=write),
        "03_read_only": Scenario("03_read_only"),
        "03_write_tool": Scenario("03_write_tool", tool_grants=write),
        "04_escrow_R1": Scenario("04_escrow_R1", horizon=5, max_instances=1, tool_grants=coord),
        "04_escrow_R2": Scenario("04_escrow_R2", horizon=5, max_instances=2, tool_grants=coord),
        "04_escrow_R2_lag": Scenario("04_escrow_R2_lag", horizon=5, max_instances=2,
                                    release_lag=4, tool_grants=coord),
        "05_frozen_only": Scenario("05_frozen_only", tool_grants=both, max_instances=2),
        "05_memory_capped": Scenario("05_memory_capped", memory_slots=1,
                                     tool_grants=both, max_instances=2),
        "05_full_gateway": Scenario("05_full_gateway", tool_grants=both, max_instances=2,
                                    enforce_gateway=True, approved_actions=all_safe),
    }


EXPECTED_UNSAFE = {
    "01_memory_off": False, "01_memory_on": True,
    "02_peer_sham": False, "02_peer_exchange": True,
    "03_read_only": False, "03_write_tool": True,
    "04_escrow_R1": False, "04_escrow_R2": True, "04_escrow_R2_lag": False,
    "05_frozen_only": True, "05_memory_capped": False, "05_full_gateway": False,
}


def run() -> dict:
    arms = {key: explore(c) for key, c in cases().items()}
    passed = all(arm["unsafe_reachable"] == EXPECTED_UNSAFE[key] for key, arm in arms.items())
    return {"schema": "plan_r_bounded_power_toy_v0", "scope": "exhaustive toy reference",
            "model_sha256": MODEL_SHA256, "passed_fixture_expectations": passed,
            "arms": arms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional local JSON report path")
    parser.add_argument("--check", action="store_true", help="Fail unless registered toy fixtures pass")
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["passed_fixture_expectations"] or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
