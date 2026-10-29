"""Exact synthetic multi-front control fixture for interface-relative experiments.

Each front has one static binary cue and admits one attack. Public rules map
that cue to a target. This fixture models access and command timing, with no
claim to reproduce StarCraft unit mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import operator
import random
from typing import Any


def _int(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return result


def _fields(data: dict, cls: type) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)!r}")
    return dict(data)


@dataclass(frozen=True)
class RuleSpec:
    fronts: int = 4
    deadline_ticks: int = 12
    target_xor: int = 0
    reward_weights: tuple[int, ...] = (1, 1, 1, 1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fronts", _int(self.fronts, "fronts", 1, 8))
        object.__setattr__(self, "deadline_ticks", _int(self.deadline_ticks, "deadline_ticks", 1, 1_000_000))
        object.__setattr__(self, "target_xor", _int(self.target_xor, "target_xor", 0, 1))
        if not isinstance(self.reward_weights, (tuple, list)):
            raise ValueError("reward_weights must be a sequence of integers")
        weights = tuple(_int(x, "reward weight", 1, 1_000_000) for x in self.reward_weights)
        if len(weights) != self.fronts:
            raise ValueError("reward_weights must have one entry for each front")
        object.__setattr__(self, "reward_weights", weights)

    @classmethod
    def from_dict(cls, data: dict) -> "RuleSpec":
        return cls(**_fields(data, cls))

    def to_dict(self) -> dict:
        return {"fronts": self.fronts, "deadline_ticks": self.deadline_ticks,
                "target_xor": self.target_xor, "reward_weights": list(self.reward_weights)}


@dataclass(frozen=True)
class InterfaceSpec:
    name: str = "camera"
    observation: str = "camera"
    command_ticks: int = 3
    switch_ticks: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ValueError("interface name must be a nonempty string of at most 128 characters")
        if self.observation not in ("camera", "global"):
            raise ValueError("observation must be camera or global")
        object.__setattr__(self, "command_ticks", _int(self.command_ticks, "command_ticks", 1, 1_000_000))
        object.__setattr__(self, "switch_ticks", _int(self.switch_ticks, "switch_ticks", 0, 1_000_000))

    @classmethod
    def from_dict(cls, data: dict) -> "InterfaceSpec":
        return cls(**_fields(data, cls))

    def to_dict(self) -> dict:
        return {"name": self.name, "observation": self.observation,
                "command_ticks": self.command_ticks, "switch_ticks": self.switch_ticks}


def freeze_observation(value: Any) -> Any:
    """Return an immutable, hashable copy of a JSON-shaped observation."""
    if isinstance(value, dict):
        return tuple((key, freeze_observation(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_observation(x) for x in value)
    return value


def optimal_schedule(rules: RuleSpec, interface: InterfaceSpec, *,
                     remaining_ticks: int | None = None, current_front: int = 0,
                     resolved: tuple[bool, ...] | list[bool] | None = None,
                     remaining_decisions: int | None = None) -> tuple[int, tuple[int, ...]]:
    """Return maximum remaining unnormalized weight and a front visitation order.

    Cues are static; a visited front reveals its cue before an attack. Public
    rules make each scheduled attack achievable by an unrestricted controller.
    Each front pays one command cost; changing fronts pays one switch cost.
    The solver considers every ordered subset with integer arithmetic.
    """
    if not isinstance(rules, RuleSpec) or not isinstance(interface, InterfaceSpec):
        raise ValueError("optimal_schedule requires RuleSpec and InterfaceSpec")
    current_front = _int(current_front, "current_front", 0, rules.fronts - 1)
    budget = rules.deadline_ticks if remaining_ticks is None else _int(remaining_ticks, "remaining_ticks", 0)
    decisions = 2 * rules.fronts if remaining_decisions is None else _int(remaining_decisions, "remaining_decisions", 0)
    done = (False,) * rules.fronts if resolved is None else tuple(resolved)
    if len(done) != rules.fronts or any(type(x) is not bool for x in done):
        raise ValueError("resolved must contain one bool for each front")
    mask = sum(1 << i for i in range(rules.fronts) if not done[i])

    @lru_cache(maxsize=None)
    def solve(pending: int, selected: int, ticks: int, actions_left: int) -> tuple[int, tuple[int, ...], int]:
        best_weight, best_order, best_time = 0, (), 0
        for front in range(rules.fronts):
            bit = 1 << front
            cost = interface.command_ticks + (interface.switch_ticks if front != selected else 0)
            actions = 2 if interface.observation == "camera" and front != selected else 1
            if not pending & bit or cost > ticks or actions > actions_left:
                continue
            tail_weight, tail_order, tail_time = solve(pending ^ bit, front, ticks - cost, actions_left - actions)
            weight = rules.reward_weights[front] + tail_weight
            order, elapsed = (front,) + tail_order, cost + tail_time
            if (weight > best_weight or
                    (weight == best_weight and (elapsed, order) < (best_time, best_order))):
                best_weight, best_order, best_time = weight, order, elapsed
        return best_weight, best_order, best_time

    weight, order, _ = solve(mask, current_front, budget, decisions)
    return weight, order


def optimal_front_from_observation(observation: dict) -> int | None:
    """Compute the next scheduled front using only public fields; cues are ignored."""
    rules = RuleSpec.from_dict(observation["rules"])
    interface = InterfaceSpec.from_dict(observation["interface"])
    tick = _int(observation["tick"], "tick", 0, rules.deadline_ticks)
    _, order = optimal_schedule(rules, interface,
                                remaining_ticks=rules.deadline_ticks - tick,
                                current_front=observation["current_front"],
                                resolved=observation["resolved"],
                                remaining_decisions=observation.get("remaining_decisions"))
    return order[0] if order else None


class MultiFrontEnv:
    """Deterministic-seed fixture with Gymnasium-shaped reset/step results.

    Each reset samples independent fair cues with a local PRNG. Policy-visible
    data omit the seed and hidden cues. An over-deadline operation stops at the
    deadline without a completed effect; completion exactly at the deadline
    counts. A decision cap bounds zero-cost switch loops. Any optimal schedule
    takes at most twice the number of fronts, within that cap from reset.
    """

    metadata = {"render_modes": [], "name": "PlutoMultiFront-v0"}

    def __init__(self, rules: RuleSpec | None = None, interface: InterfaceSpec | None = None):
        self.rules = RuleSpec() if rules is None else rules
        self.interface = InterfaceSpec() if interface is None else interface
        if not isinstance(self.rules, RuleSpec) or not isinstance(self.interface, InterfaceSpec):
            raise ValueError("MultiFrontEnv requires RuleSpec and InterfaceSpec")
        self.decision_limit = max(32, 4 * self.rules.fronts + 2 * self.rules.deadline_ticks)
        self._started = False
        self._terminated = False
        self._truncated = False

    def reset(self, *, seed: int = 0, options: dict | None = None) -> tuple[dict, dict]:
        seed = _int(seed, "seed", 0, 2**64 - 1)
        if options is not None and options != {}:
            raise ValueError("reset options must be empty")
        rng = random.Random(seed)
        self._cues = tuple(rng.randrange(2) for _ in range(self.rules.fronts))
        self.tick = 0
        self.current_front = 0
        self._status = ["pending"] * self.rules.fronts
        self._steps = 0
        self._return_weight = 0
        self._started = True
        self._terminated = False
        self._truncated = False
        return self.observation(), self._info("reset", 0)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("reset must be called first")

    def _require_active(self) -> None:
        self._require_started()
        if self._terminated or self._truncated:
            raise RuntimeError("episode is over; call reset before step")

    def observation(self) -> dict:
        self._require_started()
        visible = [cue if self.interface.observation == "global" or i == self.current_front else None
                   for i, cue in enumerate(self._cues)]
        return {"tick": self.tick, "remaining_ticks": self.rules.deadline_ticks - self.tick,
                "remaining_decisions": self.decision_limit - self._steps,
                "current_front": self.current_front, "visible_cues": visible,
                "resolved": [x != "pending" for x in self._status], "status": list(self._status),
                "rules": self.rules.to_dict(), "interface": self.interface.to_dict()}

    def _info(self, outcome: str, elapsed: int) -> dict:
        return {"evidence": "synthetic_exact_fixture", "outcome": outcome,
                "elapsed_ticks": elapsed, "episode_return": self._return_weight / sum(self.rules.reward_weights),
                "decisions": self._steps, "decision_limit": self.decision_limit}

    def legal_actions(self) -> list[dict]:
        self._require_started()
        if self._terminated or self._truncated:
            return []
        actions = [{"kind": "switch", "front": i} for i in range(self.rules.fronts)
                   if i != self.current_front]
        for i in range(self.rules.fronts):
            if self._status[i] == "pending" and (self.interface.observation == "global" or i == self.current_front):
                actions.extend({"kind": "attack", "front": i, "target": target} for target in (0, 1))
        actions.append({"kind": "wait", "ticks": 1})
        return actions

    def _validate_action(self, action: dict) -> tuple[str, int | None, int | None, int]:
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        kind = action.get("kind")
        if kind == "wait":
            if set(action) != {"kind", "ticks"}:
                raise ValueError("wait requires exactly kind and ticks")
            return kind, None, None, _int(action["ticks"], "wait ticks", 1, 1_000_000)
        if kind not in ("attack", "switch"):
            raise ValueError("unknown action kind")
        expected = {"kind", "front", "target"} if kind == "attack" else {"kind", "front"}
        if set(action) != expected:
            raise ValueError(f"{kind} has missing or unknown fields")
        front = _int(action["front"], "front", 0, self.rules.fronts - 1)
        if kind == "switch":
            if front == self.current_front:
                raise ValueError("switch requires a different front")
            return kind, front, None, self.interface.switch_ticks
        target = _int(action["target"], "target", 0, 1)
        if self._status[front] != "pending":
            raise ValueError("front has already been resolved")
        if self.interface.observation == "camera" and front != self.current_front:
            raise ValueError("camera interface must switch before attacking another front")
        cost = self.interface.command_ticks + (self.interface.switch_ticks if front != self.current_front else 0)
        return kind, front, target, cost

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        self._require_active()
        kind, front, target, cost = self._validate_action(action)
        previous_tick = self.tick
        self._steps += 1
        reward_weight = 0
        if cost > self.rules.deadline_ticks - self.tick:
            self.tick = self.rules.deadline_ticks
            self._terminated = True
            outcome = "deadline_before_completion"
        else:
            self.tick += cost
            outcome = kind
            if kind == "switch":
                self.current_front = front
            elif kind == "attack":
                self.current_front = front
                hit = target == (self._cues[front] ^ self.rules.target_xor)
                self._status[front] = "hit" if hit else "miss"
                reward_weight = self.rules.reward_weights[front] if hit else 0
                self._return_weight += reward_weight
                outcome = self._status[front]
            self._terminated = (self.tick == self.rules.deadline_ticks or
                                all(x != "pending" for x in self._status))
        self._truncated = not self._terminated and self._steps >= self.decision_limit
        if self._truncated:
            outcome = "decision_limit"
        return (self.observation(), reward_weight / sum(self.rules.reward_weights), self._terminated,
                self._truncated, self._info(outcome, self.tick - previous_tick))

    def oracle_value(self) -> float:
        """Exact attainable future return from the current public state."""
        self._require_started()
        if self._terminated or self._truncated:
            return 0.0
        weight, _ = optimal_schedule(self.rules, self.interface,
                                     remaining_ticks=self.rules.deadline_ticks - self.tick,
                                     current_front=self.current_front,
                                     resolved=[x != "pending" for x in self._status],
                                     remaining_decisions=self.decision_limit - self._steps)
        return weight / sum(self.rules.reward_weights)

    def optimal_front(self) -> int | None:
        """Next front of an optimal schedule, computed using public data only."""
        self._require_active()
        return optimal_front_from_observation(self.observation())

    def optimal_action(self) -> dict:
        """Explicit demonstration teacher: schedule optimally, then decode a visible cue."""
        self._require_active()
        front = self.optimal_front()
        if front is None:
            return {"kind": "wait", "ticks": self.rules.deadline_ticks - self.tick}
        if self.interface.observation == "camera" and front != self.current_front:
            return {"kind": "switch", "front": front}
        cue = self.observation()["visible_cues"][front]
        if cue is None:
            raise RuntimeError("teacher attempted to read an unavailable cue")
        return {"kind": "attack", "front": front, "target": cue ^ self.rules.target_xor}

    def close(self) -> None:
        """No external resources are held by the synthetic fixture."""
