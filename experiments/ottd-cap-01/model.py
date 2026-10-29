"""OTTD-CAP-01: deterministic, partially observed transport-economy micro-world.

This is an original synthetic environment, not the OpenTTD engine. All accounting
is integer and advances in discrete months. See PROTOCOL.md for the exact scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from typing import Protocol


@dataclass(frozen=True)
class Route:
    name: str
    build_cost: int
    good_monthly: int
    bad_monthly: int

    def __post_init__(self) -> None:
        if not self.name or self.build_cost <= 0 or self.good_monthly <= self.bad_monthly:
            raise ValueError("route requires a name, positive cost, good_monthly > bad_monthly")


@dataclass(frozen=True)
class Scenario:
    routes: tuple[Route, ...]
    horizon: int = 8
    starting_cash: int = 80

    def __post_init__(self) -> None:
        if not 1 <= len(self.routes) <= 8 or not 1 <= self.horizon <= 60 or self.starting_cash < 0:
            raise ValueError("scenario bounds: routes 1..8, horizon 1..60, cash >= 0")
        if len({r.name for r in self.routes}) != len(self.routes):
            raise ValueError("route names must be distinct")


@dataclass(frozen=True)
class World:
    month: int
    cash: int
    operating_profit: int = 0
    capital_spent: int = 0
    active_mask: int = 0
    latest_signal: tuple[int, int] | None = None


@dataclass(frozen=True)
class Observation:
    month: int
    cash: int
    operating_profit: int
    capital_spent: int
    active_mask: int
    latest_signal: tuple[int, int] | None


@dataclass(frozen=True)
class Action:
    kind: str
    route: int = -1


def initial(scenario: Scenario) -> World:
    return World(month=0, cash=scenario.starting_cash)


def observe(world: World) -> Observation:
    """Only public accounts, route ownership, clock and last survey result.

    Survey outcomes are transient. The environment never exposes quality_mask.
    Route prices and prior yields are public constants in Scenario.
    """
    return Observation(world.month, world.cash, world.operating_profit,
                       world.capital_spent, world.active_mask, world.latest_signal)


def legal_actions(scenario: Scenario, world: World) -> tuple[Action, ...]:
    if world.month >= scenario.horizon:
        return ()
    options = [Action("wait")]
    for i, route in enumerate(scenario.routes):
        if world.active_mask & (1 << i):
            continue
        options.append(Action("survey", i))
        if world.cash >= route.build_cost:
            options.append(Action("build", i))
    return tuple(options)


def advance(scenario: Scenario, world: World, quality_mask: int, action: Action) -> World:
    """Each action consumes one month. An investment begins earning next month.

    All already-active routes earn (or lose) monthly net *operating* income.
    Builds reduce cash and raise capital_spent but do not alter operating_profit.
    """
    if not 0 <= quality_mask < (1 << len(scenario.routes)):
        raise ValueError("quality mask out of bounds")
    if action not in legal_actions(scenario, world):
        raise ValueError(f"illegal action at month {world.month}: {action}")
    income = sum(
        (route.good_monthly if quality_mask & (1 << i) else route.bad_monthly)
        for i, route in enumerate(scenario.routes)
        if world.active_mask & (1 << i)
    )
    spend = scenario.routes[action.route].build_cost if action.kind == "build" else 0
    signal = (action.route, int(bool(quality_mask & (1 << action.route)))) if action.kind == "survey" else None
    active = world.active_mask | (1 << action.route) if action.kind == "build" else world.active_mask
    return World(
        month=world.month + 1,
        cash=world.cash + income - spend,
        operating_profit=world.operating_profit + income,
        capital_spent=world.capital_spent + spend,
        active_mask=active,
        latest_signal=signal,
    )


def net_cash(scenario: Scenario, world: World) -> int:
    return world.cash - scenario.starting_cash


def openttd_style_score(operating_profit: float, competitors: tuple[float, ...],
                        target: float = 3_000_000.0) -> float:
    """Faithful algebra of the upstream profit/rank objective, given valid ledger data.

    This never equates the synthetic operating-profit meter with actual OpenTTD
    measurements. Ties count as rivals not beaten.
    """
    if target <= 0 or not competitors:
        raise ValueError("positive target and at least one competitor required")
    profit_part = min(1.0, max(0.0, operating_profit / target))
    beaten_fraction = sum(operating_profit > x for x in competitors) / len(competitors)
    return 0.6 * profit_part + 0.4 * beaten_fraction


def seeded_world(seed: int, routes: int = 3, horizon: int = 10,
                 starting_cash: int = 80) -> tuple[Scenario, int]:
    if not 1 <= routes <= 8:
        raise ValueError("routes must be 1..8")
    rng = random.Random(seed)
    catalog = tuple(Route(f"R{i}", 12 + rng.randrange(22),
                          8 + rng.randrange(13), -5 + rng.randrange(9))
                    for i in range(routes))
    mask = rng.randrange(1 << routes)
    return Scenario(catalog, horizon, starting_cash), mask


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, scenario: Scenario, obs: Observation) -> Action: ...
    @property
    def peak_memory_bits(self) -> int: ...


class Greedy:
    """Public-prior greedy net-cash policy, no cross-turn private memory."""

    @property
    def peak_memory_bits(self) -> int:
        return 0

    def reset(self) -> None:
        return None

    def act(self, scenario: Scenario, obs: Observation) -> Action:
        remaining_income_months = scenario.horizon - obs.month - 1
        candidates = []
        for i, route in enumerate(scenario.routes):
            if obs.active_mask & (1 << i) or obs.cash < route.build_cost:
                continue
            prior = (route.good_monthly + route.bad_monthly) / 2
            if obs.latest_signal is not None and obs.latest_signal[0] == i:
                prior = route.good_monthly if obs.latest_signal[1] else route.bad_monthly
            surplus = remaining_income_months * prior - route.build_cost
            if surplus > 0:
                candidates.append((surplus, -i, i))
        return Action("build", max(candidates)[2]) if candidates else Action("wait")


class Scout:
    """Fixed clock-driven scout phase, then investment with a bit-budgeted cache.

    Route identity takes ceil(log2(number of routes)) bits and binary quality one
    more. The public clock determines the survey order, so no private phase
    counter is hidden outside this ledger. Policy code/parameters are separate.
    """

    def __init__(self, memory_bits: int, survey_count: int | None = None):
        if memory_bits < 0:
            raise ValueError("memory_bits must be nonnegative")
        self.memory_bits = memory_bits
        self.survey_count = survey_count
        self._known: dict[int, int] = {}
        self._peak = 0

    @property
    def peak_memory_bits(self) -> int:
        return self._peak

    def reset(self) -> None:
        self._known.clear()
        self._peak = 0

    def act(self, scenario: Scenario, obs: Observation) -> Action:
        width = math.ceil(math.log2(len(scenario.routes))) + 1
        slots = self.memory_bits // width
        if obs.latest_signal is not None and slots:
            i, bit = obs.latest_signal
            if i not in self._known and len(self._known) >= slots:
                self._known.pop(next(iter(self._known)))
            self._known[i] = bit
            self._peak = max(self._peak, len(self._known) * width)

        scouts = min(len(scenario.routes),
                     scenario.horizon,
                     self.survey_count if self.survey_count is not None else len(scenario.routes))
        if obs.month < scouts:
            return Action("survey", obs.month)

        candidates = dict(self._known)
        if obs.latest_signal is not None:
            candidates[obs.latest_signal[0]] = obs.latest_signal[1]
        remaining_income_months = scenario.horizon - obs.month - 1
        ranked = []
        for i, good in candidates.items():
            r = scenario.routes[i]
            if not good or obs.active_mask & (1 << i) or obs.cash < r.build_cost:
                continue
            surplus = remaining_income_months * r.good_monthly - r.build_cost
            if surplus > 0:
                ranked.append((surplus, -i, i))
        return Action("build", max(ranked)[2]) if ranked else Action("wait")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def run_episode(scenario: Scenario, quality_mask: int, policy: Policy) -> dict:
    """A hash-chained, deterministic rollout. Truth is in the receipt, never the view."""
    policy.reset()
    world = initial(scenario)
    prior = "0" * 64
    trace = []
    while world.month < scenario.horizon:
        before = observe(world)
        action = policy.act(scenario, before)
        next_world = advance(scenario, world, quality_mask, action)
        row = {"before": asdict(before), "action": asdict(action),
               "after": asdict(observe(next_world))}
        digest = sha256((prior + canonical(row)).encode("utf-8")).hexdigest()
        trace.append({**row, "sha256": digest})
        prior = digest
        world = next_world
    return {
        "kind": "synthetic_ottd_cap_01",
        "scenario": asdict(scenario),
        "quality_mask": quality_mask,
        "terminal": asdict(world),
        "net_cash": net_cash(scenario, world),
        "operating_profit": world.operating_profit,
        "peak_memory_bits": policy.peak_memory_bits,
        "steps": len(trace),
        "trace_sha256": prior,
        "trace": trace,
    }


def verify_receipt(receipt: dict) -> bool:
    """Recalculate every transition and digest, including final public accounts."""
    try:
        s = receipt["scenario"]
        scenario = Scenario(tuple(Route(**r) for r in s["routes"]), s["horizon"], s["starting_cash"])
        mask = receipt["quality_mask"]
        world = initial(scenario)
        prior = "0" * 64
        for row in receipt["trace"]:
            action = Action(**row["action"])
            if row["before"] != asdict(observe(world)):
                return False
            world = advance(scenario, world, mask, action)
            payload = {"before": row["before"], "action": row["action"], "after": asdict(observe(world))}
            if row["after"] != payload["after"]:
                return False
            prior = sha256((prior + canonical(payload)).encode("utf-8")).hexdigest()
            if row["sha256"] != prior:
                return False
        return (world.month == scenario.horizon and receipt["terminal"] == asdict(world)
                and receipt["net_cash"] == net_cash(scenario, world)
                and receipt["operating_profit"] == world.operating_profit
                and receipt["trace_sha256"] == prior)
    except (KeyError, TypeError, ValueError, IndexError):
        return False
