"""Exact finite Bayesian action/observation oracle for OTTD-CAP-01.

The oracle knows the *distribution* over quality masks, never the realized mask
before feedback. It chooses actions after each public observation, partitions
positive-probability posterior histories and solves the finite decision tree.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from fractions import Fraction

from model import Action, Scenario, World, advance, initial, legal_actions, net_cash, observe, run_episode


def _payoff(scenario: Scenario, world: World, objective: str) -> int:
    if objective == "net_cash":
        return net_cash(scenario, world)
    if objective == "operating_profit":
        return world.operating_profit
    raise ValueError("objective must be net_cash or operating_profit")


def exact_bayesian_value(scenario: Scenario, objective: str = "net_cash") -> Fraction:
    """Best expected result for history-aware policy with uniform independent bits.

    Exact rational arithmetic. Limit is deliberate: this is a finite fixture oracle,
    not a proof of optimality for a live OpenTTD session or arbitrary POMDP.
    """
    if len(scenario.routes) > 3 or scenario.horizon > 7:
        raise ValueError("exact oracle is limited to <=3 routes and <=7 months")
    _payoff(scenario, initial(scenario), objective)
    ensemble = tuple((m, initial(scenario)) for m in range(1 << len(scenario.routes)))

    @lru_cache(maxsize=None)
    def value(belief: tuple[tuple[int, World], ...]) -> Fraction:
        visible = observe(belief[0][1])
        if any(observe(world) != visible for _, world in belief):
            raise AssertionError("posterior history has inconsistent public observations")
        if visible.month == scenario.horizon:
            return Fraction(sum(_payoff(scenario, world, objective) for _, world in belief),
                            len(belief))
        best = None
        for action in legal_actions(scenario, belief[0][1]):
            groups = defaultdict(list)
            for mask, world in belief:
                successor = advance(scenario, world, mask, action)
                groups[observe(successor)].append((mask, successor))
            total = sum(
                (Fraction(len(group), len(belief)) * value(tuple(group))
                 for group in groups.values()),
                Fraction(0)
            )
            if best is None or total > best:
                best = total
        assert best is not None
        return best

    return value(ensemble)


def clairvoyant_value(scenario: Scenario, quality_mask: int,
                      objective: str = "net_cash") -> int:
    """Full-quality foresight, a valid upper bound on interface policies."""
    if len(scenario.routes) > 3 or scenario.horizon > 7:
        raise ValueError("clairvoyant oracle is limited to <=3 routes and <=7 months")
    @lru_cache(maxsize=None)
    def value(world: World) -> int:
        if world.month == scenario.horizon:
            return _payoff(scenario, world, objective)
        return max(value(advance(scenario, world, quality_mask, a))
                   for a in legal_actions(scenario, world))
    return value(initial(scenario))


def policy_mean(scenario: Scenario, factory, objective: str = "net_cash") -> Fraction:
    """Exact expectation of a deterministic policy over every equally likely world."""
    if len(scenario.routes) > 3:
        raise ValueError("policy_mean enumerates at most three hidden bits")
    outcomes = []
    for mask in range(1 << len(scenario.routes)):
        episode = run_episode(scenario, mask, factory())
        outcomes.append(episode["net_cash"] if objective == "net_cash"
                        else episode["operating_profit"])
    return Fraction(sum(outcomes), len(outcomes))


def clairvoyant_mean(scenario: Scenario, objective: str = "net_cash") -> Fraction:
    return Fraction(sum(clairvoyant_value(scenario, m, objective)
                        for m in range(1 << len(scenario.routes))),
                    1 << len(scenario.routes))
