"""Algernon M-VOI-01: finite-horizon action-value approximation certificate.

This verifies mathematical properties of given Q tables, not neural performance.
Use python -m unittest discover -s research/algernon/metacognition/tests_mvoi -v
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    names: tuple[str, ...]
    oracle_q: tuple[float, ...]
    estimated_q: tuple[float, ...]

    def __post_init__(self):
        if len(self.names) == 0 or len(self.names) != len(self.oracle_q) or len(self.oracle_q) != len(self.estimated_q):
            raise ValueError("Each action requires a label and two values")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Action names must be unique")

    def audit(self):
        o = max(range(len(self.names)), key=lambda i: self.oracle_q[i])
        a = max(range(len(self.names)), key=lambda i: self.estimated_q[i])
        error = max(abs(x-y) for x,y in zip(self.oracle_q,self.estimated_q))
        ordered = sorted(self.oracle_q,reverse=True)
        margin = ordered[0]-ordered[1] if len(ordered)>1 else float("inf")
        regret = self.oracle_q[o]-self.oracle_q[a]
        assert regret <= 2*error + 1e-12
        if margin > 2*error:
            assert o == a
        return {"oracle_action":self.names[o],"estimated_action":self.names[a],
                "q_sup_error":error,"oracle_margin":margin,
                "single_decision_regret":regret,"margin_certified":margin>2*error}


def bound_from_steps(epsilons):
    """Expected horizon regret <= 2 sum(eps_k), given a uniform sup error
    bound on optimal Q at each reached state and zero terminal residual.
    """
    if any(e < 0 for e in epsilons):
        raise ValueError("Error budgets must be nonnegative")
    return 2*sum(epsilons)
