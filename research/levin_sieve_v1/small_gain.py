"""Exact-rational certificates for fair asynchronous goal repair.

A certificate proves an error-envelope statement conditional on the specified
componentwise bounds. Physical or neural applications require an independent
Lipschitz and intervention audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Iterable, Sequence


def _q(x: int | str | Q) -> Q:
    if isinstance(x, (int, str, Q)):
        return Q(x)
    raise TypeError("use int, string, or Fraction (floats are ambiguous)")


def _matrix(rows: Iterable[Iterable[int | str | Q]]) -> tuple[tuple[Q, ...], ...]:
    a = tuple(tuple(_q(x) for x in row) for row in rows)
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("matrix must be nonempty and square")
    if any(v < 0 for row in a for v in row):
        raise ValueError("componentwise gain matrix must be nonnegative")
    return a


def _solve(a: tuple[tuple[Q, ...], ...]) -> tuple[Q, ...]:
    """Exactly solve (I-A)w=1; reject singular systems."""
    n = len(a)
    aug = [[Q(int(i == j)) - a[i][j] for j in range(n)] + [Q(1)]
           for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col]), None)
        if pivot is None:
            raise ValueError("I-A is singular; no resolvent witness")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [x - factor*y for x, y in zip(aug[r], aug[col])]
    return tuple(row[-1] for row in aug)


def mode_envelope(modes: Iterable[Iterable[Iterable[int | str | Q]]]) -> tuple[tuple[Q, ...], ...]:
    """Entrywise upper bound on allowed gain matrices for arbitrary switching."""
    ms = tuple(_matrix(m) for m in modes)
    if not ms or any(len(m) != len(ms[0]) for m in ms):
        raise ValueError("require nonempty set of equally sized mode matrices")
    n = len(ms[0])
    return tuple(tuple(max(m[i][j] for m in ms) for j in range(n))
                 for i in range(n))


@dataclass(frozen=True)
class Certificate:
    gains: tuple[tuple[Q, ...], ...]
    weights: tuple[Q, ...]
    rho: Q

    @classmethod
    def from_weights(cls, gains: Iterable[Iterable[int | str | Q]],
                     weights: Sequence[int | str | Q]) -> "Certificate":
        a = _matrix(gains)
        w = tuple(_q(v) for v in weights)
        if len(w) != len(a) or any(v <= 0 for v in w):
            raise ValueError("require one strictly positive weight per component")
        rho = max(sum((v*wj for v, wj in zip(row, w)), Q(0))/wi
                  for row, wi in zip(a, w))
        if rho >= 1:
            raise ValueError("weighted small-gain ratio is >= 1")
        cert = cls(a, w, rho)
        cert.validate()
        return cert

    @classmethod
    def synthesize(cls, gains: Iterable[Iterable[int | str | Q]]) -> "Certificate":
        """Exact resolvent witness w=(I-A)^-1 1 when spectral radius(A)<1."""
        a = _matrix(gains)
        return cls.from_weights(a, _solve(a))

    @classmethod
    def synthesize_for_modes(cls, modes: Iterable[Iterable[Iterable[int | str | Q]]]) -> "Certificate":
        """Find a conservative common certificate for arbitrary mode switching."""
        return cls.synthesize(mode_envelope(modes))

    def validate(self) -> None:
        a = _matrix(self.gains)
        w = tuple(_q(v) for v in self.weights)
        rho = _q(self.rho)
        if a != self.gains or w != self.weights or rho != self.rho:
            raise ValueError("canonical rational representation required")
        if len(w) != len(a) or any(v <= 0 for v in w) or not (Q(0) <= rho < 1):
            raise ValueError("invalid dimensions, weights, or rho")
        for row, wi in zip(a, w):
            if sum((v*wj for v, wj in zip(row, w)), Q(0)) > rho*wi:
                raise ValueError("Aw <= rho*w fails")

    def error_norm(self, error: Sequence[int | str | Q]) -> Q:
        if len(error) != len(self.weights):
            raise ValueError("state dimension mismatch")
        return max(abs(_q(x))/w for x, w in zip(error, self.weights))

    def scaled_noise(self, limits: Sequence[int | str | Q]) -> Q:
        if len(limits) != len(self.weights):
            raise ValueError("disturbance dimension mismatch")
        d = tuple(_q(v) for v in limits)
        if any(v < 0 for v in d):
            raise ValueError("disturbance limits must be nonnegative")
        return max(v/w for v, w in zip(d, self.weights))

    def radius(self, limits: Sequence[int | str | Q]) -> Q:
        return self.scaled_noise(limits)/(1-self.rho)

    def bound(self, initial_error_norm: Q, limits: Sequence[int | str | Q],
              completed_blocks: int) -> Q:
        if completed_blocks < 0 or initial_error_norm < 0:
            raise ValueError("invalid horizon or initial error")
        r = self.radius(limits)
        return r + self.rho**completed_blocks*max(initial_error_norm-r, Q(0))

    def epochs_to_goal(self, initial_error: Sequence[int | str | Q],
                       limits: Sequence[int | str | Q], tolerance: int | str | Q,
                       max_epochs: int = 10000) -> int | None:
        """Sufficient complete fair blocks; None if this bound cannot reach."""
        target = _q(tolerance)
        if target < 0 or max_epochs < 0:
            raise ValueError("invalid target or cap")
        start = self.error_norm(initial_error)
        if start <= target:
            return 0
        if target <= self.radius(limits):
            return None
        for k in range(max_epochs+1):
            if self.bound(start, limits, k) <= target:
                return k
        return None


def audit_trace(cert: Certificate,
                states: Sequence[Sequence[int | str | Q]],
                schedule: Sequence[int], limits: Sequence[int | str | Q],
                block_size: int) -> dict[str, str | int]:
    """Audit exact-rational local gains, asynchronous fairness and recovery bound."""
    cert.validate()
    n = len(cert.weights)
    if block_size < n or not schedule or len(schedule) % block_size:
        raise ValueError("require nonempty, complete, aligned fair blocks")
    if len(states) != len(schedule)+1:
        raise ValueError("need one more state than updates")
    if any(not isinstance(i, int) or not 0 <= i < n for i in schedule):
        raise ValueError("invalid coordinate index")
    if any(set(schedule[k:k+block_size]) != set(range(n))
           for k in range(0, len(schedule), block_size)):
        raise ValueError("each aligned block must update every coordinate")
    d = tuple(_q(x) for x in limits)
    cert.scaled_noise(d)
    trace = [tuple(_q(x) for x in row) for row in states]
    if any(len(x) != n for x in trace):
        raise ValueError("state dimension mismatch")
    e0 = cert.error_norm(trace[0])
    for t, i in enumerate(schedule, 1):
        before, after = trace[t-1], trace[t]
        if any(after[j] != before[j] for j in range(n) if j != i):
            raise ValueError(f"nonlocal update at step {t}")
        limit = sum((a*abs(x) for a,x in zip(cert.gains[i],before)),Q(0)) + d[i]
        if abs(after[i]) > limit:
            raise ValueError(f"local gain contract violated at step {t}")
        if cert.error_norm(after) > cert.bound(e0, d, t//block_size):
            raise AssertionError(f"certified bound violated at step {t}")
    blocks = len(schedule)//block_size
    return {"steps": len(schedule), "fair_blocks": blocks,
            "final_error_norm": str(cert.error_norm(trace[-1])),
            "certified_upper_bound": str(cert.bound(e0, d, blocks)),
            "disturbance_radius": str(cert.radius(d)),
            "rho": str(cert.rho)}


def demo() -> dict:
    """Example where ordinary sup-norm fails but a weighted certificate passes."""
    a = ((Q(0),Q(3)),(Q(1,10),Q(0)))
    cert = Certificate.from_weights(a, (Q(11,2),1))
    disturbance = (Q(1,100),Q(1,100))
    schedule = [0,1]*6
    state = (Q(10), Q(-3))
    states = [state]
    for t,i in enumerate(schedule):
        noise = disturbance[i] * (1 if t % 3 == 0 else -1)
        next_state = list(state)
        next_state[i] = sum((v*x for v,x in zip(a[i],state)),Q(0))+noise
        state = tuple(next_state)
        states.append(state)
    receipt = audit_trace(cert, states, schedule, disturbance, 2)
    return {"example":"2-node cross-coupled repair", "ordinary_row_sum": "3",
            "weights": [str(w) for w in cert.weights], **receipt,
            "target_tolerance":"1/10",
            "sufficient_blocks_for_target":cert.epochs_to_goal(states[0],disturbance,Q(1,10))}


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2, sort_keys=True))
