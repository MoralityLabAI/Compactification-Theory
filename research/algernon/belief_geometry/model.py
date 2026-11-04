"""Exact evidence algebra and finite-horizon decision geometry for Algernon.

CPU-only reference, no model inference. All posterior, curvature and alpha-vector
calculations use Fraction; only entropy uses floating point.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, permutations, product
from math import log2


def frac(x):
    return x if isinstance(x, Q) else Q(x)


def normalize(weights):
    out = tuple(map(frac, weights))
    if not out or min(out) < 0 or sum(out) <= 0:
        raise ValueError("expected nonnegative weights with positive sum")
    return tuple(x / sum(out) for x in out)


@dataclass(frozen=True)
class Evidence:
    """Observed likelihood factors, conditionally independent given hypothesis.

    Each factor represents the probability of the *realized* evidence event.
    Its complementary outcome need not be present in this container.
    """
    prior: tuple[Q, ...]
    factors: tuple[tuple[Q, ...], ...]

    def __post_init__(self):
        n = len(self.prior)
        if n < 2 or any(frac(p) <= 0 for p in self.prior) or sum(self.prior) != 1:
            raise ValueError("strictly positive normalized prior required")
        for row in self.factors:
            if len(row) != n or any(not 0 < frac(x) <= 1 for x in row):
                raise ValueError("strictly positive likelihoods in (0,1] required")

    def posterior(self, order=()):
        weights = list(self.prior)
        for i in order:
            if not isinstance(i, int) or not 0 <= i < len(self.factors):
                raise ValueError("invalid evidence index")
            weights = [weights[h] * self.factors[i][h] for h in range(len(weights))]
        return normalize(weights)

    def odds(self, order=()):
        p = self.posterior(order)
        return tuple(x / p[-1] for x in p[:-1])

    def odds_multiplier(self, i):
        row = self.factors[i]
        return tuple(x / row[-1] for x in row[:-1])

    def all_orders_equal(self):
        for size in range(len(self.factors) + 1):
            for subset in combinations(range(len(self.factors)), size):
                target = self.posterior(subset)
                for order in permutations(subset):
                    if self.posterior(order) != target:
                        return False
        return True


@dataclass(frozen=True)
class DiscreteOneForm:
    """An edge field on the monotone directed n-cube with Abelian vector values.

    For edge (mask,i), i must be absent from mask. A square's curvature is
    g(S,i)+g(S+i,j)-g(S,j)-g(S+j,i).
    """
    n: int
    dimension: int
    edges: dict[tuple[int, int], tuple[Q, ...]]

    def __post_init__(self):
        if not 1 <= self.n <= 8 or self.dimension < 1:
            raise ValueError("invalid hypercube dimensions")
        needed = {(m, i) for m in range(1 << self.n)
                  for i in range(self.n) if not m & (1 << i)}
        if set(self.edges) != needed:
            raise ValueError("every monotone edge must be assigned exactly once")
        if any(len(v) != self.dimension for v in self.edges.values()):
            raise ValueError("edge coordinate mismatch")

    def curvature(self, mask, i, j):
        if i == j or mask & (1 << i) or mask & (1 << j):
            raise ValueError("invalid square face")
        a, b = self.edges[(mask,i)], self.edges[(mask | (1 << i),j)]
        c, d = self.edges[(mask,j)], self.edges[(mask | (1 << j),i)]
        return tuple(a[k] + b[k] - c[k] - d[k] for k in range(self.dimension))

    def max_curvature(self):
        return max((max(abs(x) for x in self.curvature(mask,i,j))
                    for mask in range(1 << self.n)
                    for i in range(self.n) for j in range(i+1,self.n)
                    if not (mask & (1 << i) or mask & (1 << j))),default=Q(0))

    def integrate(self, order):
        if len(set(order)) != len(order) or any(not isinstance(i,int) or not 0 <= i < self.n for i in order):
            raise ValueError("a path visits each event at most once")
        m=0
        v=[Q(0)]*self.dimension
        for i in order:
            delta=self.edges[(m,i)]
            v=[v[k]+delta[k] for k in range(self.dimension)]
            m|=1<<i
        return tuple(v)

    @staticmethod
    def swaps(order,other):
        if set(order) != set(other) or len(order)!=len(other):
            raise ValueError("permutations must contain same events")
        pos={v:i for i,v in enumerate(other)}
        return sum(pos[order[i]]>pos[order[j]] for i in range(len(order)) for j in range(i+1,len(order)))

    def path_bound(self,order,other):
        delta=self.integrate(order)
        other_delta=self.integrate(other)
        error=max(abs(x-y) for x,y in zip(delta,other_delta))
        bound=self.swaps(order,other)*self.max_curvature()
        return {"observed":error,"certified_upper_bound":bound,"valid":error<=bound}


def flat_one_form(vectors):
    """Create a state-independent edge increment for each event."""
    if not vectors:
        raise ValueError("needs at least one increment")
    n=len(vectors); d=len(vectors[0])
    return DiscreteOneForm(n,d,{(mask,i):tuple(map(frac,vectors[i]))
                                for mask in range(1<<n) for i in range(n)
                                if not (mask & (1<<i))})


@dataclass(frozen=True)
class Probe:
    name: str
    outcomes: tuple[tuple[Q,...],...]
    cost: Q

    def __post_init__(self):
        if len(self.outcomes)<2 or not self.outcomes[0] or frac(self.cost)<0:
            raise ValueError("invalid probe")
        n=len(self.outcomes[0])
        if any(len(row)!=n or any(not 0<=frac(x)<=1 for x in row) for row in self.outcomes):
            raise ValueError("invalid outcome likelihoods")
        if any(sum(row[h] for row in self.outcomes)!=1 for h in range(n)):
            raise ValueError("outcomes must sum to one for each hypothesis")


@dataclass(frozen=True)
class Alpha:
    payoff: tuple[Q,...]
    first_action: str


class FiniteHorizon:
    """Exact POMDP alpha-vector enumeration with each probe used at most once.

    STOP selects one of the supplied terminal decision payoff vectors.
    Future probe observations can select different contingent plans.
    No generator-distribution information is inferred or hidden in these objects.
    """
    def __init__(self, terminal_decisions, probes):
        self.terminal=tuple(tuple(map(frac,v)) for v in terminal_decisions)
        self.probes=tuple(probes)
        self.n=len(self.terminal[0])
        if self.n<2 or not self.terminal or any(len(v)!=self.n for v in self.terminal):
            raise ValueError("terminal payoff vectors must share a size")
        if any(len(p.outcomes[0])!=self.n for p in self.probes):
            raise ValueError("likelihood state dimensions must match")
        if len({p.name for p in self.probes})!=len(self.probes):
            raise ValueError("probe names must be distinct")

    @lru_cache(None)
    def alpha(self,remaining_mask,horizon):
        out=[Alpha(d,"STOP") for d in self.terminal]
        if horizon <= 0:
            return tuple(out)
        for i,probe in enumerate(self.probes):
            if not (remaining_mask & (1<<i)):
                continue
            child=self.alpha(remaining_mask ^ (1<<i), horizon-1)
            for policies in product(child,repeat=len(probe.outcomes)):
                values=tuple(-probe.cost+sum(probe.outcomes[o][h]*policies[o].payoff[h]
                                             for o in range(len(probe.outcomes)))
                             for h in range(self.n))
                out.append(Alpha(values,probe.name))
        # Drop vector duplicates and componentwise dominated plans.
        distinct={(a.payoff,a.first_action):a for a in out}
        all_plans=list(distinct.values())
        return tuple(a for a in all_plans if not any(
            a.first_action==b.first_action and a!=b
            and all(x<=y for x,y in zip(a.payoff,b.payoff))
            and any(x<y for x,y in zip(a.payoff,b.payoff)) for b in all_plans))

    def scores(self,belief,horizon):
        b=normalize(belief)
        per_action={}
        for a in self.alpha((1<<len(self.probes))-1,horizon):
            v=sum(x*y for x,y in zip(a.payoff,b))
            if a.first_action not in per_action or v>per_action[a.first_action]:
                per_action[a.first_action]=v
        return per_action

    def advantage(self,belief,horizon):
        scores=self.scores(belief,horizon)
        return max((v for k,v in scores.items() if k!="STOP"),default=Q(-10**10))-scores["STOP"]


def entropy_bits(probabilities):
    return -sum(float(p)*log2(float(p)) for p in probabilities if p>0)


def matrix_rank_rational(matrix):
    m=[list(map(frac,row)) for row in matrix]
    if not m:
        return 0
    if len({len(r) for r in m})!=1:
        raise ValueError("ragged matrix")
    rows=len(m); cols=len(m[0]); rank=0
    for j in range(cols):
        pivot=next((i for i in range(rank,rows) if m[i][j]),None)
        if pivot is None: continue
        m[rank],m[pivot]=m[pivot],m[rank]
        v=m[rank][j];m[rank]=[z/v for z in m[rank]]
        for i in range(rows):
            if i!=rank and m[i][j]:
                k=m[i][j]
                m[i]=[a-k*b for a,b in zip(m[i],m[rank])]
        rank+=1
        if rank==rows: break
    return rank


def fair_bits_example():
    """Two independent fair bits, goal predict X; probe A reveals Y,
    probe B observes X with 25% symmetric error. Both probes cost 1/20.
    """
    hypotheses=tuple(product((0,1),repeat=2))
    p=tuple([Q(1,4)]*4)
    stop=tuple(tuple(Q(x==guess) for x,y in hypotheses) for guess in (0,1))
    A=Probe("REVEAL_Y",tuple(tuple(Q(y==z) for x,y in hypotheses) for z in (0,1)),Q(1,20))
    B=Probe("NOISY_X",tuple(tuple(Q(3,4) if x==z else Q(1,4) for x,y in hypotheses) for z in (0,1)),Q(1,20))
    solver=FiniteHorizon(stop,[A,B])
    return p,solver
