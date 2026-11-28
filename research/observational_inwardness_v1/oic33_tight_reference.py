"""OIC-33: independently implemented tight-pair mandatory-address bound.

Only a *necessary* bound for a successful continuation. The two-layer
candidate encoding uses 0 or exactly 1 already-consumed bit substitution.
"""
from __future__ import annotations

from oic23_exact import ExactFaultGame


def forced_addresses(n: int, horizon: int, t: int, layers: tuple[int,int]) -> tuple[int, str]:
    if not (2 <= n and 0 <= t <= horizon and len(layers) == 2):
        raise ValueError("invalid game parameters")
    zero, one = layers
    if zero & one:
        raise ValueError("a deterministic world must have a unique fault count")
    if (zero | one) >> (n*n):
        raise ValueError("candidate outside declared domain")
    choices = (1 << n) - 1
    survivors = [idx for idx in range(n*n) if (zero|one)&(1 << idx)]
    for ix, a in enumerate(survivors):
        da, za = divmod(a,n)
        for b in survivors[ix+1:]:
            db, zb = divmod(b,n)
            if da == db:
                continue
            available = ((zero >> a)&1) + ((zero >> b)&1)
            differences = sum((za + s*da) % n != (zb + s*db) % n
                              for s in range(t,horizon))
            if differences <= available:
                return 0, 'pairwise'
            if differences == available + 1 and t < horizon:
                pa, pb = (za + t*da)%n, (zb + t*db)%n
                if pa != pb:
                    choices &= (1<<pa) | (1<<pb)
                    if not choices:
                        return 0, 'intersection'
    return choices, 'allowed'


def has_pairwise_obstruction(n: int, horizon: int, t: int, layers: tuple[int,int]) -> bool:
    """Independent witness for the OIC-T44 pairwise bound."""
    if t >= horizon:
        mask = layers[0] | layers[1]
        return len({a//n for a in range(n*n) if (mask>>a)&1}) > 1
    a, reason = forced_addresses(n, horizon, t, layers)
    return reason == 'pairwise'


class ForcedGame(ExactFaultGame):
    """Exact finite Bellman reference with sound mandatory-address filtering."""
    def __init__(self,n,Q,node_cap=2000000):
        super().__init__(n,Q,1,0,node_cap)
        self.pair_cuts=0
        self.intersection_cuts=0
        self.restricted_states=0

    def _can(self,t,state):
        key=(t,state)
        if key in self._value_cache:return self._value_cache[key]
        self.visited+=1
        if self.visited>self.node_cap:raise RuntimeError('node cap reached')
        if len(self.labels(state))<=1:
            self._value_cache[key]=True
            return True
        if t>=self.Q:
            self._value_cache[key]=False
            return False
        allowed,reason=forced_addresses(self.n,self.Q,t,state)
        if reason != 'allowed':
            if reason == 'pairwise':self.pair_cuts+=1
            else:self.intersection_cuts+=1
            self._value_cache[key]=False
            return False
        if allowed != (1<<self.n)-1:self.restricted_states+=1
        for m in range(self.n):
            if not (allowed & (1<<m)):continue
            possible=True
            for y in (0,1):
                child=self.next_state(state,t,m,y)
                if child is not None and not self._can(t+1,child):
                    possible=False
                    break
            if possible:
                self._choice[key]=m
                self._value_cache[key]=True
                return True
        self._value_cache[key]=False
        return False
