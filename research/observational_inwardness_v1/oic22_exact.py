"""OIC-22: exact one-fault optimal active sensing via two mismatch layers.

The one-error oracle assumes exactly one bit substitution at a uniformly chosen
position among Q addressed observations, uniform (shift,initial phase), and a
clean deterministic cyclic transition. The one-erasure oracle analogously
assumes exactly one marked erasure at a uniformly chosen position. Neither is
a worst-case robustness guarantee. One report bit and a selected sensor address
are charged per time step, even when the observed report is erased.
"""
from __future__ import annotations
from functools import lru_cache
from oic21_exact import ExactActiveOracle

class FaultOracle:
    def __init__(self,n,Q,kind='flip'):
        if kind not in ('flip','erase'):raise ValueError('kind must be flip or erase')
        self.clean=ExactActiveOracle(n,Q)
        self.n=n;self.Q=Q;self.kind=kind
        self.all=self.clean.all
        self.dmask=self.clean.d_masks
    def after(self,M0,M1,t,m,y):
        # M0 = candidates requiring zero faults so far; M1 = candidates requiring exactly one.
        if y is None:
            if self.kind!='erase':raise ValueError('erasures only in erasure model')
            return 0,M0
        mask=self.clean.masks[t][m]
        match=mask if y else self.all^mask
        if self.kind=='flip':
            return M0&match,(M1&match)|(M0&(self.all^match))
        return M0&match,M1&match
    def successors(self,M0,M1,t,m):
        outcomes=(0,1,None) if self.kind=='erase' else (0,1)
        return tuple((y,*self.after(M0,M1,t,m,y)) for y in outcomes)
    def terminal(self,M1):
        return max((M1&dm).bit_count() for dm in self.dmask) if M1 else 0
    @lru_cache(None)
    def optimal_value(self,t,M0,M1):
        if t==self.Q:
            return self.terminal(M1)
        return max(self.action_value(t,M0,M1,m) for m in range(self.n))
    def action_value(self,t,M0,M1,m):
        return sum(self.optimal_value(t+1,a,b) for _,a,b in self.successors(M0,M1,t,m) if (a|b))
    def action_values(self,t,M0,M1):
        return tuple(self.action_value(t,M0,M1,m) for m in range(self.n))
    def optimal(self):
        return self.optimal_value(0,self.all,0)/(self.n*self.n*self.Q)
    def policy(self,t,M0,M1):
        vals=self.action_values(t,M0,M1)
        return vals.index(max(vals))
    @lru_cache(None)
    def guaranteed(self,t,M0,M1):
        """True iff a single policy identifies d on every legal 1-fault history.

        Terminal tests are for candidates with *exactly* one fault, hence M1.
        The strong at-most-one check can include M0 as well.
        """
        if t==self.Q:
            return sum(bool(M1&dm) for dm in self.dmask)<=1
        for m in range(self.n):
            if all(self.guaranteed(t+1,a,b) for _,a,b in self.successors(M0,M1,t,m) if a|b):
                return True
        return False
    def guaranteed_at_most_one(self,t,M0,M1):
        # no memoizer, used only for small n; at terminal allow both fault counts
        @lru_cache(None)
        def aux(t,a,b):
            if t==self.Q:return sum(bool((a|b)&dm) for dm in self.dmask)<=1
            return any(all(aux(t+1,x,y) for _,x,y in self.successors(a,b,t,m) if x|y)
                       for m in range(self.n))
        return aux(t,M0,M1)

if __name__=='__main__':
 for n in (3,4):
  for kind in ('flip','erase'):
   o=FaultOracle(n,n+2,kind)
   print(n,kind,o.optimal(),o.optimal_value.cache_info(),flush=True)


class MixedFaultOracle(FaultOracle):
    """Exact objective for mixture of clean (weight Q) and single-flip (weight 1).

    Equal prior probability for clean and exactly-one uniformly-positioned
    substitution. Root value / (2 * Q * n^2) is joint Bayes accuracy.
    """
    def __init__(self,n,Q):
        super().__init__(n,Q,'flip')
    def terminal_joint(self,M0,M1):
        return max((self.Q*(M0&dm).bit_count()+(M1&dm).bit_count()
                    for dm in self.dmask),default=0)
    @lru_cache(None)
    def optimal_value(self,t,M0,M1):
        if t==self.Q:return self.terminal_joint(M0,M1)
        return max(self.action_value(t,M0,M1,m) for m in range(self.n))
    def optimal(self):
        return self.optimal_value(0,self.all,0)/(2*self.n*self.n*self.Q)
