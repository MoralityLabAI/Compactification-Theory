"""OIC-25 independent Python implementation of the five-state fault game.

The finite observation state and transitions come from the previously
published OIC-23 model; action enumeration, canonicalization, pairwise
impossibility pruning and memoized search here are separately implemented.
"""
from __future__ import annotations
from oic23_exact import ExactFaultGame

class PythonPrunedGame(ExactFaultGame):
    def __init__(self, Q: int, node_cap: int = 1000000):
        super().__init__(5, Q, 1, 0, node_cap=node_cap)
        self.pair_pruned=0
        self._future_diff=[
            [
                [sum(((a%5+t*(a//5))%5)!=((b%5+t*(b//5))%5) for t in range(k,Q))
                 for b in range(25)]
                for a in range(25)
            ]
            for k in range(Q+1)
        ]

    @staticmethod
    def rotated(mask: int, offset: int) -> int:
        out=0
        for d in range(5):
            row=(mask>>(d*5))&31
            row=((row<<offset)|(row>>(5-offset)))&31 if offset else row
            out|=row<<(d*5)
        return out

    def canonical_offset(self, state):
        best=state
        chosen=0
        for c in range(1,5):
            alt=(self.rotated(state[0],c), self.rotated(state[1],c))
            if alt<best:best=alt;chosen=c
        return best,chosen

    def canonical(self, state):
        return self.canonical_offset(state)[0]

    def certificate(self):
        if not self._value_cache.get((0,self.canonical(self.root))):return None
        def build(t,state):
            labels=self.labels(state)
            if len(labels)<=1:return {'shift':labels[0]}
            canonical,c=self.canonical_offset(state)
            query=(self._choice[(t,canonical)]-c)%5
            children={}
            for y in (0,1):
                nxt=self.next_state(state,t,query,y)
                if nxt is not None:children[str(y)]=build(t+1,nxt)
            return {'query':query,'children':children}
        return build(0,self.root)

    def pair_obstructed(self, state,t):
        zero,one=state
        items=[]
        bits=zero|one
        while bits:
            least=bits&-bits
            index=least.bit_length()-1
            items.append((index, bool(zero&least)))
            bits-=least
        diff=self._future_diff[t]
        for i,(a,a_zero) in enumerate(items):
            for b,b_zero in items[i+1:]:
                if a//5!=b//5 and diff[a][b] <= a_zero+b_zero:
                    return True
        return False

    def _can(self,t,state):
        state=self.canonical(state)
        key=(t,state)
        if key in self._value_cache:return self._value_cache[key]
        self.visited+=1
        if self.visited>self.node_cap:raise RuntimeError('node cap exceeded')
        if len(self.labels(state))<=1:
            self._value_cache[key]=True
            return True
        if t>=self.Q or self.pair_obstructed(state,t):
            self.pair_pruned+=1
            self._value_cache[key]=False
            return False
        proposals=[]
        for m in range(5):
            left=self.next_state(state,t,m,0)
            right=self.next_state(state,t,m,1)
            children=tuple(child for child in (left,right) if child is not None)
            sizes=[sum(x.bit_count() for x in child) for child in children]
            proposals.append((max(sizes)*100+(abs(sizes[0]-sizes[1]) if len(sizes)>1 else 0),m,children))
        for _,m,children in sorted(proposals):
            branches=sorted(children,key=lambda s:-((s[0]|s[1]).bit_count()))
            if all(self._can(t+1,child) for child in branches):
                self._choice[key]=m
                self._value_cache[key]=True
                return True
        self._value_cache[key]=False
        return False

if __name__=='__main__':
    import sys,time
    for Q in map(int,sys.argv[1:] or ['11','12','13']):
        g=PythonPrunedGame(Q)
        begin=time.monotonic()
        ok=g.solve()
        print('Q',Q,'SAT?',ok,'visited',g.visited,'pairs_pruned',g.pair_pruned,'seconds',round(time.monotonic()-begin,2),flush=True)
