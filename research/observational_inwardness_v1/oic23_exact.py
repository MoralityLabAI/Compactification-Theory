"""Small finite exact adversarial sensor game, for OIC-23 feasibility certificates.

At most e substituted binary reports and r marked erasures.  Each transcript
is replayed with its own adaptive address choices; the controller sees only
received reports. No statistical assumptions about the hidden pair.
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
from time import perf_counter

class ExactFaultGame:
    def __init__(self,n:int,Q:int,e:int=1,r:int=0,node_cap:int=500_000):
        if not (2<=n<=12 and Q>=0 and 0<=e<=2 and 0<=r<=2):raise ValueError('invalid parameters')
        self.n,self.Q,self.e,self.r=n,Q,e,r
        self.width=(e+1)*(r+1)
        self.all_mask=(1<<(n*n))-1
        self.dmask=tuple(sum(1<<(d*n+z) for z in range(n)) for d in range(n))
        self.hit=tuple(tuple(sum(1<<(d*n+z) for d in range(n) for z in range(n) if (z+t*d)%n==m)
                                  for m in range(n)) for t in range(Q))
        self.root=(self.all_mask,)+(0,)*(self.width-1)
        self.node_cap=node_cap
        self.visited=0
        self._value_cache={}
        self._choice={}
    def idx(self,i,j):return i*(self.r+1)+j
    def labels(self,state):
        allc=0
        for m in state:allc |=m
        return tuple(d for d,mask in enumerate(self.dmask) if allc &mask)
    def next_state(self,state,t,m,y):
        hit=self.hit[t][m]
        no=self.all_mask^hit
        if y is None and not self.r:return None
        out=[0]*self.width
        for i in range(self.e+1):
            for j in range(self.r+1):
                mask=state[self.idx(i,j)]
                if not mask:continue
                if y is None:
                    if j<self.r:out[self.idx(i,j+1)] |=mask
                else:
                    clean = hit if y==1 else no
                    out[self.idx(i,j)] |=mask&clean
                    if i<self.e:out[self.idx(i+1,j)] |=mask&(self.all_mask^clean)
        if not any(out):return None
        return tuple(out)
    def solve(self):
        self.visited=0
        self._value_cache={}
        self._choice={}
        return self._can(0,self.root)
    def _can(self,t,state):
        key=(t,state)
        if key in self._value_cache:return self._value_cache[key]
        self.visited+=1
        if self.visited>self.node_cap:raise RuntimeError(f'cap exceeded: {self.visited}')
        if len(self.labels(state))<=1:
            self._value_cache[key]=True
            return True
        if t>=self.Q:
            self._value_cache[key]=False
            return False
        for m in range(self.n):
            allpass=True
            for y in ((0,1,None) if self.r else (0,1)):
                child=self.next_state(state,t,m,y)
                if child is None:continue
                if not self._can(t+1,child):
                    allpass=False
                    break
            if allpass:
                self._choice[key]=m
                self._value_cache[key]=True
                return True
        self._value_cache[key]=False
        return False
    def certificate(self):
        if not self._value_cache.get((0,self.root)):return None
        def build(t,state):
            ds=self.labels(state)
            if len(ds)<=1:return {'shift':ds[0]}
            m=self._choice[(t,state)]
            children={}
            for y in ((0,1,None) if self.r else (0,1)):
                child=self.next_state(state,t,m,y)
                if child is not None:children['?' if y is None else str(y)]=build(t+1,child)
            return {'query':m,'children':children}
        return build(0,self.root)
    def verify_tree(self,tree):
        def walk(t,state,node):
            if not isinstance(node,dict):raise ValueError('policy-tree node must be an object')
            if 'shift' in node:
                if len(self.labels(state))!=1 or node['shift']!=self.labels(state)[0]:
                    raise AssertionError('policy leaf declares an incorrect or ambiguous shift')
                return 1
            if t>=self.Q:raise AssertionError('policy exceeds registered query budget')
            m=node['query']
            if not isinstance(m,int) or not 0<=m<self.n:
                raise AssertionError('invalid sensor address')
            expected={}
            for y in ((0,1,None) if self.r else (0,1)):
                child=self.next_state(state,t,m,y)
                if child is not None:expected['?' if y is None else str(y)]=child
            if set(node['children'])!=set(expected):raise AssertionError('policy tree omits/adds feasible report branches')
            return 1+sum(walk(t+1,s,node['children'][y]) for y,s in expected.items())
        return walk(0,self.root,tree)

if __name__=='__main__':
    for n in (2,3,4):
        for e,r in ((1,0),(0,1),(1,1)):
            lo=n*(e+1)+r
            for Q in range(lo,lo+4):
                g=ExactFaultGame(n,Q,e,r,node_cap=200000)
                start=perf_counter()
                try:result=g.solve(); status='YES' if result else 'NO'
                except RuntimeError:status='CAP';result=False
                print(f'n={n} e={e} r={r} Q={Q} => {status} nodes={g.visited} time={perf_counter()-start:.3f}s',flush=True)
                if result:
                    cert=g.certificate()
                    print(' tree nodes',g.verify_tree(cert),flush=True)
                    break
                if status=='CAP':break
