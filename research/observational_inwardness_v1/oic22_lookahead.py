"""Finite-horizon truncated joint-Bayes lookahead with exact one-flip bookkeeping."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from oic22_exact import MixedFaultOracle

class FaultAwarePlanner:
 def __init__(self,n,depth=2,Q=None,clean_weight=1,flip_weight=1):
  self.n=n;self.Q=Q or n+2;self.depth=depth;self.clean_weight=clean_weight;self.flip_weight=flip_weight
  if self.Q>n+2:raise ValueError('current benchmark fixes Q=n+2')
  if clean_weight<0 or flip_weight<0 or (not clean_weight and not flip_weight):raise ValueError('invalid weights')
  self.o=MixedFaultOracle(n,self.Q)
 @lru_cache(None)
 def partial(self,t,M0,M1):
  """Integer proportional to Bayes accuracy with optional future flip positions."""
  return max(((self.clean_weight*self.Q+self.flip_weight*(self.Q-t))*(M0&dm).bit_count()+self.flip_weight*(M1&dm).bit_count()
              for dm in self.o.dmask),default=0)
 @lru_cache(None)
 def value(self,t,M0,M1,depth):
  if t==self.Q or depth==0:return self.partial(t,M0,M1)
  return max(self.avalue(t,M0,M1,depth,m) for m in range(self.n))
 def avalue(self,t,M0,M1,depth,m):
  return sum(self.value(t+1,a,b,depth-1)
             for _,a,b in self.o.successors(M0,M1,t,m) if a|b)
 @lru_cache(None)
 def state(self,history):
  t=len(history)
  if t==0:c0,c1=self.o.all,0
  else:
   c0p,c1p,m=self.state(history[:-1]);c0,c1=self.o.after(c0p,c1p,t-1,m,history[-1]) if history[-1] is not None else (c0p,c1p)
  if t==self.Q:return c0,c1,None
  depth=min(self.depth,self.Q-t)
  scores=[self.avalue(t,c0,c1,depth,m) for m in range(self.n)]
  return c0,c1,int(np.argmax(scores))
 def post(self,history):
  c0,c1,_=self.state(history); t=len(history)
  shifts=np.array([self.Q*self.clean_weight*(c0&dm).bit_count()+self.flip_weight*(c1&dm).bit_count() for dm in self.o.dmask],dtype=float)
  shifts/=shifts.sum()
  return shifts
 def episode(self,d,z,kind='clean',pos=None):
  history=()
  for t in range(self.Q):
   _,_,m=self.state(history)
   y=int((z+d*t)%self.n==m)
   if kind=='flip' and pos==t:y=1-y
   if kind=='erase' and pos==t:y=None
   history+=(y,)
  p=self.post(history)
  return int(p.argmax()==d)
 def panels(self):
  out=[]
  for kind in ('clean','flip','erase'):
   total=correct=0
   for pos in ((None,) if kind=='clean' else range(self.Q)):
    for d in range(self.n):
     for z in range(self.n):
      correct+=self.episode(d,z,kind,pos);total+=1
   out.append((kind,correct,total,round(correct/total,5)))
  return out

if __name__=='__main__':
 import time
 for n in (4,6,8):
  for depth in (1,2,3):
   p=FaultAwarePlanner(n,depth)
   start=time.time()
   print(n,depth,p.panels(),round(time.time()-start,2),'cache',p.value.cache_info().currsize,flush=True)
