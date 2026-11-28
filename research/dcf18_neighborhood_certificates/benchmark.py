"""Exact integer conditional neighborhood certificates, fresh synthetic graphs."""
import itertools as it, random, time, json, math, statistics
from fractions import Fraction
from pathlib import Path
import numpy as np

def fixture(n,family,seed):
 r=random.Random(seed); edges=set()
 def add(a,b):
  if a!=b:edges.add(tuple(sorted((a,b))))
 for i in range(n-1):add(i,i+1)
 if family=='cycle':add(0,n-1)
 if family=='ladder':
  for i in range(n//2):add(i,i+n//2)
 if family=='chords':
  for i in range(0,n-3,3):add(i,i+3)
 q=r.randrange(n); unary=[[r.randint(2,9),r.randint(2,9)] for _ in range(n)]
 # Weak interactions with heterogeneous signed dependencies.
 factors=[((i,),tuple(v)) for i,v in enumerate(unary)]
 for a,b in sorted(edges):factors.append(((a,b),tuple(r.randint(3,8) for _ in range(4))))
 return factors,q

def eliminate(factors,q):
 # exact integer mass, generic table indexing; no floating point
 live=[(tuple(s),list(v)) for s,v in factors]; vs=set(x for s,_ in live for x in s)
 while vs-{q}:
  v=min(vs-{q},key=lambda x:(len(set(y for s,_ in live if x in s for y in s)),x)); bucket=[(s,t) for s,t in live if v in s];live=[(s,t) for s,t in live if v not in s];u=tuple(sorted(set(y for s,_ in bucket for y in s)-{v})); out=[]
  for bits in it.product((0,1),repeat=len(u)):
   a=dict(zip(u,bits)); mass=0
   for b in (0,1):
    a[v]=b; prod=1
    for s,t in bucket:
     ix=0
     for x in s:ix=2*ix+a[x]
     prod*=t[ix]
    mass+=prod
   out.append(mass)
  live.append((u,out));vs.remove(v)
 m=[1,1]
 for s,t in live:
  for b in (0,1):m[b]*=t[b] if s else t[0]
 return Fraction(m[1],sum(m))

def bounds(factors,q,region,cap=8):
 touched=[(s,v) for s,v in factors if set(s)&region]; boundary=sorted(set(x for s,_ in touched for x in s)-region)
 if len(boundary)>cap:return None
 lo=Fraction(1);hi=Fraction(0)
 for bits in it.product((0,1),repeat=len(boundary)):
  a=dict(zip(boundary,bits)); local=[]
  for s,t in touched:
   u=tuple(x for x in s if x in region);vals=[]
   for bs in it.product((0,1),repeat=len(u)):
    d={**a,**dict(zip(u,bs))};ix=0
    for x in s:ix=2*ix+d[x]
    vals.append(t[ix])
   local.append((u,vals))
  p=eliminate(local,q);lo=min(lo,p);hi=max(hi,p)
 return lo,hi

def action(p):return 1 if p>=Fraction(4,5) else 0 if p<=Fraction(1,5) else -1

def certified(interval):
 if interval is None:return None
 lo,hi=interval
 return action(lo) if action(lo)==action(hi) else None

def regions(f,q):
 region={q};yield region.copy()
 for _ in range(2):
  region|={x for s,_ in f if set(s)&region for x in s};yield region.copy()

def brute(f,q,n):
 m=[0,0]
 for b in it.product((0,1),repeat=n):
  p=1
  for s,t in f:
   ix=0
   for x in s:ix=2*ix+b[x]
   p*=t[ix]
  m[b[q]]+=p
 return Fraction(m[1],sum(m))

def run():
 rows=[];audit=0
 for n,fam,seed in it.product((12,24,48),('cycle','ladder','chords'),range(24)):
  f,q=fixture(n,fam,180000+n*1000+100*('cycle','ladder','chords').index(fam)+seed)
  t=time.perf_counter_ns();p=eliminate(f,q);full=time.perf_counter_ns()-t
  if n==12 and seed<4:assert brute(f,q,n)==p;audit+=1
  intervals=[];times=[];t=time.perf_counter_ns()
  for reg in regions(f,q):
   b=bounds(f,q,reg);times.append(time.perf_counter_ns()-t)
   if b is not None:assert b[0]<=p<=b[1]
   intervals.append(b)
  for depth in range(3):
   hit=next((i for i,b in enumerate(intervals[:depth+1]) if certified(b) is not None),None)
   a=action(p) if hit is None else certified(intervals[hit]);assert a==action(p)
   rows.append(dict(n=n,family=fam,seed=seed,depth=depth,certified=hit is not None,hit=hit,near=min(abs(float(p)-.2),abs(float(p)-.8))<=.05,posterior=float(p),action=a,full_ns=full,hybrid_ns=times[depth]+full if hit is None else times[hit]))
 print('audit',audit,'rows',len(rows));Path(__file__).with_name('receipt.json').write_text(json.dumps(dict(audit=audit,rows=rows),indent=2))
 for d in range(3):
  rr=[x for x in rows if x['depth']==d];print(d,sum(x['certified'] for x in rr),len(rr),sum(x['near'] for x in rr),sum(x['near'] and x['certified'] for x in rr),statistics.mean(x['full_ns'] for x in rr)/1e6,statistics.mean(x['hybrid_ns'] for x in rr)/1e6)
if __name__=='__main__':run()
