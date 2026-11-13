"""OIC-22 independent, torch/numpy-free replay of frozen policy models.

Reconstructs exact finite belief-mismatch masks, seven OIC-21 features, and
145-weight sensor-policy decisions using Python floats. Checks every registered
initial shift/phase and all single-fault locations. Uses no imported OIC-22
training, posterior, or scorer implementation.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,math,struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

HERE=Path(__file__).resolve().parent
RES=HERE/'results'

def check(ok,msg):
 if not ok:raise AssertionError(msg)

def fp32(x):return struct.unpack('f',struct.pack('f',x))[0]

def get_archive():
 raw=base64.b64decode((RES/'oic22_fits.json.gz.b64').read_bytes().strip(),validate=True)
 return json.loads(gzip.decompress(raw))

def ent(values):
 total=sum(values)
 if total<=0:return 0.
 return -sum(v/total*math.log2(v/total) for v in values if v>0)

class IndependentObserver:
 def __init__(self,n,weights):
  self.n=n;self.Q=n+2
  self.w=weights
  self.d=[d for d in range(n) for z in range(n)]
  self.z=[z for d in range(n) for z in range(n)]
  self.groups=[math.gcd(n,d) for d in self.d]
  self.all=(1<<(n*n))-1
  self.match=[[sum(1<<i for i in range(n*n) if (self.z[i]+t*self.d[i])%n==m)
               for m in range(n)] for t in range(self.Q)]
 def posterior_mass(self,t,a,b):
  zero=(2*self.Q-t)/(2*self.Q)
  once=1/(2*self.Q)
  raw=[zero if (a>>i)&1 else once if (b>>i)&1 else 0. for i in range(self.n*self.n)]
  total=sum(raw)
  if not total:raise ValueError('impossible history')
  return [x/total for x in raw]
 def features(self,t,weight):
  n=self.n
  byshift=[0.]*n;bygroup=[0.]*(n+1)
  for i,w in enumerate(weight):byshift[self.d[i]]+=w;bygroup[self.groups[i]]+=w
  base=max(byshift);base_g=max(bygroup);H=ent(byshift);G=ent(bygroup)
  out=[]
  for m in range(n):
   mask=self.match[t][m]
   pos=[0.]*n;neg=[0.]*n;pos_g=[0.]*(n+1);neg_g=[0.]*(n+1)
   p=0.
   for i,w in enumerate(weight):
    if (mask>>i)&1:
     p+=w;pos[self.d[i]]+=w;pos_g[self.groups[i]]+=w
    else:
     neg[self.d[i]]+=w;neg_g[self.groups[i]]+=w
   p=max(0.,min(1.,p))
   score=max(pos)+max(neg)-base
   gcdscore=max(pos_g)+max(neg_g)-base_g
   hgain=max(0.,(H-p*ent(pos)-(1-p)*ent(neg))/math.log2(n))
   ggain=max(0.,(G-p*ent(pos_g)-(1-p)*ent(neg_g))/math.log2(n))
   row=(score,hgain,p*(1-p),gcdscore,ggain,p,1/(1+2*abs(p-.5)))
   out.append(tuple(fp32(x) for x in row))
  return out
 def score(self,features):
  w1=self.w['network.0.weight'];b1=self.w['network.0.bias']
  w2=self.w['network.2.weight'];b2=self.w['network.2.bias'][0]
  scores=[]
  for x in features:
   hidden=[math.tanh(sum(w*xj for w,xj in zip(row,x))+bias) for row,bias in zip(w1,b1)]
   scores.append(sum(w*hh for w,hh in zip(w2[0],hidden))+b2)
  return scores
 @lru_cache(None)
 def state(self,history):
  t=len(history)
  if t==0:a,b=self.all,0
  else:
   prev0,prev1,m=self.state(history[:-1]);raw=history[-1]
   if raw is None:a,b=prev0,prev1
   else:
    ms=self.match[t-1][m]
    match=ms if raw else self.all^ms
    a=prev0&match
    b=(prev1&match)|(prev0&(self.all^match))
  if t==self.Q:return a,b,None
  posterior=self.posterior_mass(t,a,b)
  feats=self.features(t,posterior)
  scores=self.score(feats)
  m=max(range(self.n),key=lambda i:(round(scores[i],4),-i))
  return a,b,m
 def episode(self,d,z,kind,pos):
  hist=()
  for t in range(self.Q):
   _,_,marker=self.state(hist)
   bit=int((z+t*d)%self.n==marker)
   if kind=='flip' and t==pos:bit=1-bit
   elif kind=='erase' and t==pos:bit=None
   hist+=(bit,)
  a,b,_=self.state(hist)
  weights=self.posterior_mass(self.Q,a,b)
  sums=[sum(weights[d*self.n:(d+1)*self.n]) for d in range(self.n)]
  return int(max(range(self.n),key=lambda j:(sums[j],-j))==d)
 def panel(self,kind):
  count=0;correct=0
  positions=(None,) if kind=='clean' else tuple(range(self.Q))
  for pos in positions:
   for d in range(self.n):
    for z in range(self.n):
     correct+=self.episode(d,z,kind,pos);count+=1
  return correct,count


def verify(deep=False):
 manifest=json.loads((RES/'oic22_summary.json').read_text())
 checks=manifest['sha256']
 for filename,digest in checks.items():
  if filename=='upstream_oic21_faults.json':
   filename='results/oic21_faults.json'
  path=HERE/filename
  check(path.exists(),f'missing pinned {filename}')
  check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,f'digest mismatch for {filename}')
 archive=get_archive();fits=archive['fits']
 check(archive['tag']=='OIC-22-FAULT-AWARE-01','receipt tag mismatch')
 check(len(fits)==10,'ten trained policy fits')
 check({(f['meta']['seed'],f['meta']['split']) for f in fits}==
       {(i,s) for i in range(5) for s in ('full','heldout')},'fit seed or split coverage')
 for f in fits:
  meta=f['meta'];check(meta['train_steps']==500 and meta['batch']==96,'training budget drift')
  check(meta['parameter_count']==145,'architecture drift')
  check(meta['alpha']==.5,'fault prior drift')
  check(set(f['evaluation'])=={'4','5','6','7','8'},'evaluation coverage drift')
  for n in (4,6,8):
   rows=f['evaluation'][str(n)]
   check(set(r['kind'] for r in rows)=={'clean','flip','erase'},'fault kind coverage')
   for row in rows:
    q=n+2;expected=n*n*(q if row['kind']!='clean' else 1)
    check(row['episodes']==expected,'fault population mismatch')
    check(0<=row['correct']<=expected and abs(row['accuracy']-row['correct']/expected)<1e-12,
          'fault reported rate mismatch')
 for row in manifest['results']:
  n=row['n'];split=row['split']
  sub=sorted((f for f in fits if f['meta']['split']==split),key=lambda f:f['meta']['seed'])
  for kind in ('clean','flip','erase'):
   old=row['oic22_'+kind]
   matching=[next(r for r in f['evaluation'][str(n)] if r['kind']==kind) for f in sub]
   check(sum(m['correct'] for m in matching)==old['correct'],'aggregate numerator incorrect')
   check(sum(m['episodes'] for m in matching)==old['episodes'],'aggregate denominator incorrect')
 # Independent n4 replay for every fit and full n6,n8 fault panel for one predeclared fit.
 # 'deep' checks every saved panel from all 10 fits and all 1-fault positions.
 checks_run=0
 for fit in fits:
  s=fit['meta']['seed'];split=fit['meta']['split']
  for n in ((4,6,8) if deep else (4,) if s in (0,1) else ()):
   obs=IndependentObserver(n,fit['weights'])
   for kind in ('clean','flip','erase'):
    actual=obs.panel(kind)
    row=next(p for p in fit['evaluation'][str(n)] if p['kind']==kind)
    check(actual==(row['correct'],row['episodes']),
          f'independent posterior decision mismatch n={n},seed={s},split={split},kind={kind}, expected={(row["correct"],row["episodes"])},actual={actual}')
    checks_run+=1
 return {'status':'PASS','model_fits':len(fits),'independent_panel_replays':checks_run,
         'full_population_fault_panels_archived':len(fits)*3*3}


def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--deep',action='store_true');a=p.parse_args()
 print(json.dumps(verify(deep=a.deep),sort_keys=True,indent=2))
if __name__=='__main__':main()
