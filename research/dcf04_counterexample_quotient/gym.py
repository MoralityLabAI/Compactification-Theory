"""DCF-04 unknown-cardinality DFA learning; no semantic state labels."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import collections as C, itertools as I, json, time, pathlib, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
A=range(4); KS=(4,8,12,16); BUDGET=2000; EQCAP=20

def out(t,b,w):
 s=0
 for a in w:s=int(t[s][a])
 return int(b[s])
def eq(t,b,u,c):
 q=C.deque([(0,0,())]);seen=set();edges=0
 while q:
  s,h,w=q.popleft()
  if (s,h) in seen:continue
  seen.add((s,h))
  if b[s]!=c[h]:return w,len(seen),edges
  for a in A:q.append((int(t[s][a]),int(u[h][a]),w+(a,)));edges+=1
 return None,len(seen),edges
def reach(t):
 q=[0];seen={0}
 for s in q:
  for h in t[s]:
   h=int(h)
   if h not in seen:seen.add(h);q.append(h)
 return q
def minimal(t,b):
 q=reach(t);p={s:int(b[s]) for s in q}
 while True:
  sig={s:(int(b[s]),*(p[int(h)] for h in t[s])) for s in q};keys={v:i for i,v in enumerate(sorted(set(sig.values())))};r={s:keys[v] for s,v in sig.items()}
  if len(set(r.values()))==len(set(p.values())):return len(set(r.values()))
  p=r
def gen(n,kind,rep):
 seed=2026112500+100*n+10*(kind=='map')+rep;g=np.random.default_rng(seed)
 for attempt in range(1,10001):
  t=np.stack([g.permutation(n) for _ in A],1) if kind=='perm' else g.integers(n,size=(n,4));b=g.integers(2,size=n)
  if len(reach(t))==n and minimal(t,b)==n:return t,b,seed,attempt
 raise RuntimeError('generation cap')
def data(t,b,g,n,l):
 x=g.integers(4,size=(n,l));s=np.zeros(n,dtype=int);y=[]
 for a in x.T:s=t[s,a];y.append(b[s])
 return torch.from_numpy(x),torch.from_numpy(np.stack(y,1).astype('float32'))
class Soft(nn.Module):
 def __init__(self,k):
  super().__init__();self.k=k;self.z=nn.Parameter(torch.randn(4,k,k)*.6);self.b=nn.Parameter(torch.randn(k)*.3)
 def forward(self,x):
  m=self.z.softmax(-1);h=F.one_hot(torch.zeros(len(x),dtype=torch.long),self.k).float();p=[]
  for a in x.T:h=torch.bmm(h[:,None],m[a]).squeeze(1);p.append(h@self.b.sigmoid())
  return torch.stack(p,1).clamp(1e-6,1-1e-6)
 def hard(self):return self.z.detach().argmax(-1).T.numpy(),(self.b.detach().numpy()>0).astype(int)
def neural(t,b,seed,steps,folder):
 x,y=data(t,b,np.random.default_rng(seed+90),steps*48,4);vx,vy=data(t,b,np.random.default_rng(seed+91),256,8);rows=[];models={};initial=torch.tensor(float(b[0]))
 for k in KS:
  torch.manual_seed(seed+k);m=Soft(k);opt=torch.optim.AdamW(m.parameters(),lr=.025);start=time.perf_counter()
  for i in range(steps):
   sl=slice(i*48,(i+1)*48);opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy(m(x[sl]),y[sl])+.2*F.binary_cross_entropy_with_logits(m.b[0],initial);loss.backward();opt.step()
  with torch.no_grad():v=float(F.binary_cross_entropy(m(vx),vy));u,c=m.hard();ce,p,e=eq(t,b,u,c)
  rows.append(dict(k=k,val_bce=v,exact=ce is None,ce=ce,train_s=time.perf_counter()-start));models[k]=(m,u,c);torch.save(m.state_dict(),folder/f'{seed}_{k}.pt')
 best=min(rows,key=lambda r:(r['val_bce'],r['k']));m,u,c=models[best['k']];tests=[]
 with torch.no_grad():
  for l in (4,16,64):
   x,y=data(t,b,np.random.default_rng(seed+500+l),512,l);s=np.zeros(len(x),dtype=int);ps=[]
   for a in x.numpy().T:s=u[s,a];ps.append(c[s])
   hard=np.stack(ps,1)==y.numpy();soft=(m(x)>.5)==y.bool();tests.append(dict(length=l,n=len(x),hard_trace=int(hard.all(1).sum()),soft_trace=int(soft.all(1).sum()),hard_token=float(hard.mean()),soft_token=float(soft.float().mean())))
 return u,c,dict(candidates=rows,selected=best['k'],exact=best['exact'],tests=tests,train_label_draws=steps*48*4+1,val_label_draws=2048,total_label_uses=4*(steps*48*4+steps))
def probes(u,c):
 pairs=[(s,h) for s,h in I.combinations(reach(u),2) if c[s]==c[h]];ws=[];candidates=[w for l in range(1,5) for w in I.product(A,repeat=l)]
 def at(s,w):
  for a in w:s=int(u[s,a])
  return c[s]
 while pairs and len(ws)<6:
  w=max(candidates,key=lambda w:sum(at(s,w)!=at(h,w) for s,h in pairs));r=[(s,h) for s,h in pairs if at(s,w)==at(h,w)]
  if len(r)==len(pairs):break
  ws.append(w);pairs=r;candidates.remove(w)
 return ws
def closure(ws):return list(dict.fromkeys(w[i:] for w in ws for i in range(len(w)+1)))
class Budget(Exception):pass
def learn(t,b,mode,seed,warm=(),proposal=None):
 g=np.random.default_rng(seed);cache={};S=[()];E=list(dict.fromkeys([(),*warm]));u=c=None;history=[];status='unknown';counts=C.Counter();start=time.perf_counter()
 def mq(w):
  if w not in cache:
   if len(cache)>=BUDGET:raise Budget('mq')
   cache[w]=out(t,b,w)
  return cache[w]
 def check(u,c):
  if counts['eq']>=EQCAP:raise Budget('eq')
  counts['eq']+=1
  if mode=='exact':
   w,p,e=eq(t,b,u,c);counts['eq_pairs']+=p;counts['eq_edges']+=e;return w
  for l in (1,2,4,8,16,32):
   for w in g.integers(4,size=(64,l)):
    w=tuple(map(int,w));counts['sample_attempts']+=1;counts['sample_tokens']+=len(w)
    if mq(w)!=out(u,c,w):return w
  return None
 def row(w):return tuple(mq(w+e) for e in E)
 def pref(w):
  for i in range(len(w)+1):
   if w[:i] not in S:S.append(w[:i])
 try:
  if proposal is not None:
   pu,pc=proposal;w=check(pu,pc);history.append(dict(phase='proposal',ce=w,mq=len(cache)))
   if w is None:u,c=pu,pc;status='accepted'
   else:pref(w)
  while status!='accepted':
   sr={s:row(s) for s in S};ext={s+(a,):row(s+(a,)) for s in S for a in A};rs=set(sr.values());bad=next((w for w,r in ext.items() if r not in rs),None)
   if bad is not None:S.append(bad);continue
   groups=C.defaultdict(list)
   for s,r in sr.items():groups[r].append(s)
   fix=None
   for group in groups.values():
    for s,h in I.combinations(group,2):
     for a in A:
      x,y=ext[s+(a,)],ext[h+(a,)]
      if x!=y:fix=(a,)+E[next(i for i,(p,q) in enumerate(zip(x,y)) if p!=q)];break
     if fix is not None:break
    if fix is not None:break
   if fix is not None:assert fix not in E;E.append(fix);continue
   reps=list(dict.fromkeys(sr.values()));idx={r:i for i,r in enumerate(reps)};rep={r:s for s,r in reversed(list(sr.items()))};u=np.array([[idx[ext[rep[r]+(a,)]] for a in A] for r in reps]);c=np.array([r[0] for r in reps]);w=check(u,c);history.append(dict(phase='table',states=len(u),ce=w,mq=len(cache)))
   if w is None:status='accepted';break
   pref(w)
 except Budget as e:status='budget_'+str(e)
 w,p,e=eq(t,b,u,c) if u is not None else ((),0,0)
 return dict(status=status,exact=w is None,final_ce=w,audit_pairs=p,audit_edges=e,states=None if u is None else len(u),minimal=None if u is None else minimal(u,c),mq=len(cache),mq_tokens=sum(map(len,cache)),seconds=time.perf_counter()-start,history=history,counts=dict(counts),table=None if u is None else u.tolist(),accept=None if c is None else c.tolist())
def main():
 root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);folder=root/'checkpoints';folder.mkdir(exist_ok=True);steps=int(sys.argv[2]) if len(sys.argv)>2 else 200;rows=[]
 for n,kind,rep in I.product(KS,('perm','map'),(0,1)):
  t,b,seed,attempt=gen(n,kind,rep);u,c,nnr=neural(t,b,seed,steps,folder);ss=probes(u,c);g=np.random.default_rng(seed+5000);rr=[tuple(map(int,g.integers(4,size=len(w)))) for w in ss];arms={}
  for mode in ('exact','sampled'):
   for arm,warm,pp in [('cold',(),None),('random',closure(rr),None),('neural_suffix',closure(ss),None),('neural_repair',closure(ss),(u,c))]:arms[mode+'_'+arm]=learn(t,b,mode,seed+7000,warm,pp)
  rows.append(dict(n=n,kind=kind,rep=rep,seed=seed,generation_attempts=attempt,oracle_table=t.tolist(),oracle_accept=b.tolist(),neural=nnr,neural_suffixes=ss,random_suffixes=rr,arms=arms));(root/'receipt.json').write_text(json.dumps(dict(config=dict(steps=steps,ks=KS,mq_budget=BUDGET,eq_cap=EQCAP),rows=rows),indent=2)+'\n')
  print(n,kind,rep,[(a,r['exact'],r['mq']) for a,r in arms.items()],flush=True)
if __name__=='__main__':main()
