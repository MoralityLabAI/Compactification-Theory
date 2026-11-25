"""DCF-06: hidden graph discovery and adaptive cluster filtering."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,json,time,math,itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
SPEC=json.loads(Path(__file__).with_name('protocol.json').read_text())
def system(k,family,seed):
 g=np.random.default_rng(seed+1009*k);order=g.permutation(k);parent=np.full(k,-1)
 if family=='pairs':parent[order[::2]]=order[1::2];parent[order[1::2]]=order[::2]
 if family=='cycle':parent[order]=np.roll(order,-1)
 return parent,dict(idle=g.uniform(.015,.07,k),fail=g.uniform(.025,.15,k),err=g.uniform(.07,.20,k))
def simulate(j,c,r,parent,law,seed):
 g=np.random.default_rng(seed);n,l=j.shape;k=len(parent);h=g.integers(2,size=(n,k));y=np.empty_like(j);rows=np.arange(n)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];p=parent[a];ctrl=np.where(p<0,1,h[rows,np.maximum(p,0)]);noise=g.random(n)<np.where(b,law['fail'][a],law['idle'][a]);h[rows,a]^=(b*ctrl)^noise.astype(int);y[:,t]=h[rows,d]^(g.random(n)<law['err'][d]).astype(int)
 return j,c,r,y
def sample(k,parent,law,n,l,seed,policy='random'):
 g=np.random.default_rng(seed);j=g.integers(k,size=(n,l));c=g.integers(2,size=(n,l));r=np.where(g.random((n,l))<.5,j,g.integers(k,size=(n,l)))
 if policy=='probe_shift':
  for t in range(0,l-2,3):
   a=g.integers(k,size=n);b=(a+g.integers(1,k,size=n))%k;j[:,t:t+3]=np.stack([a,b,a],1);r[:,t:t+3]=j[:,t:t+3];c[:,t:t+3]=[0,0,1]
 return simulate(j,c,r,parent,law,seed+19999)
def probe(parent,law,j,p,n,seed):
 a=np.tile([j,p,j],(n,1));c=np.tile([0,0,1],(n,1));_,_,_,y=simulate(a,c,a,parent,law,seed);change=y[:,0]^y[:,2];ctrl=y[:,1];n0=int((ctrl==0).sum());n1=n-n0;s0=int(change[ctrl==0].sum());s1=int(change[ctrl==1].sum());return dict(n0=n0,n1=n1,s0=s0,s1=s1,score=s1/max(n1,1)-s0/max(n0,1))
def discover(k,parent,law,seed):
 screen=[];confirm=[];guess=np.full(k,-1);conf=np.zeros(k)
 for j in range(k):
  rr=[dict(j=j,p=p,**probe(parent,law,j,p,SPEC['screen_per_pair'],seed+50000+100*j+p)) for p in range(k) if p!=j];screen.extend(rr);p=max(rr,key=lambda z:z['score'])['p'];r=probe(parent,law,j,p,SPEC['confirm_per_target'],seed+80000+100*j+p)
  # One-sided simultaneous bound on difference of means across K independently confirmed targets.
  radius=sum(math.sqrt(math.log(2*k/SPEC['alpha'])/(2*max(r[q],1))) for q in ('n0','n1'));lower=r['score']-radius;accept=lower>SPEC['min_lower'];confirm.append(dict(j=j,p=p,lower=lower,accept=accept,**r))
  if accept:guess[j]=p;conf[j]=lower
 return guess,conf,dict(screen=screen,confirm=confirm,probe_words=k*(k-1)*SPEC['screen_per_pair']+k*SPEC['confirm_per_target'])
def passive(data,k):
 j,c,r,y=data;n,l=j.shape;last=np.full((n,k),-1);valid=np.zeros((n,k),bool);rows=np.arange(n);counts=np.zeros((k,k,2),int);sums=counts.copy()
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];mask=(b==1)&(d==a)&valid[rows,a]
  for q in range(k):
   m=mask&(a!=q)&valid[:,q];j0=a[m];g=last[m,q];z=last[rows[m],j0]^y[m,t];np.add.at(counts,(j0,np.full(len(g),q),g),1);np.add.at(sums,(j0,np.full(len(g),q),g),z)
  valid[rows,a]=False;valid[rows,d]=True;last[rows,d]=y[:,t]
 rate=sums/np.maximum(counts,1);s=rate[:,:,1]-rate[:,:,0];s[np.min(counts,axis=2)<16]=-1;np.fill_diagonal(s,-1);g=s.argmax(1);strength=s[np.arange(k),g];return np.where(strength>SPEC['passive_threshold'],g,-1),dict(counts=counts.tolist(),sums=sums.tolist(),score=s.tolist())
def shuffled(graph,seed):
 g=np.random.default_rng(seed);return np.array([-1 if p<0 else g.choice([q for q in range(len(graph)) if q!=j and q!=p]) for j,p in enumerate(graph)])
class Filter(nn.Module):
 def __init__(self,graph):
  super().__init__();k=len(graph);self.k=k;self.register_buffer('parent',torch.tensor(graph));self.noise=nn.Parameter(torch.full((k,2),-2.));self.em=nn.Parameter(torch.full((k,),-1.6))
 def forward(self,j,c,r,y):
  p=torch.full((len(j),self.k),.5);out=[];noise=.49*self.noise.sigmoid();err=.49*self.em.sigmoid()
  for a,b,d,z in zip(j.T,c.T,r.T,y.T):
   a,b,d,z=a[:,None],b[:,None],d[:,None],z[:,None];q=p.gather(1,a);par=self.parent[a];ctrl=torch.where(par<0,1.,p.gather(1,par.clamp_min(0)));f=noise[a,b];f=f+(1-2*f)*b*ctrl;p=p.scatter(1,a,q+f*(1-2*q));q=p.gather(1,d);e=err[d];out.append((e+(1-2*e)*q)[:,0]);l1=torch.where(z.bool(),1-e,e);l0=torch.where(z.bool(),e,1-e);p=p.scatter(1,d,q*l1/(q*l1+(1-q)*l0).clamp_min(1e-8))
  return torch.stack(out,1)
 def rates(self):
  f=(.49*self.noise.detach().sigmoid()).numpy();return dict(idle=f[:,0],fail=f[:,1],err=(.49*self.em.detach().sigmoid()).numpy())
class GRU(nn.Module):
 def __init__(self,k):
  super().__init__();self.k=k;self.gru=nn.GRU(2*k+3,64,batch_first=True);self.out=nn.Linear(64,1)
 def forward(self,j,c,r,y):
  prev=F.one_hot(F.pad(y[:,:-1],(1,0),value=2),3).float();x=torch.cat([F.one_hot(j,self.k).float(),F.one_hot(r,self.k).float(),c[:,:,None].float(),prev[:,:,:2]],-1);h,_=self.gru(x);return self.out(h).squeeze(-1).sigmoid()
def fit(model,data,seed,steps,lr):
 g=torch.Generator().manual_seed(seed+909);opt=torch.optim.Adam(model.parameters(),lr=lr);start=time.perf_counter();data=tuple(torch.from_numpy(x) for x in data)
 for _ in range(steps):
  i=torch.randint(len(data[0]),(SPEC['batch'],),generator=g);j,c,r,y=(x[i] for x in data);opt.zero_grad(set_to_none=True);p=model(j,c,r,y);loss=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y.float());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 return model.eval(),dict(train_s=time.perf_counter()-start,loss=float(loss.detach()),params=sum(x.numel() for x in model.parameters()))
def clusters(graph,conf,cap):
 groups=[{j} for j in range(len(graph))]
 for j,p in sorted(enumerate(graph),key=lambda x:(-conf[x[0]],x[0])):
  if p<0:continue
  a=next(g for g in groups if j in g);b=next(g for g in groups if p in g)
  if a is not b and len(a|b)<=cap:groups.remove(a);groups.remove(b);groups.append(a|b)
 return sorted([sorted(g) for g in groups])
def predict(data,graph,law,groups):
 """Product of cluster posteriors; exact internal transitions, external parent marginal."""
 j,c,r,y=data;n,l=j.shape;k=len(graph);out=np.empty((n,l));ps=[];bits=[];owner={};tables=[]
 for idx,g in enumerate(groups):
  s=np.arange(1<<len(g));b=((s[None,:]>>np.arange(len(g))[:,None])&1);bits.append(b);ps.append(np.full((n,len(s)),1/len(s)))
  for a,z in enumerate(g):owner[z]=(idx,a)
 for z in range(k):
  gi,a=owner[z];s=np.arange(1<<len(groups[gi]));flip=s^(1<<a);p=graph[z];internal=p>=0 and owner[p][0]==gi;cmd=s^(bits[gi][owner[p][1]]<<a) if internal else flip;tables.append((gi,a,flip,cmd,internal))
 for t in range(l):
  for z,(gi,a,flip,cmd,internal) in enumerate(tables):
   idx=np.flatnonzero(j[:,t]==z)
   if not len(idx):continue
   p=ps[gi][idx];b=c[idx,t];noise=np.where(b,law['fail'][z],law['idle'][z])[:,None]
   if graph[z]<0 or internal:
    q=np.where(b[:,None],p[:,cmd],p);ps[gi][idx]=(1-noise)*q+noise*q[:,flip]
   else:
    gj,a2=owner[graph[z]];ctrl=ps[gj][idx]@bits[gj][a2];f=noise+(1-2*noise)*b[:,None]*ctrl[:,None];ps[gi][idx]=(1-f)*p+f*p[:,flip]
  for z,(gi,a,_,_,_) in enumerate(tables):
   idx=np.flatnonzero(r[:,t]==z)
   if not len(idx):continue
   p=ps[gi][idx];e=law['err'][z];obs=e+(1-2*e)*bits[gi][a];out[idx,t]=p@obs;q=p*np.where(y[idx,t,None],obs,1-obs);ps[gi][idx]=q/q.sum(1,keepdims=True)
 return out
def score(p,data,oracle=None):
 p=np.clip(p.astype(float),1e-7,1-1e-7);y=data[-1];loss=-(y*np.log(p)+(1-y)*np.log1p(-p));ret=dict(nll=float(loss.mean()),n=len(y),tokens=y.size,accuracy=float(((p>=.5)==y).mean()),brier=float(((p-y)**2).mean()),word_nll=loss.mean(1).tolist())
 if oracle is not None:
  q=np.clip(oracle,1e-7,1-1e-7);kl=q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p));ret['oracle_kl']=float(kl.mean());ret['word_kl']=kl.mean(1).tolist()
 return ret
def run(k,family,seed,out):
 out.mkdir(parents=True,exist_ok=True);parent,law=system(k,family,seed);train=sample(k,parent,law,SPEC['train_words'],SPEC['train_length'],seed+10000);active,conf,disc=discover(k,parent,law,seed+1000000*k);pas,pm=passive(train,k);graphs={'independent':np.full(k,-1),'passive':pas,'active':active,'shuffled':shuffled(active,seed+333),'known':parent};fits={};models={};cache={}
 for name,g in graphs.items():
  key=tuple(g)
  if key not in cache:
   torch.manual_seed(seed);cache[key]=fit(Filter(g),train,seed,SPEC['filter_steps'],.035)
  m,meta=cache[key];models[name]=m;fits[name]=meta;torch.save(m.state_dict(),out/f'{name}.pt')
 torch.manual_seed(seed);gru,meta=fit(GRU(k),train,seed,SPEC['gru_steps'],.003);fits['gru64']=meta;torch.save(gru.state_dict(),out/'gru64.pt')
 val=sample(k,parent,law,SPEC['validation_words'],SPEC['validation_length'],seed+20000);valrows=[];candidates={}
 for name in graphs:
  g=graphs[name];m=models[name];caps=(1,2,4) if name in ('active','known') else (1,)
  for cap in caps:
   group=clusters(g,conf if name=='active' else np.ones(k),cap);coords=sum((1<<len(v))-1 for v in group);arm=f'{name}_c{cap}';candidates[arm]=(g,m.rates(),group);p=predict(val,*candidates[arm]);v=score(p,val)['nll'];valrows.append(dict(arm=arm,groups=group,coords=coords,nll=v,objective=v+SPEC['penalty']*(coords-k)))
 selected=min((r for r in valrows if r['arm'] in ('independent_c1','passive_c1','active_c1','active_c2','active_c4')),key=lambda r:(r['objective'],r['coords'],r['arm']))['arm'];checks=[]
 for name,m in models.items():
  with torch.inference_mode():a=m(*(torch.from_numpy(x[:32,:16]) for x in val)).numpy()
  b=predict(tuple(x[:32,:16] for x in val),graphs[name],m.rates(),[[i] for i in range(k)]);d=float(np.max(np.abs(a-b)));assert d<3e-5;checks.append(dict(arm=name,max_delta=d))
 if family in ('pairs','independent'):
  d=float(np.max(np.abs(predict(val,parent,law,[list(range(k))])-predict(val,parent,law,clusters(parent,np.ones(k),2)))));assert d<1e-12;checks.append(dict(arm='exact_components',max_delta=d))
 rows=[]
 for l,policy in itertools.product(SPEC['test_lengths'],SPEC['test_policies']):
  test=sample(k,parent,law,SPEC['test_words'],l,seed+30000+l+(0 if policy=='random' else 1000),policy);oracle=predict(test,parent,law,[list(range(k))]);preds={'oracle':oracle}
  for a,cfg in candidates.items():preds[a]=predict(test,*cfg)
  preds['selected']=preds[selected]
  with torch.inference_mode():preds['gru64']=gru(*(torch.from_numpy(x) for x in test)).numpy()
  for a,p in preds.items():rows.append(dict(arm=a,length=l,policy=policy,**score(p,test,oracle)))
 result=dict(k=k,family=family,seed=seed,true_graph=parent.tolist(),rates={n:v.tolist() for n,v in law.items()},graphs={n:g.tolist() for n,g in graphs.items()},discovery=disc,passive=pm,fit=fits,unique_filter_fits=len(cache),unique_filter_train_s=sum(v[1]['train_s'] for v in cache.values()),validation=valrows,selected=selected,checks=checks,rows=rows)
 (out/'receipt.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print(k,family,seed,'active',int(np.sum(active==parent)),'passive',int(np.sum(pas==parent)),selected,[(r['arm'],round(r['oracle_kl'],4)) for r in rows if r['length']==128 and r['policy']=='random' and r['arm'] in ('active_c1','active_c2','active_c4','selected','gru64')],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True);p.add_argument('--family',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.k,a.family,a.seed,a.out)
