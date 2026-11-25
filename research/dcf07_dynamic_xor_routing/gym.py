"""DCF-07: context-conditioned XOR graph discovery and changing cluster beliefs."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,itertools,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
SPEC=json.loads(Path(__file__).with_name('protocol.json').read_text())
def system(k,seed):
 g=np.random.default_rng(seed+1907*k);graphs=[[() for _ in range(k)]]
 o=g.permutation(k);a=[() for _ in range(k)]
 for j,p in zip(o[::2],o[1::2]):a[j]=(int(p),);a[p]=(int(j),)
 graphs.append(a);o=g.permutation(k);a=[() for _ in range(k)]
 for i in range(0,k-2,3):
  group=o[i:i+3]
  for j in group:a[j]=tuple(sorted(int(p) for p in group if p!=j))
 graphs.append(a);o=g.permutation(k);a=[() for _ in range(k)]
 for i,j in enumerate(o):a[j]=tuple(sorted((int(o[(i+1)%k]),int(o[(i+2)%k]))))
 graphs.append(a);return graphs,dict(err=g.uniform(.05,.18,k),idle=g.uniform(.015,.07,k),fail=g.uniform(.03,.12,k))
def simulate(j,c,r,ctx,graphs,rates,seed):
 g=np.random.default_rng(seed);n,l=j.shape;k=len(graphs[0]);s=g.integers(2,size=(n,k));y=np.zeros((n,l),int);idx=np.arange(n)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];ctrl=np.zeros(n,int)
  for z,graph in enumerate(graphs):
   for target,parents in enumerate(graph):
    m=(ctx[:,t]==z)&(a==target)
    if m.any():ctrl[m]=np.bitwise_xor.reduce(s[m][:,parents],axis=1) if parents else 1
  p=np.where(b,rates['fail'][a],rates['idle'][a]);s[idx,a]^=b*ctrl^(g.random(n)<p).astype(int);y[:,t]=s[idx,d]^(g.random(n)<rates['err'][d]).astype(int)
 return j,c,r,ctx,y
def sample(k,graphs,rates,n,l,seed,contexts):
 g=np.random.default_rng(seed);j=g.integers(k,size=(n,l));c=g.integers(2,size=(n,l));r=np.where(g.random((n,l))<.5,j,g.integers(k,size=(n,l)));ctx=np.broadcast_to(np.asarray(contexts) if np.ndim(contexts) else np.full(l,contexts),(n,l)).copy();return simulate(j,c,r,ctx,graphs,rates,seed+333333)
def probe(k,z,j,n,graphs,rates,seed):
 order=[j]+[i for i in range(k) if i!=j]+[j];a=np.tile(order,(n,1));c=np.zeros_like(a);c[:,-1]=1;ctx=np.full_like(a,z);data=simulate(a,c,a,ctx,graphs,rates,seed);y=data[-1];return data,1-2*(y[:,0]^y[:,-1]),1-2*y[:,1:-1],order[1:-1]
def moments(effect,signs,order,maxorder):
 result={():float(-effect.mean())}
 for size in range(1,maxorder+1):
  for cols in itertools.combinations(range(len(order)),size):result[tuple(order[i] for i in cols)]=float((effect*np.prod(signs[:,cols],axis=1)).mean())
 return result
def calibrate(k,graphs,rates,seed):
 n=SPEC['rate_trials'];err=[];idle=[];corpus=[]
 for j in range(k):
  other=(j+1)%k;r=np.full((n,2),j);ctx=np.zeros_like(r);c=np.zeros_like(r);a=np.full_like(r,other);data=simulate(a,c,r,ctx,graphs,rates,seed+100*j);corpus.append(data);y=data[-1];m=float((1-2*(y[:,0]^y[:,1])).mean());q=math.sqrt(max(.02,m));err.append(np.clip((1-q)/2,.005,.245));a=a.copy();a[:,1]=j;data=simulate(a,c,r,ctx,graphs,rates,seed+100*j+1);corpus.append(data);y=data[-1];m2=float((1-2*(y[:,0]^y[:,1])).mean());idle.append(np.clip((1-m2/(1-2*err[-1])**2)/2,.001,.24))
 return np.array(err),np.array(idle),corpus
def discover(k,graphs,rates,seed,err):
 screens=[];confirms=[];corpus=[];guesses={a:[] for a in ('full','first','pooled')};fail=[];bound=math.sqrt(2*math.log(3*k*4/SPEC['alpha'])/SPEC['confirm']);raw=[]
 for z in range(4):
  zs=[];zc=[]
  for j in range(k):
   data,a,b,order=probe(k,z,j,SPEC['screen'],graphs,rates,seed+100000*z+100*j);d,e,f,_=probe(k,z,j,SPEC['confirm'],graphs,rates,seed+100000*z+100*j+1);zs.append(moments(a,b,order,2));zc.append(moments(e,f,order,2));corpus.extend([data,d])
  screens.append(zs);confirms.append(zc)
 for z in range(4):
  gg={a:[] for a in guesses};fr=[];rr=[]
  for j in range(k):
   detail={}
   for arm in guesses:
    s=screens[z][j];c=confirms[z][j]
    if arm=='pooled':
     s={p:np.mean([screens[t][j][p] for t in range(3)]) for p in s};c={p:np.mean([confirms[t][j][p] for t in range(3)]) for p in c}
    p=max((p for p in s if arm!='first' or len(p)<=1),key=lambda p:s[p]);lb=float(c[p]-bound);ok=lb>SPEC['min_lower'];gg[arm].append(p if ok else ());detail[arm]=dict(proposal=list(p),screen=float(s[p]),confirm=float(c[p]),lower=lb,accepted=bool(ok))
   p=gg['full'][-1];m=confirms[z][j][p];den=(1-2*err[j])**2*np.prod([1-2*err[h] for h in p]);fr.append(float(np.clip((1-m/den)/2,.001,.24)) if detail['full']['accepted'] else None);rr.append(detail)
  for a in guesses:guesses[a].append(gg[a])
  fail.append(fr);raw.append(rr)
 f=np.array([[np.nan if x is None else x for x in y] for y in fail[:3]]);f=np.array([np.nanmedian(f[:,j]) if np.isfinite(f[:,j]).any() else .075 for j in range(k)])
 return guesses,dict(err=err,fail=f),raw,corpus

def partition(graph,cap):
 k=len(graph);groups=[{j} for j in range(k)]
 for j,ps in sorted(enumerate(graph),key=lambda x:(-len(x[1]),x[0])):
  if not ps:continue
  ids=[g for g in groups if g.intersection((j,*ps))];u=set.union(*ids)
  if len(ids)>1 and len(u)<=cap:
   for g in ids:groups.remove(g)
   groups.append(u)
  elif len(u)>cap:
   for p in ps:
    a=next(g for g in groups if j in g);b=next(g for g in groups if p in g)
    if a is not b and len(a|b)<=cap:groups.remove(a);groups.remove(b);groups.append(a|b)
 return sorted(tuple(sorted(g)) for g in groups)
def coords(groups):return sum(2**len(g)-1 for g in groups)
def bitmat(g):return (np.arange(1<<len(g))[None,:]>>np.arange(len(g))[:,None])&1
def repartition(ps,old,new):
 """Preserve within-intersection marginals; approximate across old factors."""
 out=[];n=len(ps[0]);
 for g in new:
  bits=bitmat(g);q=np.ones((n,1<<len(g)))
  for h,p in zip(old,ps):
   common=sorted(set(g)&set(h))
   if not common:continue
   index=sum(bitmat(h)[h.index(z)]<<i for i,z in enumerate(common));proj=sum(bits[g.index(z)]<<i for i,z in enumerate(common));marg=np.stack([p[:,index==i].sum(1) for i in range(1<<len(common))],1);q*=marg[:,proj]
  out.append(q)
 return out
def predict(data,graphs,rates,parts):
 j,c,r,ctx,y=data;n,l=j.shape;k=len(graphs[0]);out=np.empty((n,l));old=None;ps=None
 assert np.all(ctx==ctx[:1])
 for t in range(l):
  z=int(ctx[0,t]);groups=parts[z];graph=graphs[z]
  if groups!=old:
   ps=[np.full((n,1<<len(g)),1/(1<<len(g))) for g in groups] if old is None else repartition(ps,old,groups);old=groups
  bits=[bitmat(g) for g in groups];owner={a:(gi,i) for gi,g in enumerate(groups) for i,a in enumerate(g)}
  for a,parents in enumerate(graph):
   idx=np.flatnonzero(j[:,t]==a)
   if not len(idx):continue
   gi,i=owner[a];p=ps[gi][idx];s=np.arange(p.shape[1]);flip=s^(1<<i);b=c[idx,t,None];noise=np.where(b,rates['fail'][a],rates['idle'][a]);inside=[h for h in parents if owner[h][0]==gi];outside=[h for h in parents if owner[h][0]!=gi]
   ctrl=np.ones(len(s),int) if not parents else np.bitwise_xor.reduce(bits[gi][[owner[h][1] for h in inside]],axis=0) if inside else np.zeros(len(s),int)
   cmd=s^(ctrl<<i);q=p[:,cmd]
   if outside:
    signed=np.ones(len(idx))
    for gj in sorted({owner[h][0] for h in outside}):
     positions=[owner[h][1] for h in outside if owner[h][0]==gj];par=np.bitwise_xor.reduce(bits[gj][positions],axis=0);signed*=ps[gj][idx]@(1-2*par)
    ext=(1-signed[:,None])/2;q=(1-ext)*q+ext*q[:,flip]
   q=np.where(b,q,p);ps[gi][idx]=(1-noise)*q+noise*q[:,flip]
  for a in range(k):
   idx=np.flatnonzero(r[:,t]==a)
   if not len(idx):continue
   gi,i=owner[a];p=ps[gi][idx];e=rates['err'][a];lik=e+(1-2*e)*bits[gi][i];out[idx,t]=p@lik;q=p*np.where(y[idx,t,None],lik,1-lik);ps[gi][idx]=q/q.sum(1,keepdims=True)
 return out
def score(p,data,q=None):
 y=data[-1];p=np.clip(p,1e-7,1-1e-7);loss=-(y*np.log(p)+(1-y)*np.log1p(-p));ret=dict(nll=float(loss.mean()),brier=float(((p-y)**2).mean()),acc=float(((p>=.5)==y).mean()),n=len(y),tokens=y.size,word_nll=loss.mean(1).tolist())
 if q is not None:
  q=np.clip(q,1e-7,1-1e-7);kl=q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p));ret.update(kl=float(kl.mean()),word_kl=kl.mean(1).tolist())
 return ret
class GRU(nn.Module):
 def __init__(self,k):super().__init__();self.k=k;self.gru=nn.GRU(2*k+7,64,batch_first=True);self.head=nn.Linear(64,1)
 def forward(self,j,c,r,z,y):
  prev=F.one_hot(F.pad(y[:,:-1],(1,0),value=2),3)[...,:2];context=F.one_hot(z,4).float();x=torch.cat((F.one_hot(j,self.k),F.one_hot(r,self.k),c[...,None],context,prev),-1).float();h,_=self.gru(x);return self.head(h).squeeze(-1).sigmoid()
def gru_fit(k,graphs,rates,seed,probes,rate_corpus):
 torch.manual_seed(seed);m=GRU(k);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=np.random.default_rng(seed+991);passive=[sample(k,graphs,rates,SPEC['passive_n_per_seen'],SPEC['passive_length'],seed+400000+z,z) for z in range(3)];groups=[]
 for seq in (passive,probes[:3*k*2],rate_corpus):groups.append(tuple(np.concatenate([d[i] for d in seq],axis=0) for i in range(5)))
 start=time.perf_counter();m.train()
 for t in range(SPEC['gru_steps']):
  d=groups[t%3];ids=rng.integers(len(d[0]),size=SPEC['gru_batch']);x=[torch.from_numpy(a[ids]) for a in d];opt.zero_grad(set_to_none=True);p=m(*x);loss=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),x[-1].float());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 return m.eval(),dict(train_s=time.perf_counter()-start,params=sum(p.numel() for p in m.parameters()),loss=float(loss.detach()),group_words=[len(g[0]) for g in groups])
def brute(data,graphs,rates):
 j,c,r,ctx,y=data;n,l=j.shape;k=len(graphs[0]);out=np.zeros((n,l));N=1<<k
 for i in range(n):
  p=np.full(N,1/N)
  for t in range(l):
   a,b,d,z=map(int,(j[i,t],c[i,t],r[i,t],ctx[i,t]));f=rates['fail'][a] if b else rates['idle'][a];q=np.zeros(N);parents=graphs[z][a]
   for s,w in enumerate(p):
    ctrl=sum((s>>h)&1 for h in parents)%2 if parents else 1;h=s^(b*ctrl<<a);q[h]+=w*(1-f);q[h^(1<<a)]+=w*f
   obs=np.array([rates['err'][d] if not (s>>d)&1 else 1-rates['err'][d] for s in range(N)]);out[i,t]=q@obs;p=q*np.where(y[i,t],obs,1-obs);p/=p.sum()
 return out
def run(k,seed,out):
 out.mkdir(parents=True,exist_ok=True);graphs,true=system(k,seed);err,idle,cal=calibrate(k,graphs,true,seed+2000000);guess,estimated,raw,probes=discover(k,graphs,true,seed+1000000,err);estimated['idle']=idle;full=[[tuple(range(k))] for _ in graphs];single=[partition(g,1) for g in graphs];random=np.random.default_rng(seed+442);shuffled=[]
 for g in guess['full']:
  shuffled.append([tuple(sorted(random.choice([p for p in range(k) if p!=j],size=len(ps),replace=False).tolist())) if ps else () for j,ps in enumerate(g)])
 configurations={'oracle':(graphs,true,full),'known_est_full':(graphs,estimated,full),'learned_full':(guess['full'],estimated,full),'first_full':(guess['first'],estimated,full),'pooled_full':(guess['pooled'],estimated,full),'shuffled_full':(shuffled,estimated,full),'learned_c1':(guess['full'],estimated,single)}
 valrows=[];chosen=[];parts=[]
 for z in range(4):
  val=sample(k,graphs,true,SPEC['val_n'],SPEC['val_length'],seed+300000+z,z);rr=[]
  for cap in SPEC['caps']:
   pp=[partition(g,cap) for g in guess['full']];v=score(predict(val,guess['full'],estimated,pp),val)['nll'];d=coords(pp[z]);rr.append(dict(cap=cap,coords=d,groups=pp[z],nll=v,objective=v+SPEC['penalty']*(d-k)))
  pick=min(rr,key=lambda r:(r['objective'],r['coords'],r['cap']));chosen.append(pick['cap']);parts.append(partition(guess['full'][z],pick['cap']));valrows.append(rr)
 configurations['adaptive']=(guess['full'],estimated,parts)
 for cap in SPEC['caps'][1:]:configurations[f'learned_c{cap}']=(guess['full'],estimated,[partition(g,cap) for g in guess['full']])
 no=guess['full'][:3]+[guess['pooled'][3]];configurations['zero_shot']=(no,estimated,parts[:3]+[partition(no[3],chosen[0])]);m,meta=gru_fit(k,graphs,true,seed,probes,cal);torch.save(m.state_dict(),out/'gru.pt')
 rows=[];checks=[];l=SPEC['test_length'];schedules={f'context{z}':np.full(l,z) for z in range(4)};schedules['switch_seen']=(np.arange(l)//SPEC['switch_period'])%3;schedules['switch_novel']=(np.arange(l)//SPEC['switch_period'])%4
 for scenario,ctx in schedules.items():
  test=sample(k,graphs,true,SPEC['test_n'],l,seed+500000+list(schedules).index(scenario),ctx);oracle=predict(test,graphs,true,full)
  for name,(gg,rr,pp) in configurations.items():
   pred=oracle if name=='oracle' else predict(test,gg,rr,pp);active_coords=np.array([coords(pp[z]) for z in ctx]);rows.append(dict(scenario=scenario,arm=name,mean_coords=float(active_coords.mean()),max_coords=int(active_coords.max()),**score(pred,test,oracle)))
  with torch.inference_mode():p=m(*(torch.from_numpy(a) for a in test)).numpy()
  rows.append(dict(scenario=scenario,arm='gru64',mean_coords=64,max_coords=64,**score(p,test,oracle)))
  if scenario=='switch_novel':
   fixture=tuple(a[:3,:20] for a in test);delta=float(np.max(np.abs(brute(fixture,graphs,true)-predict(fixture,graphs,true,full))));assert delta<1e-12;checks.append(dict(oracle_max=delta))
   alt=(*test[:-1],test[-1].copy());alt[-1][:,8:]^=1
   for name in ('adaptive','learned_full'):
    cfg=configurations[name];a=predict(test,*cfg);b=predict(alt,*cfg);d=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert d<1e-12;checks.append(dict(arm=name,causal_max=d))
   with torch.inference_mode():b=m(*(torch.from_numpy(a) for a in alt)).numpy()
   d=float(np.max(np.abs(p[:,:9]-b[:,:9])));assert d<1e-6;checks.append(dict(arm='gru64',causal_max=d))
 for z in range(3):
  test=sample(k,graphs,true,32,32,seed+88888+z,z);d=float(np.max(np.abs(predict(test,graphs,true,full)-predict(test,graphs,true,[partition(g,3) for g in graphs]))));assert d<1e-12;checks.append(dict(context=z,component_max=d))
 ret=dict(k=k,seed=seed,graphs=graphs,guesses=guess,shuffled=shuffled,rates={a:v.tolist() for a,v in true.items()},estimated={a:v.tolist() for a,v in estimated.items()},discovery=raw,validation=valrows,chosen=chosen,parts=parts,gru=meta,checks=checks,rows=rows);(out/'receipt.json').write_text(json.dumps(ret,separators=(',',':'))+'\n');print(k,seed,'exact',[sum(a==b for a,b in zip(gg,graphs)) for gg in guess.values()], 'caps',chosen, 'switch',[(r['arm'],round(r['kl'],4)) for r in rows if r['scenario']=='switch_novel'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.k,a.seed,a.out)
