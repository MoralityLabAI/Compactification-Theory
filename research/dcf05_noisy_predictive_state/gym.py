"""Noisy predictive-state compactification; observation-only fit."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,time,json,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
SEEDS=(211,223,227)
CASES=((4,'independent'),(8,'independent'),(16,'independent'),(4,'coupled'),(8,'coupled'))
def law(k,seed):
 g=np.random.default_rng(seed+901*k);return dict(idle=g.uniform(.015,.07,k),fail=g.uniform(.025,.15,k),err=g.uniform(.07,.20,k))
def generate(k,kind,p,n,l,seed):
 g=np.random.default_rng(seed);j=g.integers(k,size=(n,l));c=g.integers(2,size=(n,l));r=np.where(g.random((n,l))<.5,j,g.integers(k,size=(n,l)));h=g.integers(2,size=(n,k));y=np.zeros((n,l),np.int64);rows=np.arange(n)
 for t in range(l):
  a=j[:,t];b=c[:,t];cmd=b if kind=='independent' else b*h[rows,(a+1)%k];noise=g.random(n)<np.where(b,p['fail'][a],p['idle'][a]);h[rows,a]^=cmd^noise.astype(int);y[:,t]=h[rows,r[:,t]]^(g.random(n)<p['err'][r[:,t]]).astype(int)
 return tuple(torch.from_numpy(x) for x in (j,c,r,y))
class Factored(nn.Module):
 def __init__(self,k):
  super().__init__();self.k=k;self.flip=nn.Parameter(torch.tensor([[-2.,2.]]*k));self.em=nn.Parameter(torch.full((k,),-1.6))
 def forward(self,j,c,r,y):
  p=torch.full((len(j),self.k),.5,device=j.device);out=[];fl=self.flip.sigmoid();err=.49*self.em.sigmoid()
  for a,b,d,z in zip(j.T,c.T,r.T,y.T):
   a,b,d,z=a[:,None],b[:,None],d[:,None],z[:,None].float();q=p.gather(1,a);f=fl[a,b];p=p.scatter(1,a,q+f*(1-2*q));q=p.gather(1,d);e=err[d];pred=e+(1-2*e)*q;out.append(pred[:,0]);l1=torch.where(z.bool(),1-e,e);l0=torch.where(z.bool(),e,1-e);p=p.scatter(1,d,q*l1/(q*l1+(1-q)*l0).clamp_min(1e-8))
  return torch.stack(out,1)
class GRU(nn.Module):
 def __init__(self,k,d):
  super().__init__();self.k=k;self.gru=nn.GRU(2*k+3,d,batch_first=True);self.out=nn.Linear(d,1)
 def forward(self,j,c,r,y):
  prev=F.one_hot(F.pad(y[:,:-1],(1,0),value=2),3).float();x=torch.cat([F.one_hot(j,self.k).float(),F.one_hot(r,self.k).float(),c[:,:,None].float(),prev[:,:,:2]],-1);h,_=self.gru(x);return self.out(h).squeeze(-1).sigmoid()
def fit(name,k,data,seed,steps,batch):
 torch.manual_seed(seed);m=Factored(k) if name=='factor' else GRU(k,int(name[3:]));opt=torch.optim.Adam(m.parameters(),lr=.035 if name=='factor' else .003);g=torch.Generator().manual_seed(seed+909);start=time.perf_counter()
 for i in range(steps):
  idx=torch.randint(len(data[0]),(batch,),generator=g);j,c,r,y=(x[idx] for x in data);opt.zero_grad(set_to_none=True);p=m(j,c,r,y);loss=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y.float());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 return m.eval(),time.perf_counter()-start,float(loss.detach())
def history_features(data,k):
 j,c,r,y=[x.numpy() for x in data];n,l=j.shape;last=np.full((n,k),2);fc=np.zeros((n,k),int);ic=fc.copy();par=fc.copy();rows=np.arange(n);keys=np.zeros((n,l),int)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];fc[rows,a]+=b;ic[rows,a]+=1-b;par[rows,a]^=b;keys[:,t]=(((last[rows,d]*4+np.minimum(fc[rows,d],3))*4+np.minimum(ic[rows,d],3))*2+par[rows,d]);last[rows,d]=y[:,t];fc[rows,d]=0;ic[rows,d]=0;par[rows,d]=0
 return keys
def empirical(data,k):
 keys=history_features(data,k).ravel();y=data[-1].numpy().ravel();n=np.bincount(keys,minlength=96);s=np.bincount(keys,weights=y,minlength=96);return (s+1)/(n+2)
def oracle_ind(data,k,law):
 j,c,r,y=[x.numpy() for x in data];n,l=j.shape;p=np.full((n,k),.5);out=np.zeros((n,l));rows=np.arange(n)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];f=np.where(b,1-law['fail'][a],law['idle'][a]);q=p[rows,a];p[rows,a]=q+f*(1-2*q);q=p[rows,d];e=law['err'][d];out[:,t]=e+(1-2*e)*q;l1=np.where(y[:,t],1-e,e);l0=np.where(y[:,t],e,1-e);p[rows,d]=q*l1/(q*l1+(1-q)*l0)
 return out
def oracle_joint(data,k,kind,law):
 j,c,r,y=[x.numpy() for x in data];n,l=j.shape;states=np.arange(1<<k);p=np.full((n,len(states)),1/len(states));out=np.zeros((n,l));bits=((states[None,:]>>np.arange(k)[:,None])&1);base=np.empty((k,2,len(states)),int);flip=np.empty((k,len(states)),int)
 for a in range(k):
  flip[a]=states^(1<<a);base[a,0]=states;base[a,1]=states^((1 if kind=='independent' else bits[(a+1)%k])<<a)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];q=np.take_along_axis(p,base[a,b],axis=1);f=np.where(b,law['fail'][a],law['idle'][a])[:,None];p=(1-f)*q+f*np.take_along_axis(q,flip[a],axis=1);e=law['err'][d,None];obs=e+(1-2*e)*bits[d];out[:,t]=(p*obs).sum(1);p*=np.where(y[:,t,None],obs,1-obs);p/=p.sum(1,keepdims=True)
 return out
def scores(p,y,oracle):
 p=np.clip(p.astype(float),1e-6,1-1e-6);loss=-(y*np.log(p)+(1-y)*np.log1p(-p));q=np.clip(oracle,1e-6,1-1e-6);kl=q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p));conf=np.maximum(p,1-p);hit=(p>=.5)==y;bins=np.minimum((conf*10).astype(int),9);ece=sum(abs(float(hit[bins==i].mean())-float(conf[bins==i].mean()))*float((bins==i).mean()) for i in range(10) if np.any(bins==i));mask=conf>=.8
 return dict(n=len(y),tokens=y.size,nll=float(loss.mean()),brier=float(((p-y)**2).mean()),acc=float(hit.mean()),oracle_kl=float(kl.mean()),ece=ece,coverage80=float(mask.mean()),accuracy80=float(hit[mask].mean()) if mask.any() else None,nll_by_word=loss.mean(1).tolist(),kl_by_word=kl.mean(1).tolist())
def main(args):
 root=args.out;root.mkdir(parents=True,exist_ok=True);rows=[];fits=[];checks=[]
 for k,kind in CASES:
  for seed in SEEDS:
   p=law(k,seed);train=generate(k,kind,p,2048,16,seed+10);models={};fit_meta={};empirical_p=empirical(train,k)
   for name in ('factor','gru16','gru64'):
    m,t,l=fit(name,k,train,seed,args.steps,args.batch);models[name]=m;fit_meta[name]=dict(params=sum(v.numel() for v in m.parameters()),train_s=t,loss=l);torch.save(m.state_dict(),root/f'{k}_{kind}_{seed}_{name}.pt')
   fp=models['factor'];fit_meta['factor']['flip_mae']=float(np.abs(fp.flip.detach().sigmoid().numpy()-np.stack([p['idle'],1-p['fail']],1)).mean()) if kind=='independent' else None;fit_meta['factor']['em_mae']=float(np.abs(.49*fp.em.detach().sigmoid().numpy()-p['err']).mean());fits.append(dict(k=k,kind=kind,seed=seed,law={a:v.tolist() for a,v in p.items()},models=fit_meta))
   for l in (16,64,128):
    test=generate(k,kind,p,256,l,seed+10000+l);y=test[-1].numpy();oracle=oracle_ind(test,k,p) if kind=='independent' else oracle_joint(test,k,kind,p)
    if kind=='independent' and k<=8 and l==16:
     e=float(np.max(np.abs(oracle-oracle_joint(test,k,kind,p))));assert e<1e-10;checks.append(dict(k=k,seed=seed,max_joint_factor_diff=e))
    preds={'oracle':oracle,'chance':np.full_like(oracle,.5),'empirical':empirical_p[history_features(test,k)]}
    with torch.inference_mode():preds.update({a:m(*test).numpy() for a,m in models.items()})
    for a,v in preds.items():rows.append(dict(k=k,kind=kind,seed=seed,length=l,arm=a,**scores(v,y,oracle)))
   print(k,kind,seed,[(a,round(v['train_s'],2)) for a,v in fit_meta.items()],[(r['arm'],round(r['nll'],4)) for r in rows if r['k']==k and r['kind']==kind and r['seed']==seed and r['length']==128],flush=True)
   (root/'receipt.json').write_text(json.dumps(dict(config=dict(seeds=SEEDS,cases=CASES,steps=args.steps,batch=args.batch,train_n=2048,train_l=16,test_n=256,torch=torch.__version__,numpy=np.__version__,python=platform.python_version()),fits=fits,checks=checks,rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=64);main(p.parse_args())
