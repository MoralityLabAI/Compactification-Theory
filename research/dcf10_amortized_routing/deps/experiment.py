"""DCF-09: known graph library, context uncertainty, budgeted belief representations."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,time,sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
DEPS=next(p for p in (ROOT/'deps',ROOT.parent/'dcf08_correlation_routing') if (p/'router.py').is_file())
sys.path.insert(0,str(DEPS))
from gym import system,sample
from followup import union_partition
from router import Kernel
S=json.loads((ROOT/'spec.json').read_text()); Q=4

def transition(h):
    a=np.full((Q,Q),h/(Q-1));np.fill_diagonal(a,1-h);return a

def regime(sc,n,l,seed):
    rng=np.random.default_rng(seed);z=np.zeros((n,l),int);z[:,0]=rng.integers(Q,size=n)
    for t in range(1,l):
        h=(.04 if (t//16)%2==0 else .65) if sc=='bursty30' else S['assumed_hazard']
        z[:,t]=(z[:,t-1]+(rng.random(n)<h)*rng.integers(1,Q,size=n))%Q
    e={'clean':0.,'noisy15':.15,'noisy30':.30,'missing50':0.,'hidden':0.,'bursty30':.30}[sc]
    lab=(z+(rng.random(z.shape)<e)*rng.integers(1,Q,size=z.shape))%Q
    if sc=='missing50':lab[rng.random(z.shape)<.5]=-1
    if sc=='hidden':lab[:]=-1
    return z,lab,e

def label_lik(lab,e):return np.where(lab[:,None]<0,1.,np.where(lab[:,None]==np.arange(Q),1-e,e/(Q-1)))
def normalize(p):
    d=p.sum(-1,keepdims=True);assert (d>0).all() and np.isfinite(d).all();return p/d

def filter_belief(data,labels,e,graphs,rates,part,mode,h=None):
    """Mixture maintains context-specific factors; shared collapses over contexts; MAP uses labels alone."""
    h=S['assumed_hazard'] if h is None else h
    j,c,r,_,y=data;n,l=j.shape;A=transition(h);ks=[Kernel(g,part,rates) for g in graphs];q=np.full((n,Q),.25)
    ps=[np.full((n,Q if mode=='mixture' else 1,1<<len(g)),1/(1<<len(g))) for g in part]
    out=np.empty((n,l));ctx=np.empty((n,l,Q));postctx=np.empty_like(ctx);start=time.perf_counter();work=0
    for t in range(l):
        if t:
            prior=q@A
            if mode=='mixture':
                mix=q[:,:,None]*A[None,:,:]/prior[:,None,:]
                ps=[np.einsum('nab,nai->nbi',mix,p) for p in ps]
            q=prior
        q=normalize(q*label_lik(labels[:,t],e));ctx[:,t]=q
        if mode=='map':
            z=q.argmax(1);pred=np.empty(n)
            for a in range(Q):
                ix=np.flatnonzero(z==a)
                if not len(ix):continue
                fac=[p[ix,0].copy() for p in ps];pred[ix]=ks[a].step(fac,j[ix,t],c[ix,t],r[ix,t],y[ix,t]);work+=len(ix)
                for p,f in zip(ps,fac):p[ix,0]=f
            out[:,t]=pred
        else:
            pred=np.empty((n,Q));post=[]
            for a in range(Q):
                fac=[p[:,a if mode=='mixture' else 0].copy() for p in ps];pred[:,a]=ks[a].step(fac,j[:,t],c[:,t],r[:,t],y[:,t]);post.append(fac);work+=n
            out[:,t]=(q*pred).sum(1);q=normalize(q*np.where(y[:,t,None],pred,1-pred))
            if mode=='mixture':ps=[np.stack([post[a][i] for a in range(Q)],1) for i in range(len(part))]
            else:ps=[np.einsum('na,nai->ni',q,np.stack([post[a][i] for a in range(Q)],1))[:,None] for i in range(len(part))]
        postctx[:,t]=q
    assert np.isfinite(out).all() and out.min()>0 and out.max()<1
    return out,ctx,postctx,dict(bytes=int(q[0].nbytes+sum(p[0].nbytes for p in ps)),kernel_rows=work,particle_steps=0,wall_s=time.perf_counter()-start)

def particle(data,labels,e,graphs,rates,N,seed):
    """Label-conditioned proposal, packed uint16 state/context, stratified resampling."""
    j,c,r,_,y=data;n,l=j.shape;k=len(graphs[0]);M=1<<k;A=transition(S['assumed_hazard']);rng=np.random.default_rng(seed)
    packed=(rng.integers(M,size=(n,N))+(rng.integers(Q,size=(n,N))<<k)).astype(np.uint16)
    out=np.empty((n,l));ctx=np.empty((n,l,Q));postctx=np.empty_like(ctx);s=np.arange(M);control=np.empty((Q,k,M),int)
    for z,g in enumerate(graphs):
        for a,pa in enumerate(g):control[z,a]=np.bitwise_xor.reduce((s[None,:]>>np.array(pa)[:,None])&1,axis=0) if pa else 1
    start=time.perf_counter()
    for t in range(l):
        ll=label_lik(labels[:,t],e);oldz=packed>>k;x=packed&(M-1)
        pr=(np.full((n,N,Q),.25) if t==0 else A[oldz])*ll[:,None,:];w=pr.sum(2);pr=pr/w[:,:,None]
        z=np.minimum((rng.random((n,N,1))>np.cumsum(pr,2)).sum(2),Q-1);w=normalize(w);ctx[:,t]=np.stack([(w*(z==a)).sum(1) for a in range(Q)],1)
        a=j[:,t,None];b=c[:,t,None];d=r[:,t,None];noise=np.where(b,rates['fail'][a],rates['idle'][a]);flip=b*control[z,a,x]^(rng.random((n,N))<noise);x=x^(flip<<a)
        err=rates['err'][d];p=err+(1-2*err)*((x>>d)&1);out[:,t]=(w*p).sum(1);w=normalize(w*np.where(y[:,t,None],p,1-p));postctx[:,t]=np.stack([(w*(z==a)).sum(1) for a in range(Q)],1)
        p=(x+(z<<k)).astype(np.uint16);cdf=np.cumsum(w,1);cdf[:,-1]=1.;u=(np.arange(N)[None,:]+rng.random((n,N)))/N;ix=np.stack([np.searchsorted(row,v,side='right') for row,v in zip(cdf,u)]);packed=np.take_along_axis(p,ix,1)
    assert np.isfinite(out).all() and out.min()>0 and out.max()<1
    return out,ctx,postctx,dict(bytes=packed[0].nbytes,kernel_rows=0,particle_steps=n*l*N,wall_s=time.perf_counter()-start)

def loss(p,y):
    p=np.clip(p,1e-10,1-1e-10);return -(y*np.log(p)+(1-y)*np.log1p(-p))
def kl(q,p):return loss(p,q)-loss(q,q)
def measure(result,data,ref,priv):
    pred,pre,post,meta=result;y=data[-1];z=data[3];nll=loss(pred,y);k=kl(ref,pred);kp=kl(priv,pred);p=np.take_along_axis(pre,z[:,:,None],2)[:,:,0]
    return dict(nll=float(nll.mean()),kl_same_info=float(k.mean()),kl_privileged=float(kp.mean()),accuracy=float(((pred>=.5)==y).mean()),context_acc=float((pre.argmax(2)==z).mean()),context_post_acc=float((post.argmax(2)==z).mean()),context_nll=float(-np.log(np.clip(p,1e-12,1)).mean()),word_nll=nll.mean(1).tolist(),word_kl=k.mean(1).tolist(),**meta)
def size(part,mode):return 8*(Q+(Q if mode=='mixture' else 1)*sum(1<<len(g) for g in part))
def candidates(graphs):return list(dict.fromkeys(tuple(union_partition(graphs,cap)) for cap in S['partition_caps']))
def fixture(k,g,r,sc,n,l,seed):
    z,lab,e=regime(sc,n,l,seed);return sample(k,g,r,n,l,seed+100000,z),lab,e

def val_select(k,g,r,seed):
    vals=[];parts=candidates(g);fix=[fixture(k,g,r,sc,S['val_words'],S['val_length'],seed+10000+i) for i,sc in enumerate(S['scenarios'][1:5])]
    for mode,part in itertools.product(('mixture','shared','map'),parts):
        b=size(part,mode)
        if b>max(S['budgets_bytes']):continue
        v=[float(loss(filter_belief(d,lab,e,g,r,part,mode)[0],d[-1]).mean()) for d,lab,e in fix];vals.append(dict(mode=mode,part=part,bytes=b,val_nll=float(np.mean(v)),val_cells=v))
    selected=[]
    for b,mode in itertools.product(S['budgets_bytes'],('mixture','shared','map')):
        best=min((x for x in vals if x['mode']==mode and x['bytes']<=b),key=lambda x:(x['val_nll'],x['bytes']));selected.append(dict(budget=b,**best))
    return vals,selected

def run(k,seed,out):
    g,r=system(k,seed);full=(tuple(range(k)),);vals,sel=val_select(k,g,r,seed);out.mkdir(parents=True,exist_ok=True);rows=[];checks=[]
    for i,sc in enumerate(S['scenarios']):
        d,lab,e=fixture(k,g,r,sc,S['test_words'],S['test_length'],seed+300000+i)
        ref=filter_belief(d,lab,e,g,r,full,'mixture');priv=filter_belief(d,d[3],0.,g,r,full,'mixture')[0];rows.append(dict(scenario=sc,arm='exact_same_info',budget=None,**measure(ref,d,ref[0],priv)))
        cache={}
        for x in sel:
            mode,part,b=x['mode'],x['part'],x['budget'];key=mode,part
            if key not in cache:cache[key]=filter_belief(d,lab,e,g,r,part,mode)
            result=cache[key];assert result[3]['bytes']==x['bytes']<=b;rows.append(dict(scenario=sc,arm=mode,budget=b,part=part,**measure(result,d,ref[0],priv)))
        for b,rep in itertools.product(S['budgets_bytes'],S['particle_seeds']):
            result=particle(d,lab,e,g,r,b//2,seed+500000+100*i+rep);assert result[3]['bytes']==b;rows.append(dict(scenario=sc,arm='particle',budget=b,rep=rep,**measure(result,d,ref[0],priv)))
        if sc=='noisy30':
            sub=tuple(a[:4,:20] for a in d);sublab=lab[:4,:20];alt=(*sub[:-1],sub[-1].copy());alt[-1][:,8:]^=1
            for mode,part in [('mixture',full)]+list(cache):
                a=filter_belief(sub,sublab,e,g,r,part,mode)[0];b=filter_belief(alt,sublab,e,g,r,part,mode)[0];delta=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert delta<1e-12;checks.append(dict(mode=mode,prefix_delta=delta))
            a=particle(sub,sublab,e,g,r,320,seed)[0];b=particle(alt,sublab,e,g,r,320,seed)[0];delta=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert delta==0;checks.append(dict(mode='particle',prefix_delta=delta))
        print(k,seed,sc,[(x['arm'],x['budget'],round(x['kl_same_info'],5)) for x in rows if x['scenario']==sc and x['budget'] in (640,None)],flush=True)
        (out/'receipt.json').write_text(json.dumps(dict(k=k,seed=seed,graphs=g,rates={a:v.tolist() for a,v in r.items()},validation=vals,selected=sel,checks=checks,rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.k,a.seed,a.out)
