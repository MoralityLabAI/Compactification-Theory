"""DCF-13: utility of query-directed factor inference. All decisions are synthetic."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,time,itertools,platform
from pathlib import Path
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'protocol.json').read_text())

@njit(cache=True)
def parity(x):
    p=0
    while x:
        p^=x&1;x>>=1
    return p

@njit(cache=True)
def value(x,mask,kind):
    if kind==0 or kind==3:return parity(x&mask)
    if kind==1:return int((x&mask)==mask)
    v=x&mask;n=0
    while v:n+=v&1;v>>=1
    return int(n>=2)

@njit(cache=True)
def likelihood(x,mask,kind,y,e,mix):
    p=float(value(x,mask,kind))
    if kind==3:p=(1-mix)*p+mix*value(x,mask,2)
    p=e+(1-2*e)*p
    return p if y else 1-p

@njit(cache=True)
def enumerate_factor(component,query,masks,kinds,ys,errs,mix):
    positions=np.empty(32,np.int64);nv=0
    for b in range(32):
        if (component>>b)&1:positions[nv]=b;nv+=1
    num=0.;den=0.;work=0
    for u in range(1<<nv):
        x=0
        for b in range(nv):x|=((u>>b)&1)<<positions[b]
        w=1.
        for j in range(len(masks)):
            if masks[j]&component==masks[j]:
                w*=likelihood(x,masks[j],kinds[j],ys[j],errs[j],mix);work+=1
        den+=w
        if parity(x&query):num+=w
    return num/den,work,1<<nv

@njit(cache=True)
def components(k,masks):
    comp=np.empty(k,np.int64)
    for a in range(k):comp[a]=1<<a
    for m in masks:
        u=0
        for a in range(k):
            if (m>>a)&1:u|=comp[a]
        for a in range(k):
            if (u>>a)&1:comp[a]=u
    return comp

@njit(cache=True)
def predict(k,queries,masks,kinds,ys,errs,mode,cap,mix):
    n=len(queries);probs=np.empty(n);work=np.zeros(n,np.int64);states=work.copy();skips=work.copy();maxdims=work.copy()
    for i in range(n):
        q=queries[i];mm=masks[i];kk=kinds[i];yy=ys[i];ee=errs[i]
        if mode==0:
            p,w,a=enumerate_factor((1<<k)-1,q,mm,kk,yy,ee,mix);probs[i]=p;work[i]=w;states[i]=a;maxdims[i]=k;continue
        if mode==1:
            comp=np.empty(k,np.int64)
            for b in range(k):comp[b]=15<<(4*(b//4))
        else:comp=components(k,mm)
        todo=q;too_large=False
        while todo:
            b=0
            while not ((todo>>b)&1):b+=1
            m=comp[b];todo&=~m;d=0
            for a in range(k):d+=(m>>a)&1
            maxdims[i]=max(maxdims[i],d)
            if d>cap:too_large=True
        if too_large:probs[i]=.5;skips[i]=1;continue
        todo=q;sign=1.
        while todo:
            b=0
            while not ((todo>>b)&1):b+=1
            m=comp[b];todo&=~m;p,w,a=enumerate_factor(m,q,mm,kk,yy,ee,mix);sign*=1-2*p;work[i]+=w;states[i]+=a
        probs[i]=(1-sign)/2
    return probs,work,states,skips,maxdims

def fixture(k,seed,n,scenario):
    rng=np.random.default_rng(seed);F=S['factors'];masks=np.empty((n,F),np.int64);kind=masks.copy();ys=masks.copy();errs=np.empty((n,F));queries=np.empty(n,np.int64);truth=np.empty(n,np.int64)
    for i in range(n):
        order=np.arange(k) if scenario=='stable' else rng.permutation(k);groups=[order[a:a+4] for a in range(0,k,4)]
        x=int(rng.integers(1<<k));g=groups[int(rng.integers(len(groups)))];qbits=rng.choice(g,size=2,replace=False)
        if rng.random()<.25:qbits=rng.choice(k,size=2,replace=False)
        q=sum(1<<int(b) for b in qbits);queries[i]=q;truth[i]=parity(x&q)
        for j in range(F):
            if j<k:bb=[j];t=0;e=.12
            else:
                g=groups[(j-k)%len(groups)];a=2 if rng.random()<.5 else 3;bb=rng.choice(g,size=a,replace=False);t=int(rng.choice([0,1,2] if a==3 else [0,1]));e=float(rng.uniform(.04,.16))
                if scenario=='bridge' and j>=F-2:bb=rng.choice(k,size=3,replace=False);t=int(rng.integers(3))
                if scenario=='ambiguous' and a==3:t=3
            m=sum(1<<int(b) for b in bb);actual=int(rng.choice([0,2])) if t==3 else t
            y=value(x,m,actual)^int(rng.random()<e);masks[i,j]=m;kind[i,j]=t;ys[i,j]=y;errs[i,j]=e
    return queries,masks,kind,ys,errs,truth

def decide(p,truth,ref):
    a=np.where(p>=.8,1,np.where(p<=.2,0,-1));u=np.where(a<0,0,np.where(a==truth,1,-4));exp=np.where(a<0,0,np.where(a==1,5*ref-4,1-5*ref));oracle=np.maximum(0,np.maximum(5*ref-4,1-5*ref));return dict(utility=float(u.mean()),expected_utility=float(exp.mean()),regret=float((oracle-exp).mean()),coverage=float((a>=0).mean()),wrong=int(((a>=0)&(a!=truth)).sum()),actions=a.tolist(),word_utility=u.tolist())

def run():
    rows=[];rng=np.random.default_rng(111)
    for k,seed,sc in itertools.product(S['widths'],S['seeds'],S['scenarios']):
        d=fixture(k,seed+100*k+10000*S['scenarios'].index(sc),S['n'],sc);q,mm,kk,yy,ee,truth=d;args=(k,q,mm,kk,yy,ee)
        arms={'full':(0,k,.5),'fixed4':(1,4,.5),'adaptive4':(2,4,.5),'adaptive8':(2,8,.5),'adaptive_full':(2,k,.5),'nominal8':(2,8,0.)}
        ref=predict(*args,*arms['full'])[0];times={a:[] for a in arms};results={}
        for a,c in arms.items():results[a]=predict(*args,*c)
        for _ in range(S['reps']):
            for a in rng.permutation(list(arms)):
                t=time.perf_counter();r=predict(*args,*arms[a]);times[a].append(time.perf_counter()-t);assert all(np.array_equal(x,y) for x,y in zip(r,results[a]))
        for a,(p,w,s,skip,dims) in results.items():
            err=float(np.max(np.abs(p[skip==0]-ref[skip==0]))) if (skip==0).any() else 0.
            if a.startswith('adaptive'):assert err<1e-10
            rows.append(dict(k=k,seed=seed,scenario=sc,arm=a,n=len(q),**decide(p,truth,ref),mean_factor_evals=float(w.mean()),mean_assignments=float(s.mean()),skip=float(skip.mean()),mean_component=float(dims.mean()),seconds=float(np.median(times[a])),timing_samples=times[a],max_certified_error=err))
        print(k,seed,sc,[(r['arm'],round(r['expected_utility'],3),round(r['seconds']*1000,2)) for r in rows[-6:]],flush=True)
        (ROOT/'receipt.json').write_text(json.dumps(dict(protocol=S,environment=dict(python=platform.python_version(),numpy=np.__version__),rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':run()
