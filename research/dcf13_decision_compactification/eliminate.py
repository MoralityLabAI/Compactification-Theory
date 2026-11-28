"""DCF-13B: exact min-scope elimination and robust likelihood intervals."""
from gym import *
@njit(cache=True)
def pop(x):
    n=0
    while x:n+=x&1;x>>=1
    return n
@njit(cache=True)
def expand(u,mask,k):
    x=0;b=0
    for i in range(k):
        if (mask>>i)&1:x|=((u>>b)&1)<<i;b+=1
    return x
@njit(cache=True)
def compress(x,mask,k):
    y=0;b=0
    for i in range(k):
        if (mask>>i)&1:y|=((x>>i)&1)<<b;b+=1
    return y
@njit(cache=True)
def eliminate(k,query,masks,kinds,ys,errs,cap,mix,bound):
    F=len(masks);scopes=np.zeros(F+k,np.int64);active=np.zeros(F+k,np.bool_);tables=[np.empty(0,np.float64) for _ in range(F+k)];work=0;live=0;peak=0;width=0
    for j in range(F):
        m=masks[j];scopes[j]=m;active[j]=True;n=1<<pop(m);f=np.empty(n)
        for u in range(n):
            x=expand(u,m,k);p=likelihood(x,m,kinds[j],ys[j],errs[j],mix)
            if kinds[j]==3 and bound:
                a=likelihood(x,m,3,ys[j],errs[j],0.);b=likelihood(x,m,3,ys[j],errs[j],1.);p=min(a,b) if bound<0 else max(a,b)
            f[u]=p
        tables[j]=f;live+=n;work+=n;width=max(width,pop(m))
    peak=live;remaining=((1<<k)-1)&~query;idx=F
    while remaining:
        best=-1;bestwidth=k+1;union=0
        for b in range(k):
            if not ((remaining>>b)&1):continue
            m=0
            for j in range(idx):
                if active[j] and ((scopes[j]>>b)&1):m|=scopes[j]
            d=pop(m)
            if d<bestwidth:bestwidth=d;best=b;union=m
        width=max(width,bestwidth)
        if bestwidth>cap:return .5,.5,work,peak,width,1
        remaining&=~(1<<best)
        if union==0:continue
        target=union&~(1<<best);f=np.zeros(1<<pop(target));which=np.empty(idx,np.int64);nw=0
        for j in range(idx):
            if active[j] and ((scopes[j]>>best)&1):which[nw]=j;nw+=1
        peak=max(peak,live+len(f))
        for u in range(1<<bestwidth):
            x=expand(u,union,k);p=1.
            for t in range(nw):
                j=which[t];p*=tables[j][compress(x,scopes[j],k)];work+=1
            f[compress(x,target,k)]+=p
        for t in range(nw):
            j=which[t];active[j]=False;live-=len(tables[j]);tables[j]=np.empty(0,np.float64)
        tables[idx]=f;active[idx]=True;scopes[idx]=target;idx+=1;live+=len(f)
    sums=np.zeros(2)
    for u in range(1<<pop(query)):
        x=expand(u,query,k);p=1.
        for j in range(idx):
            if active[j]:p*=tables[j][compress(x,scopes[j],k)];work+=1
        sums[parity(x&query)]+=p
    return sums[0],sums[1],work,peak,width,0
@njit(cache=True)
def ve(k,q,mm,kk,yy,ee,cap,mix,robust):
    n=len(q);p=np.empty(n);lo=np.empty(n);hi=np.empty(n);w=np.zeros(n,np.int64);mem=w.copy();width=w.copy();skip=w.copy()
    for i in range(n):
        if not robust:
            a,b,v,m,d,s=eliminate(k,q[i],mm[i],kk[i],yy[i],ee[i],cap,mix,0);p[i]=b/(a+b);lo[i]=p[i];hi[i]=p[i];w[i]=v;mem[i]=m;width[i]=d;skip[i]=s
        else:
            a,b,v,m,d,s=eliminate(k,q[i],mm[i],kk[i],yy[i],ee[i],cap,mix,-1);c,e,u,m2,d2,s2=eliminate(k,q[i],mm[i],kk[i],yy[i],ee[i],cap,mix,1);lo[i]=b/(b+c);hi[i]=e/(e+a);w[i]=v+u;mem[i]=max(m,m2);width[i]=max(d,d2);skip[i]=max(s,s2);p[i]=lo[i] if lo[i]>=.8 else hi[i] if hi[i]<=.2 else .5
    return p,lo,hi,w,mem,width,skip

def brute_numpy(k,q,masks,kinds,ys,errs,mix):
    x=np.arange(1<<k);bits=((x[:,None]>>np.arange(k))&1);w=np.ones(len(x))
    for m,t,y,e in zip(masks,kinds,ys,errs):
        c=bits[:,[a for a in range(k) if (m>>a)&1]].sum(1);a=c%2 if t in (0,3) else c==int(m).bit_count() if t==1 else c>=2
        a=a.astype(float)
        if t==3:a=(1-mix)*a+mix*(c>=2)
        a=e+(1-2*e)*a;w*=a if y else 1-a
    return float(w[(bits[:,[a for a in range(k) if (q>>a)&1]].sum(1)%2)==1].sum()/w.sum())

def run_followup():
    rng=np.random.default_rng(179);rows=[];checks=[]
    for k,seed,sc in itertools.product(S['widths'],S['seeds'],S['scenarios']):
        d=fixture(k,seed+300000+100*k+10000*S['scenarios'].index(sc),128,sc);q,mm,kk,yy,ee,truth=d;args=(k,q,mm,kk,yy,ee);ref=predict(*args,0,k,.5)[0]
        arms={'full':lambda:predict(*args,0,k,.5),'adaptive8':lambda:predict(*args,2,8,.5),'adaptive_full':lambda:predict(*args,2,k,.5),'ve4':lambda:ve(*args,4,.5,False),'ve6':lambda:ve(*args,6,.5,False),'ve_full':lambda:ve(*args,k,.5,False),'ve_nominal':lambda:ve(*args,k,0.,False),'ve_robust':lambda:ve(*args,k,.5,True)}
        ret={a:f() for a,f in arms.items()};ts={a:[] for a in arms}
        for rep in range(3):
            for a in rng.permutation(list(arms)):
                t=time.perf_counter();r=arms[a]();ts[a].append(time.perf_counter()-t);assert all(np.array_equal(x,y) for x,y in zip(r,ret[a]))
        for a,v in ret.items():
            if a.startswith('ve'):p,lo,hi,w,mem,width,skip=v
            else:p,w,assign,skip,width=v;mem=np.zeros_like(w)
            if a in ('ve4','ve6','ve_full'):
                e=float(np.max(np.abs(p[skip==0]-ref[skip==0]))) if (skip==0).any() else 0;assert e<1e-10
            if a=='ve_robust':assert (lo<=ref+1e-12).all() and (hi>=ref-1e-12).all();assert (np.array(decide(p,truth,ref)['actions'])<0).all() or decide(p,truth,ref)['regret']>=-1e-12
            rows.append(dict(k=k,seed=seed,scenario=sc,arm=a,n=len(q),**decide(p,truth,ref),seconds=float(np.median(ts[a])),timing_samples=ts[a],mean_work=float(w.mean()),max_table_entries=int(mem.max()),max_scope=int(width.max()),skip=float(skip.mean())))
        for i in range(4):
            p=brute_numpy(k,q[i],mm[i],kk[i],yy[i],ee[i],.5);e=abs(p-ref[i]);assert e<1e-10
            lo,hi=ret['ve_robust'][1:3];
            for m in (0.,.25,.5,.75,1.):
                p=brute_numpy(k,q[i],mm[i],kk[i],yy[i],ee[i],m);assert lo[i]-1e-12<=p<=hi[i]+1e-12
            checks.append(dict(k=k,seed=seed,scenario=sc,episode=i,full_delta=e,robust_lo=float(lo[i]),robust_hi=float(hi[i])))
        print(k,seed,sc,[(r['arm'],round(r['expected_utility'],3),round(r['seconds']*1000,3),r['max_scope']) for r in rows[-8:]],flush=True)
        (ROOT/'followup_receipt.json').write_text(json.dumps(dict(rows=rows,checks=checks),separators=(',',':'))+'\n')
if __name__=='__main__':run_followup()
