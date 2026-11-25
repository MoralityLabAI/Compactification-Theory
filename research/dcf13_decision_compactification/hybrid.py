"""DCF-13C: frozen hybrid small component enumeration / bounded elimination."""
from eliminate import *
@njit(cache=True)
def hybrid(k,q,mm,kk,yy,ee,threshold,cap):
    n=len(q);p=np.empty(n);fallback=np.zeros(n,np.int64);skips=fallback.copy();work=fallback.copy();scope=fallback.copy()
    for i in range(n):
        comp=components(k,mm[i]);todo=q[i];large=False;mx=0
        while todo:
            b=0
            while not ((todo>>b)&1):b+=1
            m=comp[b];todo&=~m;mx=max(mx,pop(m));large|=pop(m)>threshold
        if large:
            a,b,w,mem,d,s=eliminate(k,q[i],mm[i],kk[i],yy[i],ee[i],cap,.5,0);p[i]=b/(a+b);fallback[i]=1;skips[i]=s;work[i]=w;scope[i]=d
        else:
            todo=q[i];sign=1.;scope[i]=mx
            while todo:
                b=0
                while not ((todo>>b)&1):b+=1
                m=comp[b];todo&=~m;a,w,_=enumerate_factor(m,q[i],mm[i],kk[i],yy[i],ee[i],.5);sign*=1-2*a;work[i]+=w
            p[i]=(1-sign)/2
    return p,fallback,skips,work,scope

def main():
    cfg=json.loads((ROOT/'hybrid_protocol.json').read_text());rng=np.random.default_rng(199);rows=[]
    for k,seed,sc in itertools.product(S['widths'],cfg['seeds'],S['scenarios']):
        d=fixture(k,seed+600000+100*k+10000*S['scenarios'].index(sc),cfg['n'],sc);q,mm,kk,yy,ee,truth=d;args=(k,q,mm,kk,yy,ee)
        arms={'full':lambda:predict(*args,0,k,.5),'component_full':lambda:predict(*args,2,k,.5),'ve6':lambda:ve(*args,6,.5,False),'hybrid':lambda:hybrid(*args,cfg['threshold'],cfg['cap'])}
        ret={a:f() for a,f in arms.items()};times={a:[] for a in arms};ref=ret['full'][0]
        for rep in range(cfg['reps']):
            for a in rng.permutation(list(arms)):
                t=time.perf_counter();r=arms[a]();times[a].append(time.perf_counter()-t);assert all(np.array_equal(x,y) for x,y in zip(r,ret[a]))
        for a,v in ret.items():
            p=v[0];err=float(np.max(np.abs(p-ref)));assert err<1e-10
            rows.append(dict(k=k,seed=seed,scenario=sc,arm=a,n=len(q),**decide(p,truth,ref),seconds=float(np.median(times[a])),timing_samples=times[a],max_delta=err,fallback=float(v[1].mean()) if a=='hybrid' else None,skip=float(v[2].mean()) if a=='hybrid' else None,scope=int(v[4].max()) if a=='hybrid' else None))
        print(k,seed,sc,[(r['arm'],round(r['seconds']*1000,3),r['fallback']) for r in rows[-4:]],flush=True)
        (ROOT/'hybrid_receipt.json').write_text(json.dumps(dict(protocol=cfg,rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':main()
