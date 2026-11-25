"""DCF-12: compiled candidate generation and exact factor projection."""
from router11 import *
P12=json.loads((HERE/'spec12.json').read_text())

def projection_tables(parts):
    m=len(parts);k=sum(map(len,parts[0]));M=max(1<<len(g) for p in parts for g in p)
    fs=np.zeros((m,k,2),np.int64);nf=np.array([len(p) for p in parts],np.int64)
    rec=np.zeros((m,m,k,5),np.int64);nr=np.zeros((m,m),np.int64)
    om=np.zeros((m,m,k,M),np.int64);nm=om.copy()
    for a,p in enumerate(parts):
        for i,s in enumerate(slices(p)):fs[a,i]=s.start,s.stop-s.start
        for b,q in enumerate(parts):
            for i,h in enumerate(p):
                for j,g in enumerate(q):
                    common=sorted(set(h)&set(g))
                    if not common:continue
                    t=nr[a,b];nr[a,b]+=1
                    os,oz=fs[a,i];ns=slices(q)[j];nz=ns.stop-ns.start
                    rec[a,b,t]=os,oz,ns.start,nz,1<<len(common)
                    for u in range(oz):om[a,b,t,u]=sum(((u>>h.index(z))&1)<<v for v,z in enumerate(common))
                    for u in range(nz):nm[a,b,t,u]=sum(((u>>g.index(z))&1)<<v for v,z in enumerate(common))
    return fs,nf,rec,nr,om,nm

@njit(cache=True)
def ent(p):
    v=0.
    for x in p:v-=x*np.log(max(x,1e-300))
    return v

@njit(cache=True)
def proj_row(buf,q,a,b,fs,nf,rec,nr,om,nm,make):
    out=np.ones_like(buf) if make else np.empty((0,0))
    marginal=np.empty(om.shape[-1]);drop=0.
    for z in range(4):
        h=0.
        for i in range(nf[a]):
            s,n=fs[a,i];h-=ent(buf[z,s:s+n])
        for t in range(nr[a,b]):
            s,n,v,w,d=rec[a,b,t]
            marginal[:d]=0.
            for u in range(n):marginal[om[a,b,t,u]]+=buf[z,s+u]
            total=0.
            for u in range(d):total+=marginal[u]
            for u in range(d):marginal[u]/=total
            h+=ent(marginal[:d])
            if make:
                for u in range(w):out[z,v+u]*=marginal[nm[a,b,t,u]]
        if make:
            for i in range(nf[b]):
                s,n=fs[b,i];total=0.
                for u in range(n):total+=out[z,s+u]
                for u in range(n):out[z,s+u]/=total
        drop+=q[z]*max(h,0.)
    return out,drop

@njit(cache=True)
def costs(buf,q,ids,expected,fixed,fs,nf,rec,nr,om,nm,fee,mu,horizon):
    cost=np.full_like(expected,np.inf);checks=0
    for i in range(len(ids)):
        a=ids[i];best=np.argmin(expected[i]);cost[i,a]=expected[i,a]
        for b in (best,fixed):
            if b==a or np.isfinite(cost[i,b]):continue
            _,d=proj_row(buf[i],q[i],a,b,fs,nf,rec,nr,om,nm,False)
            cost[i,b]=expected[i,b]+(fee+mu*d)/horizon;checks+=4
    return cost,checks

@njit(cache=True)
def apply_routes(buf,q,ids,dest,fs,nf,rec,nr,om,nm):
    switches=0;lost=0.
    for i in range(len(ids)):
        a,b=ids[i],dest[i]
        if a==b:continue
        p,d=proj_row(buf[i],q[i],a,b,fs,nf,rec,nr,om,nm,True)
        buf[i]=p;switches+=1;lost+=d
    return switches,lost

@njit(cache=True)
def compiled_schedule(cost,gamma):
    if gamma==0.:
        a=np.empty(len(cost),np.int64)
        for i in range(len(cost)):a[i]=np.argmin(cost[i])
        return a
    mask=np.zeros(cost.shape[1],np.bool_)
    for i in range(len(cost)):
        for j in range(cost.shape[1]):mask[j]|=np.isfinite(cost[i,j])
    cols=np.flatnonzero(mask);small=np.empty((len(cost),len(cols)))
    for j in range(len(cols)):small[:,j]=cost[:,cols[j]]
    a,v=choose(small,gamma)
    return cols[a]

@njit(cache=True)
def core(j,c,read,y,lab,e,A,local,fixed,gamma,dynamic,period,fee,mu,horizon,idle,fail,err,off,ln,bit,slots,cmd,flip,eo,el,sign,fs,nf,rec,nr,om,nm):
    n,l=j.shape;q=np.full((n,4),.25);ids=np.full(n,fixed,np.int64);buf=np.zeros((n,4,int(slots.max())))
    for i in range(nf[fixed]):
        s,w=fs[fixed,i];buf[:,:,s:s+w]=1/w
    out=np.empty((n,l));paths=np.empty((n,l),np.int64);checks=0;changes=0;lost=0.;used=0;layouts=0
    for t in range(l):
        if t:
            prior=q@A;mix_context(buf,q,prior,A,ids,slots);q=prior
        for i in range(n):
            if lab[i,t]>=0:
                for z in range(4):q[i,z]*=1-e if lab[i,t]==z else e/3
            q[i]/=q[i].sum()
        if dynamic and t%period==0:
            cost,v=costs(buf,q,ids,q@local,fixed,fs,nf,rec,nr,om,nm,fee,mu,horizon);checks+=v
            dest=compiled_schedule(cost,gamma);v,d=apply_routes(buf,q,ids,dest,fs,nf,rec,nr,om,nm);changes+=v;lost+=d;ids=dest
        out[:,t]=fused_step(buf,q,ids,j[:,t],c[:,t],read[:,t],y[:,t],idle,fail,err,off,ln,bit,cmd,flip,eo,el,sign)
        paths[:,t]=ids;layouts+=len(np.unique(ids))
        for i in range(n):used+=slots[ids[i]]*32+40
    return out,paths,checks,changes,lost,used,layouts

def fast(d,lab,e,g,r,parts,local,fixed,dynamic,gamma):
    j,c,read,_,y=d;n,l=j.shape;cfg=config(g,parts);tab=projection_tables(parts);A=transition(S['assumed_hazard']);local=np.ascontiguousarray(local)
    args=(j,c,read,y,lab,e,A,local,fixed,gamma,dynamic,P12['period'],P12['fee'],P12['mu'],P12['horizon'],r['idle'],r['fail'],r['err'],*cfg,*tab)
    t=time.perf_counter();p,paths,checks,changes,lost,used,layouts=core(*args);seconds=time.perf_counter()-t
    assert np.isfinite(p).all() and p.min()>0 and p.max()<1
    return p,paths,dict(seconds=seconds,allocated_bytes=int(cfg[3].max())*32+40,active_bytes=used/(n*l),layouts=layouts/l,projection_context_evals=checks,switches_per_word=changes/n,discarded_kl_per_word=lost/n)

def run(k,seed,out):
    g,r=system(k,seed);parts,local,joint=cal11(k,seed,g,r);rows=[];audits=[];selection=[]
    for b in P12['caps']:
        ii=[i for i,p in enumerate(parts) if size(p,'mixture')+8<=b];ps=[parts[i] for i in ii];lo=local[:,ii];fixed=min(range(len(ii)),key=lambda i:(joint[ii[i]],size(ps[i],'mixture')));selection.append(dict(cap=b,parts=ps,local=lo.tolist(),fixed=fixed))
    for si,sc in enumerate(P12['scenarios']):
        for n in P12['batches']:
            d,lab,e=fixture(k,g,r,sc,n,P12['length'],seed+P12['seed_offset']+si*1000+n);ref=filter_belief(d,lab,e,g,r,(tuple(range(k)),),'mixture')[0]
            for sel in selection:
                cap,ps,lo,fixed=sel['cap'],sel['parts'],np.array(sel['local']),sel['fixed'];gamma=P12['gamma_per_row']*n
                calls={f'{mode}_{backend}':(lambda mode=mode,backend=backend: fast(d,lab,e,g,r,ps,lo,fixed,mode!='fixed',gamma if mode=='cohort' else 0.) if backend=='fast' else route11(d,lab,e,g,r,ps,lo,fixed,'fixed' if mode=='fixed' else 'ind' if mode=='ind' else 'cohort','fused',gamma if mode=='cohort' else 0.)) for mode,backend in itertools.product(('fixed','ind','cohort'),('old','fast'))}
                for f in calls.values():f()
                pred={};paths={};records={a:[] for a in calls};rng=np.random.default_rng(seed+cap+si+n)
                for rep in range(P12['reps']):
                    for a in rng.permutation(list(calls)):
                        p,pa,m=calls[a]();assert m['allocated_bytes']<=cap
                        if a in pred:assert np.array_equal(pa,paths[a]) and np.array_equal(p,pred[a])
                        pred[a]=p;paths[a]=pa;records[a].append(m)
                for mode in ('fixed','ind','cohort'):
                    a,b=mode+'_old',mode+'_fast';delta=float(np.max(np.abs(pred[a]-pred[b])));same=bool(np.array_equal(paths[a],paths[b]));audits.append(dict(cap=cap,scenario=sc,batch=n,mode=mode,delta=delta,paths_equal=same));assert delta<1e-10 and same,(k,seed,cap,sc,n,mode,delta)
                for a in calls:
                    m=records[a][0];samples=[v['seconds'] for v in records[a]];p=pred[a];rows.append(dict(cap=cap,scenario=sc,batch=n,arm=a,gamma=gamma if a.startswith('cohort') else 0.,kl=float(kl(ref,p).mean()),nll=float(loss(p,d[-1]).mean()),**{**m,'seconds':float(np.median(samples)),'timing_samples':samples}))
                print(k,seed,sc,n,cap,[(a,round(rows[-6+i]['seconds']*1000,3)) for i,a in enumerate(calls)],flush=True)
                (out/f'{k}_{seed}.json').write_text(json.dumps(dict(k=k,seed=seed,rows=rows,audits=audits,selection=selection),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--k',type=int);p.add_argument('--seed',type=int);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for k,seed in itertools.product([a.k] if a.k else P12['widths'],[a.seed] if a.seed else P12['seeds']):run(k,seed,a.out)
