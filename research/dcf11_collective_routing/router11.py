"""DCF-11: collective layout pricing and fused CPU inference; shared padded belief buffers."""
from router10 import *
from numba import njit
import numba
P11=json.loads((HERE/'spec11.json').read_text())

@njit(cache=True)
def choose(cost,gamma):
    n,m=cost.shape;best=np.inf;answer=np.zeros(n,np.int64);tmp=np.empty(n,np.int64)
    for mask in range(1,1<<m):
        count=0
        for p in range(m):count+=(mask>>p)&1
        v=gamma*count
        for i in range(n):
            lo=np.inf;a=-1
            for p in range(m):
                if (mask>>p)&1 and cost[i,p]<lo:lo=cost[i,p];a=p
            v+=lo;tmp[i]=a
        if v<best:best=v;answer[:]=tmp
    return answer,best

def schedule(cost,gamma):
    if gamma==0:return cost.argmin(1)
    cols=np.flatnonzero(np.isfinite(cost).any(0));idx,v=choose(np.ascontiguousarray(cost[:,cols]),gamma);assert np.isfinite(v) and (idx>=0).all();return cols[idx]

def config(graphs,parts):
    P=len(parts);k=len(graphs[0]);M=max(1<<len(g) for p in parts for g in p);off=np.zeros((P,k),np.int64);length=off.copy();bit=off.copy();slots=np.zeros(P,np.int64)
    cmd=np.zeros((P,Q,k,M),np.int64);flip=cmd.copy();eo=np.full((P,Q,k,2),-1,np.int64);el=np.zeros_like(eo);sign=np.zeros((P,Q,k,2,M))
    for pi,p in enumerate(parts):
        offs=np.cumsum([0]+[1<<len(g) for g in p]);slots[pi]=offs[-1]
        for gi,g in enumerate(p):
            for b,a in enumerate(g):off[pi,a]=offs[gi];length[pi,a]=1<<len(g);bit[pi,a]=b
        for z,g in enumerate(graphs):
            kernel=Kernel(g,p,dict())
            for a,(gi,f,c,others) in enumerate(kernel.table):
                flip[pi,z,a,:len(f)]=f;cmd[pi,z,a,:len(c)]=c
                for i,(oi,sgn) in enumerate(others):eo[pi,z,a,i]=offs[oi];el[pi,z,a,i]=len(sgn);sign[pi,z,a,i,:len(sgn)]=sgn
    return off,length,bit,slots,cmd,flip,eo,el,sign

@njit(cache=True)
def mix_context(buf,q,prior,A,ids,slots):
    tmp=np.empty(4)
    for i in range(len(ids)):
        for s in range(slots[ids[i]]):
            for b in range(4):
                total=0.
                for a in range(4):total+=q[i,a]*A[a,b]*buf[i,a,s]
                tmp[b]=total/prior[i,b]
            for b in range(4):buf[i,b,s]=tmp[b]

@njit(cache=True)
def fused_step(buf,q,ids,j,c,r,y,idle,fail,err,off,length,bit,cmd,flip,eo,el,sign):
    n=len(ids);out=np.empty(n);tmp=np.empty(cmd.shape[-1]);p=np.empty(4)
    for i in range(n):
        lay=ids[i];a=j[i];b=c[i];d=r[i];o=off[lay,a];m=length[lay,a];ro=off[lay,d];rm=length[lay,d];rb=bit[lay,d];f=fail[a] if b else idle[a];e=err[d]
        for z in range(4):
            ext=1.
            if b:
                for v in range(2):
                    h=eo[lay,z,a,v]
                    if h>=0:
                        s=0.
                        for u in range(el[lay,z,a,v]):s+=buf[i,z,h+u]*sign[lay,z,a,v,u]
                        ext*=s
            ext=(1-ext)/2
            for u in range(m):
                if b:
                    v=cmd[lay,z,a,u];vf=cmd[lay,z,a,flip[lay,z,a,u]]
                    tmp[u]=(1-ext)*buf[i,z,o+v]+ext*buf[i,z,o+vf]
                else:tmp[u]=buf[i,z,o+u]
            for u in range(m):buf[i,z,o+u]=(1-f)*tmp[u]+f*tmp[flip[lay,z,a,u]]
            pred=0.;den=0.
            for u in range(rm):
                lk=e+(1-2*e)*((u>>rb)&1);pred+=buf[i,z,ro+u]*lk;den+=buf[i,z,ro+u]*(lk if y[i] else 1-lk)
            p[z]=pred
            for u in range(rm):
                lk=e+(1-2*e)*((u>>rb)&1);buf[i,z,ro+u]*=(lk if y[i] else 1-lk)/den
        v=0.;den=0.
        for z in range(4):v+=q[i,z]*p[z];den+=q[i,z]*(p[z] if y[i] else 1-p[z])
        out[i]=v
        for z in range(4):q[i,z]*=(p[z] if y[i] else 1-p[z])/den
    return out

def slices(p):
    offsets=np.cumsum([0]+[1<<len(g) for g in p]);return [slice(a,b) for a,b in zip(offsets[:-1],offsets[1:])]

def grouped_step(buf,q,ids,j,c,r,y,ss,ks):
    out=np.empty(len(ids));calls=0
    for p in np.unique(ids):
        ix=np.flatnonzero(ids==p);pred=np.empty((len(ix),Q))
        for z in range(Q):
            fac=[buf[ix,z,s].copy() for s in ss[p]];pred[:,z]=ks[p][z].step(fac,j[ix],c[ix],r[ix],y[ix]);calls+=1
            for s,f in zip(ss[p],fac):buf[ix,z,s]=f
        out[ix]=(q[ix]*pred).sum(1);q[ix]=normalize(q[ix]*np.where(y[ix,None],pred,1-pred))
    return out,calls

def route11(d,lab,e,g,r,parts,local,fixed,mode,backend,gamma=0.):
    j,c,read,_,y=d;n,l=j.shape;A=transition(S['assumed_hazard']);q=np.full((n,Q),.25);ids=np.full(n,fixed,np.int64)
    off,ln,bit,slots,cmd,flip,eo,el,sign=config(g,parts);ss=[slices(p) for p in parts];ks=[[Kernel(h,p,r) for h in g] for p in parts]
    buf=np.zeros((n,Q,int(slots.max())));allocated=buf[0].nbytes+q[0].nbytes+ids.itemsize
    for s in ss[fixed]:buf[:,:,s]=1/(s.stop-s.start)
    out=np.empty((n,l));paths=np.empty((n,l),np.int64);changes=0;checks=0;discard=0.;calls=0;obj=0.;layouts=0;used=0;sch_s=0.;project_s=0.;kernel_s=0.;start=time.perf_counter()
    for t in range(l):
        if t:
            prior=q@A;mix_context(buf,q,prior,A,ids,slots);q=prior
        q=normalize(q*label_lik(lab[:,t],e))
        if mode!='fixed' and t%P11['period']==0:
            u=time.perf_counter();expected=q@local;best=expected.argmin(1);cost=np.full_like(expected,np.inf)
            # Identical candidate sets; individual retain screening is invalid for a collective objective.
            for old in np.unique(ids):
                ix=np.flatnonzero(ids==old);cost[ix,old]=expected[ix,old];opts=np.stack((best[ix],np.full(len(ix),fixed)),1)
                for p in np.unique(opts):
                    if p==old:continue
                    ii=ix[np.any(opts==p,axis=1)];fac=[buf[ii,:,s].reshape(len(ii)*Q,-1) for s in ss[old]];_,drop=project(fac,parts[old],parts[p],False);drop=(drop.reshape(len(ii),Q)*q[ii]).sum(1);checks+=len(ii)*Q;cost[ii,p]=expected[ii,p]+(P11['fee']+P11['mu']*drop)/P11['horizon']
            dest=schedule(cost,gamma);obj+=float(cost[np.arange(n),dest].sum()+gamma*len(np.unique(dest)));sch_s+=time.perf_counter()-u;u=time.perf_counter()
            for old in np.unique(ids):
                for p in np.unique(dest[ids==old]):
                    if p==old:continue
                    ii=np.flatnonzero((ids==old)&(dest==p));fac=[buf[ii,:,s].reshape(len(ii)*Q,-1) for s in ss[old]];new,drop=project(fac,parts[old],parts[p]);discard+=float((drop.reshape(len(ii),Q)*q[ii]).sum());changes+=len(ii);buf[ii]=0
                    for s,v in zip(ss[p],new):buf[ii,:,s]=v.reshape(len(ii),Q,-1)
            ids=dest;project_s+=time.perf_counter()-u
        u=time.perf_counter()
        if backend=='fused':out[:,t]=fused_step(buf,q,ids,j[:,t],c[:,t],read[:,t],y[:,t],r['idle'],r['fail'],r['err'],off,ln,bit,cmd,flip,eo,el,sign);calls+=1
        else:out[:,t],v=grouped_step(buf,q,ids,j[:,t],c[:,t],read[:,t],y[:,t],ss,ks);calls+=v
        kernel_s+=time.perf_counter()-u;paths[:,t]=ids;layouts+=len(np.unique(ids));used+=int(slots[ids].sum())*Q*8+n*40
    elapsed=time.perf_counter()-start
    assert np.isfinite(out).all() and out.min()>0 and out.max()<1 and np.max(np.abs(q.sum(1)-1))<1e-10
    return out,paths,dict(seconds=elapsed,allocated_bytes=allocated,active_bytes=used/(n*l),layouts=layouts/l,kernel_calls=calls,switches_per_word=changes/n,projection_context_evals=checks,discarded_kl_per_word=discard/n,scheduler_s=sch_s,materialize_s=project_s,kernel_s=kernel_s,objective=obj)

def cal11(k,seed,g,r):
    parts=pool(g,max(P11['budgets'])-8);off=P11['seed_offset'];local=np.empty((Q,len(parts)));joint=[]
    clean=[sample(k,g,r,P11['val_words'],P11['val_length'],seed+off+z,z) for z in range(Q)]
    val=[fixture(k,g,r,sc,P11['val_words'],P11['val_length'],seed+off+100+i) for i,sc in enumerate(P11['scenarios'])]
    for i,p in enumerate(parts):
        for z,d in enumerate(clean):local[z,i]=loss(filter_belief(d,d[3],0.,g,r,p,'mixture')[0],d[-1]).mean()
        joint.append(float(np.mean([loss(filter_belief(d,lab,e,g,r,p,'mixture')[0],d[-1]).mean() for d,lab,e in val])))
    return parts,local,joint

ARMS={'fixed_group':('fixed','group',0.),'fixed_fused':('fixed','fused',0.),'ind_group':('ind','group',0.),'ind_fused':('ind','fused',0.),'cohort04_group':('cohort','group',.04),'cohort04_fused':('cohort','fused',.04),'cohort16_fused':('cohort','fused',.16)}

def run11(k,seed):
    g,r=system(k,seed);parts,local,joint=cal11(k,seed,g,r);rows=[];audits=[];selection=[]
    for cap in P11['budgets']:
        ii=[i for i,p in enumerate(parts) if size(p,'mixture')+8<=cap];ps=[parts[i] for i in ii];lo=local[:,ii];fixed=min(range(len(ii)),key=lambda i:(joint[ii[i]],size(ps[i],'mixture')));selection.append(dict(cap=cap,parts=ps,local=lo.tolist(),fixed=fixed))
        for si,sc in enumerate(P11['scenarios']):
            n=P11['batch'];d,lab,e=fixture(k,g,r,sc,n,P11['length'],seed+P11['seed_offset']+1000+si);ref=filter_belief(d,lab,e,g,r,(tuple(range(k)),),'mixture')[0];rng=np.random.default_rng(seed+cap+si+n);preds={};paths={};rr={a:[] for a in ARMS}
            for a,cfg in ARMS.items():route11(d,lab,e,g,r,ps,lo,fixed,*cfg)
            for rep in range(P11['reps']):
                for a in rng.permutation(list(ARMS)):
                    p,pa,m=route11(d,lab,e,g,r,ps,lo,fixed,*ARMS[a]);assert m['allocated_bytes']<=cap
                    if a in preds:assert np.array_equal(p,preds[a]) and np.array_equal(pa,paths[a])
                    preds[a]=p;paths[a]=pa;rr[a].append(m)
            for a,b in [('fixed_group','fixed_fused'),('ind_group','ind_fused'),('cohort04_group','cohort04_fused')]:
                delta=float(np.max(np.abs(preds[a]-preds[b])));assert delta<1e-10 and np.array_equal(paths[a],paths[b]),(k,seed,cap,sc,a,delta);audits.append(dict(cap=cap,scenario=sc,arm=a,delta=delta))
            for a in ARMS:
                p=preds[a];m=rr[a][0];meta={**m,'seconds':float(np.median([v['seconds'] for v in rr[a]])),'timing_samples':[v['seconds'] for v in rr[a]]};rows.append(dict(cap=cap,scenario=sc,batch=n,arm=a,kl=float(kl(ref,p).mean()),nll=float(loss(p,d[-1]).mean()),**meta))
            print(k,seed,cap,sc,[(x['arm'],round(x['kl'],5),round(x['seconds'],4),round(x['layouts'],2)) for x in rows[-len(ARMS):]],flush=True)
    return dict(k=k,seed=seed,rows=rows,audits=audits,selection=selection)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'results');ap.add_argument('--k',type=int);ap.add_argument('--seed',type=int);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for k,seed in itertools.product([a.k] if a.k else P11['widths'],[a.seed] if a.seed else P11['seeds']):
        result=run11(k,seed);(a.out/f'{k}_{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
