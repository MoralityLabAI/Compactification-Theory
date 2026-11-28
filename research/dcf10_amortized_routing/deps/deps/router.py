"""DCF-08: correlation-priced per-trajectory partition routing."""
from gym import *
from followup import union_partition
from functools import lru_cache
S=json.loads(Path(__file__).with_name('spec.json').read_text())
@lru_cache(None)
def bm(g):return bitmat(g)
@lru_cache(None)
def intersections(old,new):
    out=[]
    for g in new:
        fac=[]
        for i,h in enumerate(old):
            common=sorted(set(g)&set(h))
            if common:
                ix=sum(bm(h)[h.index(z)]<<a for a,z in enumerate(common));pr=sum(bm(g)[g.index(z)]<<a for a,z in enumerate(common));fac.append((i,np.eye(1<<len(common))[ix],pr))
        out.append(fac)
    return out
def entropy(p):return -(p*np.log(np.maximum(p,1e-300))).sum(1)
def project(ps,old,new,make=True):
    cost=-sum(entropy(p) for p in ps);qs=[]
    for g,fac in zip(new,intersections(old,new)):
        q=np.ones((len(ps[0]),1<<len(g))) if make else None
        for i,mat,pr in fac:
            m=ps[i]@mat;m/=m.sum(1,keepdims=True);cost+=entropy(m)
            if make:q*=m[:,pr]
        if make:qs.append(q/q.sum(1,keepdims=True))
    return qs,np.maximum(cost,0.)
class Kernel:
    def __init__(self,g,p,rates):
        self.rates=rates;self.bits=[bm(h) for h in p];self.owner={a:(i,b) for i,h in enumerate(p) for b,a in enumerate(h)};self.table=[]
        for a,parents in enumerate(g):
            i,b=self.owner[a];s=np.arange(1<<len(p[i]));flip=s^(1<<b);inside=[h for h in parents if self.owner[h][0]==i];outside=[h for h in parents if self.owner[h][0]!=i];ctrl=np.ones(len(s),int) if not parents else np.bitwise_xor.reduce(self.bits[i][[self.owner[h][1] for h in inside]],axis=0) if inside else np.zeros(len(s),int);others=[]
            for z in sorted({self.owner[h][0] for h in outside}):
                pos=[self.owner[h][1] for h in outside if self.owner[h][0]==z];others.append((z,1-2*np.bitwise_xor.reduce(self.bits[z][pos],axis=0)))
            self.table.append((i,flip,s^(ctrl<<b),others))
    def step(self,ps,j,c,r,y):
        out=np.empty(len(j));rates=self.rates
        for a,(i,flip,cmd,others) in enumerate(self.table):
            ix=np.flatnonzero(j==a)
            if not len(ix):continue
            p=ps[i][ix];b=c[ix,None];noise=np.where(b,rates['fail'][a],rates['idle'][a]);q=p[:,cmd]
            if others:
                signed=np.ones(len(ix))
                for z,par in others:signed*=ps[z][ix]@par
                e=(1-signed[:,None])/2;q=(1-e)*q+e*q[:,flip]
            q=np.where(b,q,p);ps[i][ix]=(1-noise)*q+noise*q[:,flip]
        for a,(i,b) in self.owner.items():
            ix=np.flatnonzero(r==a)
            if not len(ix):continue
            p=ps[i][ix];e=rates['err'][a];lik=e+(1-2*e)*self.bits[i][b];out[ix]=p@lik;q=p*np.where(y[ix,None],lik,1-lik);ps[i][ix]=q/q.sum(1,keepdims=True)
        return out
def decode(labels,raw=False):
    n,l=labels.shape;out=np.empty_like(labels);q=np.full((n,4),.25)
    for t in range(l):
        z=labels[:,t]
        if raw:out[:,t]=np.where(z<0,out[:,t-1] if t else 0,z);continue
        if t:q=.875*q+.125/3*(1-q)
        q*=np.where(z[:,None]<0,1.,np.where(z[:,None]==np.arange(4),.85,.05));q/=q.sum(1,keepdims=True);out[:,t]=q.argmax(1)
    return out
def schedule(n,l,seed,scenario):
    g=np.random.default_rng(seed);ctx=np.empty((n,l),int);ctx[:,0]=g.integers(4,size=n)
    for t in range(1,l):
        h=.75 if scenario=='rapid' else .04 if scenario=='slow' else (.65 if (t//16)%2 else .04) if scenario=='bursty' else .125
        ctx[:,t]=(ctx[:,t-1]+(g.random(n)<h)*g.integers(1,4,size=n))%4
    lab=ctx.copy();e=.15 if scenario=='noisy15' else .30 if scenario=='noisy30' else 0
    if e:lab=(lab+(g.random(lab.shape)<e)*g.integers(1,4,size=lab.shape))%4
    if scenario=='missing50':lab[g.random(lab.shape)<.5]=-1
    return ctx,lab
def run_filter(data,context,graphs,rates,parts,nll,arm,instant,fixed,record=False):
    j,c,r,_,y=data;n,l=j.shape;size=np.array([coords(p) for p in parts]);lam=S['lambda'];fee=S['switch_fee'];mu=S['mu'] if arm=='correlation' else 0.
    kernels={(z,i):Kernel(g,p,rates) for z,g in enumerate(graphs) for i,p in enumerate(parts)};states={};pred=np.empty((n,l));routes=np.empty((n,l),int);loss=np.zeros(n);sw=np.zeros(n,int);count=np.full(n,2.);last=None;evals=0;start=time.perf_counter()
    for t in range(l):
        z=context[:,t];count+=0 if last is None else z!=last;last=z.copy();H=np.clip((16+t)/count,1,32);target=np.empty(n,int)
        if t==0:
            target[:]=fixed if arm=='fixed' else instant[z]
            for p in np.unique(target):
                ix=np.flatnonzero(target==p);states[p]=(ix,[np.full((len(ix),1<<len(g)),1/(1<<len(g))) for g in parts[p]])
        else:
            new={}
            for old,(ix,ps) in states.items():
                if arm=='fixed':dest=np.full(len(ix),fixed)
                elif arm=='instant':dest=instant[z[ix]]
                else:
                    zz=z[ix];candidate=np.stack([np.full(len(ix),old),instant[zz],np.full(len(ix),fixed)],1);utility=nll[zz[:,None],candidate]+lam*size[candidate];pen=np.zeros_like(utility)
                    for p in np.unique(candidate):
                        if p==old:continue
                        if mu:_,d=project(ps,parts[old],parts[p],False);evals+=len(ix)
                        else:d=np.zeros(len(ix))
                        pen+=np.where(candidate==p,fee+mu*d[:,None],0.)
                    dest=candidate[np.arange(len(ix)),(utility+pen/H[ix,None]).argmin(1)]
                target[ix]=dest
                for p in np.unique(dest):
                    loc=np.flatnonzero(dest==p);ids=ix[loc];q=[a[loc] for a in ps]
                    if p!=old:q,d=project(q,parts[old],parts[p]);loss[ids]+=d;sw[ids]+=1
                    new.setdefault(p,[]).append((ids,q))
            states={p:(np.concatenate([v[0] for v in vs]),[np.concatenate([v[1][a] for v in vs]) for a in range(len(parts[p]))]) for p,vs in new.items()}
        for p,(ix,ps) in states.items():
            for zz in np.unique(z[ix]):
                loc=np.flatnonzero(z[ix]==zz);ids=ix[loc];q=[v[loc] for v in ps];pred[ids,t]=kernels[zz,p].step(q,j[ids,t],c[ids,t],r[ids,t],y[ids,t])
                for v,w in zip(ps,q):v[loc]=w
        routes[:,t]=target
    elapsed=time.perf_counter()-start;assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
    if record:return pred,routes
    return pred,dict(mean_coords=float(size[routes].mean()),max_coords=int(size[routes].max()),switches_per_word=float(sw.mean()),projection_kl_per_word=float(loss.mean()),candidate_projection_checks=evals,wall_s=elapsed,word_coords=size[routes].mean(1).tolist(),word_switches=sw.tolist(),word_projection=loss.tolist())
def validation(k,graphs,true,g,rate,seed):
    parts=list(dict.fromkeys(tuple(partition(h,cap)) for h in g for cap in (1,2,3,4)));fixed_candidates=[tuple(union_partition(g,c)) for c in (1,2,3,4)]
    for p in fixed_candidates:
        if p not in parts:parts.append(p)
    nll=np.empty((4,len(parts)));size=np.array([coords(p) for p in parts])
    for z in range(4):
        val=sample(k,graphs,true,S['val_words'],S['val_length'],seed+300000+z,z)
        for i,p in enumerate(parts):nll[z,i]=score(predict(val,g,rate,[list(p)]*4),val)['nll']
    obj=nll+S['lambda']*size;instant=np.argmin(obj,axis=1);fixed=min([parts.index(p) for p in fixed_candidates],key=lambda a:obj[:,a].mean());return parts,nll,instant,fixed
def run(k,seed,out):
    out.mkdir(parents=True,exist_ok=True);graphs,true=system(k,seed);err,idle,cal=calibrate(k,graphs,true,seed+2000000);guess,rate,raw,probes=discover(k,graphs,true,seed+1000000,err);rate['idle']=idle;del cal,probes;g=guess['full'];parts,nll,instant,fixed=validation(k,graphs,true,g,rate,seed);full=[(tuple(range(k)),)];zero=np.zeros((4,1));iz=np.zeros(4,int);rows=[];checks=[]
    for si,scenario in enumerate(S['scenarios']):
        ctx,lab=schedule(S['test_words'],S['test_length'],seed+800000+si,scenario);data=sample(k,graphs,true,len(ctx),ctx.shape[1],seed+900000+si,ctx);noisy=si>=3;decoded=decode(lab) if noisy else lab;rawctx=decode(lab,True);oracle,_=run_filter(data,ctx,graphs,true,full,zero,'fixed',iz,0)
        arms=['full_learned','fixed','instant','dwell','correlation']+(['raw_correlation'] if noisy else [])
        for arm in arms:
            used=rawctx if arm=='raw_correlation' else decoded
            if arm=='full_learned':p,meta=run_filter(data,used,g,rate,full,zero,'fixed',iz,0)
            else:p,meta=run_filter(data,used,g,rate,parts,nll,'correlation' if arm=='raw_correlation' else arm,instant,fixed)
            sc=score(p,data,oracle);cost=sc['nll']+S['lambda']*meta['mean_coords']+S['switch_fee']*meta['switches_per_word']/ctx.shape[1];rows.append(dict(scenario=scenario,arm=arm,objective=cost,context_acc=float((used==ctx).mean()),oracle_nll=score(oracle,data)['nll'],**meta,**sc))
        if si==2:
            sub=tuple(a[:3,:20] for a in data);o,_=run_filter(sub,ctx[:3,:20],graphs,true,full,zero,'fixed',iz,0);e=float(np.max(np.abs(o-brute(sub,graphs,true))));assert e<1e-12;checks.append(dict(oracle=e));a,ar=run_filter(sub,ctx[:3,:20],g,rate,parts,nll,'correlation',instant,fixed,True);alt=(*sub[:-1],sub[-1].copy());alt[-1][:,8:]^=1;b,br=run_filter(alt,ctx[:3,:20],g,rate,parts,nll,'correlation',instant,fixed,True);assert np.array_equal(ar[:,:9],br[:,:9]);e=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert e==0;checks.append(dict(causal=e))
        print(k,seed,scenario,[(r['arm'],round(r['kl'],5),round(r['mean_coords'],2),round(r['switches_per_word'],1)) for r in rows if r['scenario']==scenario],flush=True)
    ret=dict(k=k,seed=seed,graphs=graphs,learned=g,rate={a:v.tolist() for a,v in rate.items()},graph_exact=sum(a==b for a,b in zip(graphs,g)),parts=parts,nll=nll.tolist(),instant=instant.tolist(),fixed=int(fixed),checks=checks,rows=rows);(out/'receipt.json').write_text(json.dumps(ret,separators=(',',':'))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.k,a.seed,a.out)
