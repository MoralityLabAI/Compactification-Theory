"""DCF-10: exact projection screening and periodic/event routing."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
from experiment import *
from adaptive import pool
from router import project
P=json.loads((HERE/'protocol.json').read_text())

def route(d,lab,e,g,r,parts,local,fixed,mode):
    j,c,read,_,y=d;n,l=j.shape;A=transition(S['assumed_hazard']);q=np.full((n,Q),.25)
    sizes=np.array([size(p,'mixture')+P['extra_route_bytes'] for p in parts]);ks=[[Kernel(h,p,r) for h in g] for p in parts]
    states={fixed:(np.arange(n),[np.full((n,Q,1<<len(a)),1/(1<<len(a))) for a in parts[fixed]])}
    out=np.empty((n,l));paths=np.empty((n,l),int);sw=0;discard=0.;checks=0;decisions=0;used=0;peak=0;kernel_rows=0;start=time.perf_counter()
    fee=P['fee']/P['horizon'];mu=P['mu']/P['horizon']
    for t in range(l):
        if t:
            prior=q@A;mix=q[:,:,None]*A[None,:,:]/prior[:,None,:]
            for p,(ix,ps) in states.items():states[p]=(ix,[np.einsum('nab,nai->nbi',mix[ix],x) for x in ps])
            q=prior
        q=normalize(q*label_lik(lab[:,t],e));destinations={};changed=False
        if mode!='fixed':
            for old,(ix,ps) in states.items():
                expected=q[ix]@local;best=expected.argmin(1)
                choices=np.stack([np.full(len(ix),old),best,np.full(len(ix),fixed)],1)
                cost=np.take_along_axis(expected,choices,1);retain=cost[:,0].copy()
                eligible=np.ones(len(ix),bool) if mode!='event' or t%P['period']==0 else (retain-cost.min(1)>P['override_gain'])
                decisions+=int(eligible.sum())
                for p in np.unique(choices):
                    if p==old:continue
                    mask=choices==p
                    if mode=='step':sel=np.arange(len(ix))
                    else:sel=np.flatnonzero(eligible & np.any(mask & (cost+fee<retain[:,None]),axis=1))
                    if mode!='step':cost=np.where(mask,np.inf,cost)
                    if not len(sel):continue
                    flat=[x[sel].reshape(len(sel)*Q,-1) for x in ps]
                    _,drop=project(flat,parts[old],parts[p],False);drop=(drop.reshape(len(sel),Q)*q[ix[sel]]).sum(1);checks+=len(sel)*Q
                    if mode=='step':cost+=np.where(mask,fee+mu*drop[:,None],0.)
                    else:cost[sel]=np.where(mask[sel],expected[sel,p,None]+fee+mu*drop[:,None],cost[sel])
                target=choices[np.arange(len(ix)),cost.argmin(1)];destinations[old]=target
                changed|=bool(np.any(target!=old))
        if changed:
            new={}
            for old,(ix,ps) in states.items():
                target=destinations[old]
                for p in np.unique(target):
                    ii=np.flatnonzero(target==p);ids=ix[ii];factors=[x[ii] for x in ps]
                    if p!=old:
                        z,drop=project([x.reshape(len(ii)*Q,-1) for x in factors],parts[old],parts[p]);factors=[x.reshape(len(ii),Q,-1) for x in z];sw+=len(ii);discard+=float((drop.reshape(len(ii),Q)*q[ids]).sum())
                    new.setdefault(p,[]).append((ids,factors))
            states={p:(np.concatenate([x[0] for x in rows]),[np.concatenate([x[1][i] for x in rows],axis=0) for i in range(len(parts[p]))]) for p,rows in new.items()}
        for p,(ix,ps) in states.items():
            pred=np.empty((len(ix),Q));post=[]
            for z in range(Q):
                f=[x[:,z].copy() for x in ps];pred[:,z]=ks[p][z].step(f,j[ix,t],c[ix,t],read[ix,t],y[ix,t]);post.append(f);kernel_rows+=len(ix)
            out[ix,t]=(q[ix]*pred).sum(1);q[ix]=normalize(q[ix]*np.where(y[ix,t,None],pred,1-pred));states[p]=(ix,[np.stack([post[a][i] for a in range(Q)],1) for i in range(len(ps))]);paths[ix,t]=p;used+=len(ix)*sizes[p];peak=max(peak,int(sizes[p]))
    elapsed=time.perf_counter()-start
    assert np.isfinite(out).all() and out.min()>0 and out.max()<1
    return out,paths,dict(seconds=elapsed,peak_bytes=peak,mean_bytes=used/(n*l),projection_context_evals=checks,decision_rows=decisions,switches_per_word=sw/n,discarded_kl_per_word=discard/n,kernel_rows=kernel_rows)

def calibrate(k,seed,g,r):
    parts=pool(g,max(P['budgets'])-P['extra_route_bytes']);off=P['seed_offset'];local=np.empty((Q,len(parts)));joint=[]
    clean=[sample(k,g,r,P['val_words'],P['val_length'],seed+off+z,z) for z in range(Q)]
    val=[fixture(k,g,r,sc,P['val_words'],P['val_length'],seed+off+100+i) for i,sc in enumerate(P['scenarios'])]
    for i,p in enumerate(parts):
        for z,d in enumerate(clean):local[z,i]=loss(filter_belief(d,d[3],0.,g,r,p,'mixture')[0],d[-1]).mean()
        joint.append(float(np.mean([loss(filter_belief(d,lab,e,g,r,p,'mixture')[0],d[-1]).mean() for d,lab,e in val])))
    return parts,local,joint

def run(k,seed):
    g,r=system(k,seed);parts,local,joint=calibrate(k,seed,g,r);rows=[];audits=[];selection=[]
    for b in P['budgets']:
        ii=[i for i,p in enumerate(parts) if size(p,'mixture')+P['extra_route_bytes']<=b];ps=[parts[i] for i in ii];lo=local[:,ii];fixed=min(range(len(ii)),key=lambda i:(joint[ii[i]],size(ps[i],'mixture')));selection.append(dict(budget=b,parts=ps,local=lo.tolist(),joint=[joint[i] for i in ii],fixed=fixed))
        for i,sc in enumerate(P['scenarios']):
            d,lab,e=fixture(k,g,r,sc,P['test_words'],P['test_length'],seed+P['seed_offset']+1000+i)
            ref=filter_belief(d,lab,e,g,r,(tuple(range(k)),),'mixture')[0]
            modes=('fixed','step','screen','event');runs={a:[] for a in modes};preds={};paths={};rng=np.random.default_rng(seed+100*b+i)
            for a in modes:route(d,lab,e,g,r,ps,lo,fixed,a)
            for rep in range(P['reps']):
                for a in rng.permutation(modes):
                    p,pa,m=route(d,lab,e,g,r,ps,lo,fixed,a);assert m['peak_bytes']<=b
                    if a in preds:assert np.array_equal(p,preds[a]) and np.array_equal(pa,paths[a])
                    preds[a]=p;paths[a]=pa;runs[a].append(m)
            err=float(np.max(np.abs(preds['step']-preds['screen'])));route_equal=bool(np.array_equal(paths['step'],paths['screen']));assert err<1e-12 and route_equal
            sub=tuple(x[:3,:24] for x in d);alt=(*sub[:-1],sub[-1].copy());alt[-1][:,8:]^=1
            u,v,_=route(sub,lab[:3,:24],e,g,r,ps,lo,fixed,'event');u2,v2,_=route(alt,lab[:3,:24],e,g,r,ps,lo,fixed,'event');causal=float(np.max(np.abs(u[:,:9]-u2[:,:9])));assert causal==0 and np.array_equal(v[:,:9],v2[:,:9]);audits.append(dict(budget=b,scenario=sc,screen_error=err,route_equal=route_equal,causal=causal))
            for a in modes:
                p=preds[a];m=runs[a][0];ww=kl(ref,p).mean(1);m={**m,'seconds':float(np.median([v['seconds'] for v in runs[a]])),'timing_samples':[v['seconds'] for v in runs[a]]}
                rows.append(dict(budget=b,scenario=sc,arm=a,kl=float(ww.mean()),nll=float(loss(p,d[-1]).mean()),word_kl=ww.tolist(),**m))
            print(k,seed,b,sc,[(x['arm'],round(x['kl'],6),round(x['seconds'],3),x['projection_context_evals']) for x in rows[-4:]],flush=True)
    return dict(k=k,seed=seed,selection=selection,rows=rows,audits=audits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'results');ap.add_argument('--k',type=int);ap.add_argument('--seed',type=int);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for k,seed in itertools.product([a.k] if a.k else P['widths'],[a.seed] if a.seed else P['seeds']):
        x=run(k,seed);(a.out/f'{k}_{seed}.json').write_text(json.dumps(x,indent=2)+'\n')
