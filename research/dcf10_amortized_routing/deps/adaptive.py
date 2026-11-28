"""DCF-09B: posterior-weighted partition routing, fixed budget and full-state controls."""
from experiment import *
from gym import partition
from router import project
F=json.loads((ROOT/'followup_spec.json').read_text())

def pool(g,b):
    ps=list(dict.fromkeys(candidates(g)+[tuple(partition(h,cap)) for h in g for cap in S['partition_caps']]+[(tuple(range(len(g[0]))),)]))
    return [p for p in ps if size(p,'mixture')<=b]

def dynamic(d,lab,e,g,r,parts,local,fixed,mu):
    j,c,read,_,y=d;n,l=j.shape;A=transition(S['assumed_hazard']);q=np.full((n,Q),.25);sizes=np.array([size(p,'mixture') for p in parts]);ks=[[Kernel(h,p,r) for h in g] for p in parts];states={fixed:(np.arange(n),[np.full((n,Q,1<<len(a)),1/(1<<len(a))) for a in parts[fixed]])};out=np.empty((n,l));cp=np.empty((n,l,Q));ca=np.empty_like(cp);bytes_used=np.empty((n,l));sw=np.zeros(n,int);lost=np.zeros(n);work=0;checks=0;start=time.perf_counter()
    for t in range(l):
        if t:
            prior=q@A;mix=q[:,:,None]*A[None,:,:]/prior[:,None,:]
            for p,(ix,ps) in states.items():states[p]=(ix,[np.einsum('nab,nai->nbi',mix[ix],x) for x in ps])
            q=prior
        q=normalize(q*label_lik(lab[:,t],e));cp[:,t]=q;dest_groups={}
        for old,(ix,ps) in states.items():
            expected=q[ix]@local;best=expected.argmin(1);choices=np.stack([np.full(len(ix),old),best,np.full(len(ix),fixed)],1);cost=np.take_along_axis(expected,choices,1);cache={}
            for p in np.unique(choices):
                if p==old:continue
                # Projection is calculated conditionally on each possible context, then weighted by posterior q.
                flat=[x.reshape(len(ix)*Q,-1) for x in ps];_,drop=project(flat,parts[old],parts[p],False);drop=(drop.reshape(len(ix),Q)*q[ix]).sum(1);cache[p]=drop;checks+=len(ix)*Q;cost+=np.where(choices==p,(F['fee']+mu*drop[:,None])/F['horizon'],0.)
            target=choices[np.arange(len(ix)),cost.argmin(1)]
            for p in np.unique(target):
                ii=np.flatnonzero(target==p);ids=ix[ii];factors=[x[ii] for x in ps]
                if p!=old:
                    new,drop=project([x.reshape(len(ii)*Q,-1) for x in factors],parts[old],parts[p]);factors=[x.reshape(len(ii),Q,-1) for x in new];sw[ids]+=1;lost[ids]+=(drop.reshape(len(ii),Q)*q[ids]).sum(1)
                dest_groups.setdefault(p,[]).append((ids,factors))
        states={p:(np.concatenate([x[0] for x in rows]),[np.concatenate([x[1][i] for x in rows]) for i in range(len(parts[p]))]) for p,rows in dest_groups.items()}
        for p,(ix,ps) in states.items():
            pred=np.empty((len(ix),Q));post=[]
            for z in range(Q):
                f=[x[:,z].copy() for x in ps];pred[:,z]=ks[p][z].step(f,j[ix,t],c[ix,t],read[ix,t],y[ix,t]);post.append(f);work+=len(ix)
            out[ix,t]=(q[ix]*pred).sum(1);q[ix]=normalize(q[ix]*np.where(y[ix,t,None],pred,1-pred));states[p]=(ix,[np.stack([post[a][i] for a in range(Q)],1) for i in range(len(ps))]);bytes_used[ix,t]=sizes[p]
        ca[:,t]=q
    assert np.isfinite(out).all() and out.min()>0 and out.max()<1
    return out,cp,ca,dict(bytes=int(bytes_used.max()),mean_bytes=float(bytes_used.mean()),switches_per_word=float(sw.mean()),discarded_kl_per_word=float(lost.mean()),projection_context_evals=checks,kernel_rows=work,particle_steps=0,wall_s=time.perf_counter()-start)

def main(k,seed,out):
    g,r=system(k,seed);off=F['seed_offset'];full=(tuple(range(k)),);rows=[];val=[]
    # Full bit-state controls, regardless of whether they fit each budget, reported with their actual size.
    tests=[fixture(k,g,r,sc,F['test_words'],F['test_length'],seed+off+20000+i) for i,sc in enumerate(F['scenarios'])]
    refs=[]
    for sc,(d,lab,e) in zip(F['scenarios'],tests):
        ref=filter_belief(d,lab,e,g,r,full,'mixture');priv=filter_belief(d,d[3],0,g,r,full,'mixture')[0];refs.append((ref[0],priv))
        for mode in ('map','shared','mixture'):
            a=ref if mode=='mixture' else filter_belief(d,lab,e,g,r,full,mode);rows.append(dict(scenario=sc,budget=None,arm='full_'+mode,**measure(a,d,ref[0],priv)))
    vf=[fixture(k,g,r,sc,F['val_words'],F['val_length'],seed+off+10000+i) for i,sc in enumerate(F['scenarios'])]
    clean=[]
    for z in range(Q):
        a=sample(k,g,r,F['val_words'],F['val_length'],seed+off+11000+z,z);clean.append((a,a[3],0.))
    allparts=pool(g,max(F['budgets']));local=np.empty((Q,len(allparts)));joint=[]
    for i,p in enumerate(allparts):
        for z,(d,lab,e) in enumerate(clean):local[z,i]=loss(filter_belief(d,lab,e,g,r,p,'mixture')[0],d[-1]).mean()
        joint.append(float(np.mean([loss(filter_belief(d,lab,e,g,r,p,'mixture')[0],d[-1]).mean() for d,lab,e in vf])))
    for b in F['budgets']:
        ix=[i for i,p in enumerate(allparts) if size(p,'mixture')<=b];parts=[allparts[i] for i in ix];loc=local[:,ix];fixed=min(range(len(ix)),key=lambda i:(joint[ix[i]],size(parts[i],'mixture')));val.append(dict(budget=b,parts=parts,local=loc.tolist(),joint=[joint[i] for i in ix],fixed=fixed))
        for sc,(d,lab,e),(ref,priv) in zip(F['scenarios'],tests,refs):
            a=filter_belief(d,lab,e,g,r,parts[fixed],'mixture');a[3].update(mean_bytes=a[3]['bytes'],switches_per_word=0.,discarded_kl_per_word=0.,projection_context_evals=0);rows.append(dict(scenario=sc,budget=b,arm='fixed',**measure(a,d,ref,priv)))
            for mu in (0.,F['mu']):
                a=dynamic(d,lab,e,g,r,parts,loc,fixed,mu);assert a[3]['bytes']<=b;rows.append(dict(scenario=sc,budget=b,arm='correlation' if mu else 'dwell',**measure(a,d,ref,priv)))
    out.mkdir(parents=True,exist_ok=True);(out/'receipt.json').write_text(json.dumps(dict(k=k,seed=seed,validation=val,rows=rows),separators=(',',':'))+'\n');print(k,seed,[(x['arm'],x['budget'],x['scenario'],round(x['kl_same_info'],5)) for x in rows],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.k,a.seed,a.out)
