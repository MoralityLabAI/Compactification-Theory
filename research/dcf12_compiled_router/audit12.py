"""DCF-12 independent projection, causality and batch-composition audits."""
from router12 import *
from router import project

def joint(factors,p,k):
    x=np.arange(1<<k);q=np.ones((4,len(x)))
    for v,g in zip(factors,p):
        ix=sum(((x>>z)&1)<<i for i,z in enumerate(g));q*=v[:,ix]
    return q

def main():
    rng=np.random.default_rng(915);checks=[];cases=[];causal=[]
    for k in (6,8):
        for i in range(32):
            def randpart():
                x=rng.permutation(k);return tuple(sorted(tuple(sorted(y)) for y in np.array_split(x,int(rng.integers(1,k+1)))))
            a,b=randpart(),randpart();parts=[a,b];tab=projection_tables(parts);q=rng.dirichlet(np.ones(4));fac=[rng.dirichlet(np.ones(1<<len(g)),size=4) for g in a];buf=np.zeros((4,max(sum(1<<len(g) for g in p) for p in parts)))
            for s,f in zip(slices(a),fac):buf[:,s]=f
            out,d=proj_row(buf,q,0,1,*tab,True);old=joint(fac,a,k);new=joint([out[:,s] for s in slices(b)],b,k);direct=float(q@((old*(np.log(old)-np.log(new))).sum(1)));ref,refd=project(fac,a,b);delta=max(float(np.max(np.abs(out[:,s]-f))) for s,f in zip(slices(b),ref));err=max(abs(d-direct),abs(d-float(q@refd)));assert err<1e-12 and delta<1e-12
            checks.append(dict(k=k,case=i,kl_error=err,factor_error=delta))
    for path in sorted((HERE/'results').glob('*.json')):
        rec=json.loads(path.read_text());k,seed=rec['k'],rec['seed'];g,r=system(k,seed)
        for sel in rec['selection']:
            cap,parts,local,fixed=sel['cap'],[tuple(tuple(x) for x in p) for p in sel['parts']],np.array(sel['local']),sel['fixed']
            for si,sc in enumerate(P12['scenarios']):
                d,lab,e=fixture(k,g,r,sc,32,64,seed+P12['seed_offset']+30000+si);gamma=P12['gamma_per_row']*32
                p,pa,_=fast(d,lab,e,g,r,parts,local,fixed,True,gamma);ind,ia,_=fast(d,lab,e,g,r,parts,local,fixed,True,0.)
                alt=(*d[:-1],d[-1].copy());alt[-1][:,8:]^=1;ap,aa,_=fast(alt,lab,e,g,r,parts,local,fixed,True,gamma);err=float(np.max(np.abs(p[:,:9]-ap[:,:9])));assert err==0. and np.array_equal(pa[:,:9],aa[:,:9]);causal.append(dict(k=k,seed=seed,cap=cap,scenario=sc,error=err))
                for i in range(8):
                    solo=tuple(x[i:i+1] for x in d);s,sa,_=fast(solo,lab[i:i+1],e,g,r,parts,local,fixed,True,P12['gamma_per_row']);delta=float(np.max(np.abs(ind[i]-s[0])));assert delta<1e-10 and np.array_equal(ia[i],sa[0])
                    cases.append(dict(k=k,seed=seed,cap=cap,scenario=sc,row=i,independent_delta=delta,cohort_delta=float(np.max(np.abs(p[i]-s[0]))),route_changes=int((pa[i]!=sa[0]).sum()),cohort_minus_solo_nll=float((loss(p[i],d[-1][i])-loss(s[0],d[-1][i])).mean())))
    (HERE/'audit_receipt.json').write_text(json.dumps(dict(projections=checks,causal=causal,batch_composition=cases),indent=2)+'\n');print(len(checks),max(x['kl_error'] for x in checks),max(x['factor_error'] for x in checks),len(causal),len(cases),sum(x['route_changes']>0 for x in cases),max(x['cohort_delta'] for x in cases))
if __name__=='__main__':main()
