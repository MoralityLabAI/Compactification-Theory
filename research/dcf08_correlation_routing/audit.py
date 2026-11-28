"""Independent full enumeration of projection KL, and marginal preservation."""
from router import *

def joint(ps,part,k):
    x=np.arange(1<<k);q=np.ones((len(ps[0]),len(x)))
    for p,g in zip(ps,part):
        ix=sum(((x>>z)&1)<<i for i,z in enumerate(g));q*=p[:,ix]
    return q

def main():
    g=np.random.default_rng(703);rows=[]
    for k in (6,8):
        for i in range(32):
            def part():
                a=g.permutation(k).tolist();cuts=sorted(g.choice(np.arange(1,k),size=int(g.integers(0,k)),replace=False));return tuple(sorted(tuple(sorted(x)) for x in np.split(a,cuts)))
            old,new=part(),part();ps=[g.dirichlet(np.ones(1<<len(h)),size=8) for h in old];q,d=project(ps,old,new);p0=joint(ps,old,k);p1=joint(q,new,k);direct=(p0*(np.log(p0)-np.log(p1))).sum(1);err=float(np.max(np.abs(d-direct)));assert err<1e-12
            # Check every target factor is the correct marginal under original joint.
            x=np.arange(1<<k);mx=0.
            for h,m in zip(new,q):
                ix=sum(((x>>z)&1)<<j for j,z in enumerate(h));m0=p0@np.eye(1<<len(h))[ix];mx=max(mx,float(np.max(np.abs(m-m0))))
            assert mx<1e-12;rows.append(dict(k=k,case=i,kl_error=err,marginal_error=mx))
    (Path(__file__).parent/'audit_receipt.json').write_text(json.dumps(rows,indent=2)+'\n');print(len(rows),max(r['kl_error'] for r in rows),max(r['marginal_error'] for r in rows))
if __name__=='__main__':main()
