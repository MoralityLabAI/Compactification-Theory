"""Independent joint-state enumeration; explicit context and hidden-bit mass transport."""
from experiment import *
def brute(data,labels,e,g,r,h):
    j,c,read,_,y=data;n,l=j.shape;k=len(g[0]);M=1<<k;out=np.empty((n,l));prior=np.empty((n,l,Q));post=np.empty_like(prior)
    A=transition(h)
    for i in range(n):
        p=np.full((Q,M),1/(Q*M))
        for t in range(l):
            if t:p=A.T@p
            p*=label_lik(labels[i:i+1,t],e)[0,:,None];p/=p.sum();prior[i,t]=p.sum(1)
            a,b,d=int(j[i,t]),int(c[i,t]),int(read[i,t]);f=r['fail'][a] if b else r['idle'][a];nxt=np.zeros_like(p)
            for z in range(Q):
                pa=g[z][a]
                for s in range(M):
                    ctrl=sum((s>>v)&1 for v in pa)%2 if pa else 1;dest=s^(b*ctrl<<a);w=p[z,s];nxt[z,dest]+=w*(1-f);nxt[z,dest^(1<<a)]+=w*f
            lik=np.array([r['err'][d]+(1-2*r['err'][d])*((s>>d)&1) for s in range(M)])
            out[i,t]=(nxt*lik).sum();p=nxt*(lik if y[i,t] else 1-lik);p/=p.sum();post[i,t]=p.sum(1)
    return out,prior,post

def main():
    checks=[]
    for k,seed,sc in itertools.product(S['widths'],S['seeds'],('clean','noisy30','hidden','bursty30')):
        g,r=system(k,seed);d,lab,e=fixture(k,g,r,sc,3,20,seed+123400);a=filter_belief(d,lab,e,g,r,(tuple(range(k)),),'mixture');b=brute(d,lab,e,g,r,S['assumed_hazard']);err=max(float(np.max(np.abs(x-y))) for x,y in zip(a[:3],b));assert err<1e-12;checks.append(dict(k=k,seed=seed,scenario=sc,max_delta=err))
    (ROOT/'audit_receipt.json').write_text(json.dumps(checks,indent=2)+'\n');print(len(checks),max(x['max_delta'] for x in checks))
if __name__=='__main__':main()
