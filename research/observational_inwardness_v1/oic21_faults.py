"""OIC-21 noisy online sensor selection, exact forward stochastic posterior.

Sensor position chosen adaptively from the *received* (possibly corrupted) bits.
The prior, candidate cyclic transition laws, and raw sensor likelihood are
supplied by the observer. We evaluate one substitution or one marked erasure
at every possible position over ALL (d,z0) hypotheses for n=4,6,8.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from functools import lru_cache
import numpy as np
from math import gcd, log2
from oic21_train import ScoreNet
from oic21_bench import load_model


def entropy_mass(a):
    s=a.sum()
    if not s:return 0.0
    p=a/s
    return -float(np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-300)),0.0)))


class OnlineFilter:
    def __init__(self,n,noise=0.04):
        if not 0<noise<0.5:raise ValueError('0<noise<1/2 required')
        self.n=n;self.noise=noise
        self.states=np.array([(d,z) for d in range(n) for z in range(n)],dtype=np.int64)
        self.groups=np.array([gcd(n,d) for d,z in self.states])
        self.shifts=self.states[:,0]
        self.hit=np.stack([np.stack([((self.states[:,1]+t*self.shifts)%n==m) for m in range(n)],axis=0) for t in range(n+2)],axis=0)
    def features(self,w,t):
        n=self.n
        byshift=np.bincount(self.shifts,weights=w,minlength=n)
        pmax=byshift.max()
        entropy=entropy_mass(byshift)
        bygroup=np.bincount(self.groups,weights=w,minlength=n+1)
        groupmax=bygroup.max();grouph=entropy_mass(bygroup)
        feats=[]
        for m in range(n):
            hits=self.hit[t,m]
            hitw=w*hits
            no=w-hitw
            p=float(hitw.sum()); p=min(1.0,max(0.0,p))
            hshift=np.bincount(self.shifts,weights=hitw,minlength=n)
            nshift=byshift-hshift
            hgroup=np.bincount(self.groups,weights=hitw,minlength=n+1)
            ngroup=bygroup-hgroup
            pma=float(hshift.max()+nshift.max()-pmax)
            pgma=float(hgroup.max()+ngroup.max()-groupmax)
            eh=entropy-p*entropy_mass(hshift)-(1-p)*entropy_mass(nshift)
            gh=grouph-p*entropy_mass(hgroup)-(1-p)*entropy_mass(ngroup)
            feats.append((pma,max(0.,eh/log2(n)),p*(1-p),
                         pgma,max(0.,gh/log2(n)),p,1/(1+2*abs(p-.5))))
        return np.array(feats,dtype=np.float32)
    def update(self,w,t,m,received):
        if received is None:return w
        hits=self.hit[t,m]
        match=hits if received else ~hits
        w=np.where(match,w*(1-self.noise),w*self.noise)
        s=w.sum()
        if s==0:raise ValueError('numerical posterior underflow')
        return w/s
    def predict(self,w):
        mass=np.bincount(self.shifts,weights=w,minlength=self.n)
        return int(mass.argmax()),mass


def score_pos(features,weights):
    x=features
    w1=np.array(weights['network.0.weight'],dtype=np.float32)
    b1=np.array(weights['network.0.bias'],dtype=np.float32)
    w2=np.array(weights['network.2.weight'],dtype=np.float32)
    b2=np.array(weights['network.2.bias'],dtype=np.float32)
    return int(np.argmax((np.tanh(x@w1.T+b1)@w2.T+b2).ravel()))


def episode(filt:OnlineFilter,weights,d,z,kind=None,position=None):
    n=filt.n;Q=n+2
    w=np.ones(n*n,dtype=np.float64)/(n*n)
    actions=[];observed=[]
    for t in range(Q):
        marks=filt.features(w,t)
        m=score_pos(marks,weights)
        raw=int((z+d*t)%n==m)
        obs=raw
        if t==position and kind=='flip':obs=1-raw
        if t==position and kind=='erase':obs=None
        actions.append(m);observed.append(obs)
        w=filt.update(w,t,m,obs)
    estimate,posterior=filt.predict(w)
    return {'correct':int(estimate==d),'estimate':estimate,'posterior_true':float(posterior[d]),
            'actions':actions,'observed':observed}


def exhaustive_fault_panel(n,weights,noise=0.04):
    filt=OnlineFilter(n,noise=noise)
    Q=n+2
    out=[]
    for kind in ('clean','flip','erase'):
        right=0;cnt=0;mass=0
        locations=(None,) if kind=='clean' else tuple(range(Q))
        for pos in locations:
            for d in range(n):
                for z in range(n):
                    e=episode(filt,weights,d,z,kind=kind,position=pos)
                    right+=e['correct'];cnt+=1;mass+=e['posterior_true']
        out.append({'kind':kind,'correct':right,'episodes':cnt,'accuracy':right/cnt,
                    'mean_posterior_true':mass/cnt})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--splits',nargs='+',default=['full','heldout']);p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)));p.add_argument('--ns',type=int,nargs='+',default=[4,6,8]);a=p.parse_args()
    start=time.time();out=[]
    for n in a.ns:
        for split in a.splits:
            for seed in a.seeds:
                _,r=load_model(split,seed)
                result=exhaustive_fault_panel(n,r['weights'])
                out.append({'n':n,'split':split,'seed':seed,'panels':result})
                print('n',n,split,seed,{v['kind']:round(v['accuracy'],4) for v in result},flush=True)
    result={'status':'evaluated_online_bayesian_corruption_panels',
            'sensor_noise_assumption':0.04,
            'n':a.ns,'splits':a.splits,'seeds':a.seeds,'results':out,
            'elapsed_seconds':round(time.time()-start,3)}
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()

class CachedOnlinePolicy:
    """Memoize policy/belief by public received history, exact same filter."""
    def __init__(self,n,weights,noise=0.04):
        self.filter=OnlineFilter(n,noise)
        self.Q=n+2
        self.w1=np.array(weights['network.0.weight'],dtype=np.float32)
        self.b1=np.array(weights['network.0.bias'],dtype=np.float32)
        self.w2=np.array(weights['network.2.weight'],dtype=np.float32)
        self.b2=np.array(weights['network.2.bias'],dtype=np.float32)
    @lru_cache(None)
    def state(self,history):
        t=len(history)
        if t==0:
            w=np.ones(self.filter.n*self.filter.n,dtype=np.float64)/(self.filter.n*self.filter.n)
        else:
            wp,m_prev=self.state(history[:-1]);w=self.filter.update(wp,t-1,m_prev,history[-1])
        if t==self.Q:return w,None
        feats=self.filter.features(w,t)
        scores=(np.tanh(feats@self.w1.T+self.b1)@self.w2.T+self.b2).ravel()
        m=int(np.argmax(scores))
        return w,m
    def episode(self,d,z,kind=None,position=None):
        n=self.filter.n
        history=()
        for t in range(self.Q):
            _,m=self.state(history)
            true_hit=int((z+d*t)%n==m)
            obs=true_hit
            if kind=='flip' and position==t:obs=1-obs
            if kind=='erase' and position==t:obs=None
            history+= (obs,)
        w,_=self.state(history)
        estimate,p=self.filter.predict(w)
        return int(estimate==d),float(p[d])
    def panels(self):
        n=self.filter.n
        out=[]
        for kind in ('clean','flip','erase'):
            corr=0;mass=0;episodes=0
            positions=(None,) if kind=='clean' else range(self.Q)
            for pos in positions:
                for d in range(n):
                    for z in range(n):
                        right,pr=self.episode(d,z,kind=kind,position=pos)
                        corr+=right;mass+=pr;episodes+=1
            out.append({'kind':kind,'correct':corr,'episodes':episodes,'accuracy':corr/episodes,
                        'mean_posterior_true':mass/episodes})
        return out


def fast_fault_panel(n,weights,noise=0.04):
    planner=CachedOnlinePolicy(n,weights,noise)
    records=planner.panels()
    return records,planner.state.cache_info()._asdict()
