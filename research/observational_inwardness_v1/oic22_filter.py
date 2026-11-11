"""OIC-22: bounded-substitution Bayesian filter for adaptive labeled sensors.

Belief over exact finite (d,z0) hypotheses. A substitution may occur at most
once in the horizon. A marked erasure is skipped, and consumes no information.
The prior parameter alpha denotes probability of one substitution *somewhere*
in the Q-step episode, with its location uniformly random; the other
probability 1-alpha is an uncorrupted episode. No privileged phase or d.
"""
from __future__ import annotations
import base64, gzip, json, math
from pathlib import Path
from functools import lru_cache
import numpy as np
from math import gcd, log2


def _entropy_mass(a):
    mass=a.sum()
    if mass<=0:return 0.0
    p=a/mass
    return -float(np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-300)),0.0)))


class OnlineFilter:
    def __init__(self,n,noise=0.04):
        self.n=n
        self.states=np.array([(d,z) for d in range(n) for z in range(n)],dtype=np.int64)
        self.shifts=self.states[:,0]
        self.groups=np.array([gcd(n,int(d)) for d in self.shifts])
        self.hit=np.stack([np.stack([((self.states[:,1]+t*self.shifts)%n==m) for m in range(n)],axis=0) for t in range(n+2)],axis=0)
    def features(self,w,t):
        n=self.n
        byshift=np.bincount(self.shifts,weights=w,minlength=n)
        mx=float(byshift.max());entropy=_entropy_mass(byshift)
        bygroup=np.bincount(self.groups,weights=w,minlength=n+1)
        gx=float(bygroup.max());g_h=_entropy_mass(bygroup)
        out=[]
        for m in range(n):
            hits=self.hit[t,m]
            hitw=w*hits
            no=w-hitw
            p=float(hitw.sum());p=max(0.0,min(1.0,p))
            hshift=np.bincount(self.shifts,weights=hitw,minlength=n)
            nshift=byshift-hshift
            hgroup=np.bincount(self.groups,weights=hitw,minlength=n+1)
            ngroup=bygroup-hgroup
            ag=float(hshift.max()+nshift.max()-mx)
            gg=float(hgroup.max()+ngroup.max()-gx)
            eh=entropy-p*_entropy_mass(hshift)-(1-p)*_entropy_mass(nshift)
            gh=g_h-p*_entropy_mass(hgroup)-(1-p)*_entropy_mass(ngroup)
            out.append((ag,max(0.,eh/log2(n)),p*(1-p),gg,max(0.,gh/log2(n)),p,1/(1+2*abs(p-.5))))
        return np.array(out,dtype=np.float32)


HERE=Path(__file__).resolve().parent


def archived_scorers():
    p=HERE/'results'/'oic21_fits.json.gz.b64'
    archive=json.loads(gzip.decompress(base64.b64decode(p.read_bytes().strip(),validate=True)))
    return {(f['metadata']['split'],f['metadata']['seed']):f['weights'] for f in archive['fits']}


def scorer_values(x,params):
    w1=np.asarray(params['network.0.weight'],dtype=np.float32)
    b1=np.asarray(params['network.0.bias'],dtype=np.float32)
    w2=np.asarray(params['network.2.weight'],dtype=np.float32)
    b2=np.asarray(params['network.2.bias'],dtype=np.float32)
    return (np.tanh(x@w1.T+b1)@w2.T+b2).ravel()


class BoundedFlipObserver:
    """Exact at-most-one mismatch mask with a declared fault-location prior."""
    def __init__(self,n,weights=None,alpha=0.5,strategy='learned',q=None):
        if n<2 or not 0<=alpha<=1 or strategy not in ('learned','greedy','entropy','cycle','fixed0'):
            raise ValueError('invalid OIC-22 filter configuration')
        self.n=n;self.Q=q or n+2;self.alpha=float(alpha);self.weights=weights;self.strategy=strategy
        self.o=OnlineFilter(n,noise=0.04)
        if self.Q>n+2:
            raise ValueError('OnlineFilter only precomputes n+2 steps; use Q=n+2 here')
        self.d=np.repeat(np.arange(n),n)
        self.initial=np.tile(np.arange(n),n)
    @lru_cache(None)
    def state(self,history):
        t=len(history)
        if t==0:
            c0=np.ones(self.n*self.n,dtype=bool)
            c1=np.zeros(self.n*self.n,dtype=bool)
        else:
            c0prev,c1prev,_=self.state(history[:-1])
            prev_t=t-1
            prev_m=self.state(history[:-1])[2]
            bit=history[-1]
            if bit is None:
                c0,c1=c0prev,c1prev
            else:
                matches=self.o.hit[prev_t,prev_m]==bool(bit)
                c0=c0prev&matches
                c1=(c1prev&matches)|(c0prev&~matches)
        w0=(1-self.alpha)+self.alpha*(self.Q-t)/self.Q
        w1=self.alpha/self.Q
        masses=c0.astype(float)*w0+c1.astype(float)*w1
        if masses.sum()<=0:
            # If a declared >1-flip corruption happens, the filter must report
            # failure rather than fabricate a posterior; caller detects this.
            raise ValueError('received history incompatible with one-substitution budget')
        w=masses/masses.sum()
        if t>=self.Q:return (c0,c1,None)
        if self.strategy=='cycle':m=t%self.n
        elif self.strategy=='fixed0':m=0
        else:
            features=self.o.features(w,t)
            if self.strategy=='greedy':val=features[:,0]+1e-5*features[:,1]
            elif self.strategy=='entropy':val=features[:,1]
            else:val=scorer_values(features,self.weights)
            m=int(np.argmax(np.round(val.astype(np.float64),decimals=4)))
        return c0,c1,m
    def posterior(self,history):
        c0,c1,_=self.state(tuple(history));t=len(history)
        w=(c0.astype(float)*((1-self.alpha)+self.alpha*(self.Q-t)/self.Q)
            +c1.astype(float)*self.alpha/self.Q)
        w/=w.sum()
        masses=np.bincount(self.d,weights=w,minlength=self.n)
        return masses
    def episode(self,d,z,kind='clean',pos=None):
        history=()
        addresses=[]
        for t in range(self.Q):
            _,_,m=self.state(history)
            addresses.append(m)
            raw=int((z+d*t)%self.n==m)
            if kind=='flip' and pos==t:raw=1-raw
            elif kind=='erase' and pos==t:raw=None
            history+=(raw,)
        p=self.posterior(history)
        return {'correct':int(np.argmax(p)==d),'posterior_true':float(p[d]),
                'estimate':int(np.argmax(p)), 'actions':addresses,'received':history}
    def panels(self, kinds=('clean','flip','erase')):
        result=[]
        for kind in kinds:
            correct=0;mass=0.0;count=0
            positions=(None,) if kind=='clean' else range(self.Q)
            for pos in positions:
                for d in range(self.n):
                    for z in range(self.n):
                        trial=self.episode(d,z,kind,pos)
                        correct+=trial['correct'];mass+=trial['posterior_true'];count+=1
            result.append({'kind':kind,'correct':correct,'episodes':count,
                           'accuracy':correct/count,'mean_posterior_true':mass/count})
        return result
