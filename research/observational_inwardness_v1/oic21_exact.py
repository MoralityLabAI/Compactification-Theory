"""Exact finite-state active single-marker OIC-21 design (standard library)."""
from functools import lru_cache
from itertools import product
from math import gcd, log2

class ExactActiveOracle:
    def __init__(self,n,horizon):
        if not isinstance(n,int) or n<2 or not isinstance(horizon,int) or horizon<0:raise ValueError('bad n or horizon')
        self.n=n;self.horizon=horizon;self.all=(1<<(n*n))-1
        self.masks=tuple(tuple(sum((1<<(d*n+z)) for d in range(n) for z in range(n) if (z+d*t)%n==m)
                for m in range(n)) for t in range(horizon))
        self.d_masks=tuple(((1<<n)-1)<<(d*n) for d in range(n))
        self.g_masks=tuple(sum(self.d_masks[d] for d in range(n) if gcd(n,d)==g) for g in sorted({gcd(n,d) for d in range(n)}))
    def split(self,mask,t,m):
        yes=mask&self.masks[t][m]
        return yes,mask^yes
    def terminal(self,mask,target='shift'):
        classes=self.d_masks if target=='shift' else self.g_masks
        return max((mask&cls).bit_count() for cls in classes) if mask else 0
    @lru_cache(None)
    def best_adaptive(self,mask,t,target='shift'):
        if not mask:return 0
        if t==self.horizon:return self.terminal(mask,target)
        return max(self.best_adaptive(yes,t+1,target)+self.best_adaptive(no,t+1,target)
                   for marker in range(self.n) for yes,no in (self.split(mask,t,marker),))
    def best_action(self,mask,t,target='shift'):
        if t>=self.horizon:raise ValueError('at horizon')
        values=[self.best_adaptive(yes,t+1,target)+self.best_adaptive(no,t+1,target)
                for marker in range(self.n) for yes,no in (self.split(mask,t,marker),)]
        return values.index(max(values)),max(values)
    def fixed_score(self,word,target='shift'):
        states=[self.all]
        for t,m in enumerate(word):
            nexts=[]
            for mask in states:
                a,b=self.split(mask,t,m)
                if a:nexts.append(a)
                if b:nexts.append(b)
            states=nexts
        return sum(self.terminal(mask,target) for mask in states)
    def best_fixed(self,target='shift'):
        if self.horizon==0:return (),self.terminal(self.all,target)
        # translational symmetry: fix first query marker to zero
        best_word,best_score=None,-1
        for word_rest in product(range(self.n), repeat=self.horizon-1):
            word=(0,)+word_rest
            score=self.fixed_score(word,target)
            if score>best_score:
                best_score=score;best_word=word
        return best_word,best_score
    def prior_accuracy(self,target='shift'):
        return self.terminal(self.all,target)/(self.n*self.n)
    def adaptive_accuracy(self,target='shift'):
        return self.best_adaptive(self.all,0,target)/(self.n*self.n)
    def fixed_accuracy(self,target='shift'):
        w,s=self.best_fixed(target)
        return s/(self.n*self.n),w

if __name__=='__main__':
 import argparse,time
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=4);p.add_argument('--h',type=int,default=6)
 a=p.parse_args(); start=time.time();o=ExactActiveOracle(a.n,a.h)
 print('n',a.n,'horizon',a.h,'prior',o.prior_accuracy());print('adaptive',o.adaptive_accuracy(),'states',o.best_adaptive.cache_info())
 print('fixed',o.fixed_accuracy(),'elapsed',time.time()-start)
