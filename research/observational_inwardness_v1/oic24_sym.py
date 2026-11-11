"""Exploratory exact OIC-24 rotational-symmetry Bellman quotient."""
from oic23_exact import ExactFaultGame

class SymFaultGame(ExactFaultGame):
    def rotate_mask(self,mask,c):
        n=self.n
        if c==0:return mask
        row_mask=(1<<n)-1
        out=0
        for d in range(n):
            row=(mask>>(d*n))&row_mask
            rot=((row<<c)|(row>>(n-c)))&row_mask
            out|=rot<<(d*n)
        return out
    def canonical(self,state):
        best=state;best_c=0
        for c in range(1,self.n):
            rot=tuple(self.rotate_mask(mask,c) for mask in state)
            if rot<best:
                best=rot;best_c=c
        return best,best_c
    def _can(self,t,state):
        state,_=self.canonical(state)
        key=(t,state)
        if key in self._value_cache:return self._value_cache[key]
        self.visited+=1
        if self.visited>self.node_cap:raise RuntimeError(f'cap exceeded: {self.visited}')
        if len(self.labels(state))<=1:
            self._value_cache[key]=True
            return True
        if t>=self.Q:
            self._value_cache[key]=False
            return False
        for m in range(self.n):
            allpass=True
            for y in ((0,1,None) if self.r else (0,1)):
                child=self.next_state(state,t,m,y)
                if child is not None and not self._can(t+1,child):
                    allpass=False
                    break
            if allpass:
                self._choice[key]=m
                self._value_cache[key]=True
                return True
        self._value_cache[key]=False
        return False
    def certificate(self):
        if not self._value_cache.get((0,self.canonical(self.root)[0])):return None
        def build(t,state):
            ds=self.labels(state)
            if len(ds)<=1:return {'shift':ds[0]}
            canonical,c=self.canonical(state)
            mc=self._choice[(t,canonical)]
            m=(mc-c)%self.n
            children={}
            for y in ((0,1,None) if self.r else (0,1)):
                next_=self.next_state(state,t,m,y)
                if next_ is not None:children['?' if y is None else str(y)]=build(t+1,next_)
            return {'query':m,'children':children}
        return build(0,self.root)
