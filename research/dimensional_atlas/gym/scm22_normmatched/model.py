"""SCM-22: an independent small weight-tied attention model for exact F11 algebra."""
from __future__ import annotations
import hashlib
import io
import random
import time
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

P=11
OP1=((1,1),(1,3),(2,1),(3,2))
OP2=((1,2),(2,4),(3,1),(4,3))

@dataclass(frozen=True)
class Case:
    first:int
    second:int
    x:int
    nuisance:int

def exact(c:Case)->int:
    a,b=OP1[c.first];t=(a*c.x+b)%P
    a,b=OP2[c.second];return (a*t+b)%P

def input_features(items:list[Case])->torch.Tensor:
    return torch.tensor([[x.first,x.second,x.x,x.nuisance] for x in items],dtype=torch.long)

def sample_cases(n:int,seed:int)->list[Case]:
    rng=random.Random(seed)
    return [Case(rng.randrange(4),rng.randrange(4),rng.randrange(P),rng.randrange(8)) for _ in range(n)]

def make_pairs(n:int,seed:int)->list[tuple[Case,Case]]:
    rng=random.Random(seed)
    pairs=[];seen=set()
    while len(pairs)<n:
        a=Case(rng.randrange(4),rng.randrange(4),rng.randrange(P),rng.randrange(8))
        b=Case((a.first+rng.randrange(1,4))%4,a.second,a.x,a.nuisance)
        sig=(a.first,b.first,a.second,a.x,a.nuisance)
        if sig in seen or exact(a)==exact(b):continue
        seen.add(sig);pairs.append((a,b))
    return pairs

class SharedBlock(nn.Module):
    def __init__(self,d:int=48,heads:int=4):
        super().__init__()
        assert d%heads==0
        self.heads=heads;self.scale=(d//heads)**-.5
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,2*d);self.fc2=nn.Linear(2*d,d)
    def forward(self,h:torch.Tensor)->torch.Tensor:
        b,t,d=h.shape
        q,k,v=self.qkv(self.ln1(h)).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        q=q.transpose(1,2);k=k.transpose(1,2);v=v.transpose(1,2)
        attention=(q@k.transpose(-1,-2)*self.scale).softmax(-1)
        h=h+.5*self.proj((attention@v).transpose(1,2).reshape(b,t,d))
        return h+.5*self.fc2(F.gelu(self.fc1(self.ln2(h))))

class LoopedMath(nn.Module):
    def __init__(self,d:int=48,visits:int=4):
        super().__init__()
        self.width=d;self.visits=visits
        self.op1=nn.Embedding(4,d);self.op2=nn.Embedding(4,d)
        self.operand=nn.Embedding(P,d);self.nuisance=nn.Embedding(8,d)
        self.cls=nn.Parameter(torch.randn(1,1,d)*.05)
        self.shared=SharedBlock(d);self.last=nn.LayerNorm(d);self.head=nn.Linear(d,P)
    def forward(self,x:torch.Tensor,*,patch:tuple[int,torch.Tensor]|None=None,record:bool=False):
        batch=x.shape[0]
        h=torch.cat([self.cls.expand(batch,-1,-1),torch.stack([
            self.op1(x[:,0]),self.op2(x[:,1]),self.operand(x[:,2]),self.nuisance(x[:,3])],1)],dim=1)
        states=[]
        for visit in range(self.visits):
            h=self.shared(h)
            if patch is not None and patch[0]==visit:
                assert patch[1].shape==(batch,self.width)
                h=torch.cat((h[:,:1]+patch[1][:,None,:],h[:,1:]),dim=1)
            if record:states.append(h[:,0])
        return self.head(self.last(h[:,0])),states

def checkpoint_digest(model:nn.Module)->str:
    # hash raw tensor bytes in deterministic key order, independent of torch save metadata
    hasher=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        data=value.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(name.encode());hasher.update(str(tuple(value.shape)).encode());hasher.update(data)
    return hasher.hexdigest()

def train(seed:int,updates:int=400,batch:int=128)->tuple[LoopedMath,dict]:
    torch.manual_seed(seed)
    model=LoopedMath();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    gen=random.Random(seed+101);start=time.perf_counter()
    for step in range(updates):
        examples=[Case(gen.randrange(4),gen.randrange(4),gen.randrange(P),gen.randrange(8)) for _ in range(batch)]
        model.train();pred,_=model(input_features(examples))
        target=torch.tensor([exact(x) for x in examples])
        loss=F.cross_entropy(pred,target)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
    model.eval()
    with torch.no_grad():
        sanity=sample_cases(512,seed+314159);out,_=model(input_features(sanity));accuracy=int((out.argmax(-1)==torch.tensor([exact(x) for x in sanity])).sum())
    return model,{'seed':seed,'updates':updates,'batch':batch,'training_examples':updates*batch,
                  'sanity_correct':accuracy,'sanity_n':512,
                  'checkpoint_sha256':checkpoint_digest(model), 'training_seconds':time.perf_counter()-start}
