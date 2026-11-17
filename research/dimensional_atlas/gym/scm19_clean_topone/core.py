"""SCM-19 public features, correctness labels, small tied model and accounting."""
from __future__ import annotations
import functools
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..model import SharedTopKSAE,TiedBlock
from ..tasks import PROGRAMS,TOPOLOGIES,MathTask,make_evalmath_case,load_evalmath
from ..scm18_rtx3050.core import canonical,digest,sha,write_new,device_setup,release

HERE=Path(__file__).resolve().parent
FAMILIES=('signed','product','sparse')
OOD=('rational','squared','lacunary','twisted_product')
NEXT={'I':'N','N':'S','S':'R','R':'I'}


class Journal:
    def __init__(self,path,config):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('a',encoding='utf-8',buffering=1,newline='\n')
        self.config=config;self.work=Counter();self.stage='initialization';self.started=time.perf_counter()
        self.min_free=None
    def event(self,kind,**values):
        self.file.write(json.dumps({'event':kind,'stage':self.stage,'elapsed':time.perf_counter()-self.started,**values},sort_keys=True,allow_nan=False)+'\n')
    def memory(self):
        if not torch.cuda.is_initialized():return {'allocated_mb':0.,'reserved_mb':0.,'free_mb':None}
        free,_=torch.cuda.mem_get_info();self.min_free=free if self.min_free is None else min(free,self.min_free)
        if free<self.config['min_free_vram_mb']*2**20:raise RuntimeError('VRAM headroom below registered limit')
        return {'allocated_mb':torch.cuda.max_memory_allocated()/2**20,
                'reserved_mb':torch.cuda.max_memory_reserved()/2**20,'free_mb':free/2**20}
    def forward(self,batch,tokens,visits):
        self.work.update(model_calls=1,forward_examples=batch,recurrent_example_visits=batch*visits,
                         recurrent_token_visits=batch*tokens*visits)
        self.event('forward_called',batch=batch,tokens=tokens,visits=visits,**self.memory())
    def report(self):
        return {'work':dict(self.work),'seconds':time.perf_counter()-self.started,
                'minimum_sampled_free_mb':self.min_free/2**20 if self.min_free is not None else None,
                'peak':self.memory(),'memory_sampling':'every forward entry and optimizer completion'}


class OracleAudit:
    def __init__(self,source,journal):
        self.module=load_evalmath(Path(source));self.counts=Counter();self.journal=journal;self.caches={}
        # All cached functions are exact integer functions; no model/gold decisions cached.
        for name in ('unary','convolution','reciprocal','execute'):
            original=getattr(self.module,name)
            @functools.lru_cache(maxsize=32768)
            def cached(*args,_original=original):return _original(*args)
            self.caches[name]=cached
            def wrapped(*args,_name=name,_cached=cached):
                self.counts[_name]+=1
                normalized=tuple(tuple(v) if isinstance(v,list) else v for v in args)
                return _cached(*normalized)
            setattr(self.module,name,wrapped)
        for name in ('make_case','check_candidate','grade'):
            original=getattr(self.module,name)
            def wrapped(*args,_name=name,_original=original,**kwargs):
                self.counts[_name]+=1
                if _name in ('check_candidate','grade'):journal.event('oracle_call',name=_name)
                return _original(*args,**kwargs)
            setattr(self.module,name,wrapped)
        def forbidden(*args,**kwargs):raise RuntimeError('official EvalMath roster forbidden throughout SCM19')
        self.module.roster=forbidden
    def report(self):
        return {'source_requests':dict(self.counts),'physical_cache':{k:v.cache_info()._asdict() for k,v in self.caches.items()},'official_roster_calls':0}


def partitions(n,minimum=1):
    if n==0:yield ()
    for first in range(minimum,n+1):
        for tail in partitions(n-first,first):yield (first,)+tail


PARTITIONS={n:list(partitions(n)) for n in range(9)}


def numeric(v):
    v=int(v);a=abs(v)
    return [math.copysign(math.log1p(a)/8,v) if v else 0.,v/(1+a),float(v==0),
            (v%17-8)/8,(v%31-15)/15]


def features(task):
    """Read only public demos/topology, never source graph/query future/oracle."""
    rows=[]
    for demo,(xs,ys) in enumerate(task.demos):
        items=[]
        for n in range(13):
            x,y=xs[n],ys[n]
            basic=numeric(x)+numeric(y)+numeric(y-x)+numeric(y+x)
            basic += [n/12,float(n%2),float(demo==0),float(demo==1)]
            basic += [float(task.topology==topo) for topo in TOPOLOGIES]
            monomials=[]
            if n<=8:
                for parts in PARTITIONS[n]:
                    value=1
                    for part in parts:value*=xs[part]
                    monomials.append(value)
            assert len(monomials)<=22
            monomials += [0]*(22-len(monomials))
            scale=max(1,abs(y),*(abs(v) for v in monomials))
            vector=[y/scale]+[v/scale for v in monomials]
            items.append(basic+vector)
        rows.append(items)
    value=torch.tensor(rows,dtype=torch.float32)
    if value.shape!=(2,13,51) or not torch.isfinite(value).all():raise ValueError('invalid public feature tensor')
    return value


def groups(oracle):
    series=[oracle.source(random.Random(19200000+i),'signed') for i in range(4)]
    result={};sizes={}
    for topology in TOPOLOGIES:
        fingerprints={g:tuple(oracle.execute(x,topology,g) for x in series) for g in PROGRAMS}
        result[topology]={g:[q for q in PROGRAMS if fingerprints[q]==fingerprints[g]] for g in PROGRAMS}
        sizes[topology]=len(set(fingerprints.values()))
    return result,{'probe_input_sha256':canonical(series),'finite_probe_class_counts':sizes,
                   'status':'candidate_groups_verified_again_on_each_training_case_not_global_identity_certificates'}


def correct_mask(task,oracle,grouping):
    mask=torch.zeros(64,dtype=torch.bool)
    pairs=list(task.oracle_case.demos)+[(task.oracle_case.query,task.oracle_case.future)]
    for graph in grouping[task.topology][task.source_graph]:
        if all(oracle.execute(x,task.topology,graph)==y for x,y in pairs):mask[PROGRAMS.index(graph)]=True
    assert bool(mask[PROGRAMS.index(task.source_graph)])
    return mask


def make_pool(oracle,count,start,role,families,seen):
    result=[]
    for attempt in range(count*40):
        family=families[attempt%len(families)];topology=TOPOLOGIES[attempt%4]
        graph=PROGRAMS[(17*(attempt//4))%64]
        case=make_evalmath_case(oracle,start+attempt,case_id=f'scm19/{role}/{start+attempt}',
                              family=family,topology=topology,graph=graph)
        if case.signature in seen:continue
        seen.add(case.signature);result.append(case)
        if len(result)==count:return result
    raise RuntimeError('unique pool exhausted')


def counterfactual(task,oracle):
    graph=(NEXT[task.source_graph[0]],*task.source_graph[1:]);case=task.oracle_case
    demos=tuple((x,oracle.execute(x,task.topology,graph)) for x,_ in case.demos)
    future=oracle.execute(case.query,task.topology,graph)
    other=replace(case,identifier=case.identifier+'/donor',demos=demos,future=future,source_graph=graph)
    return MathTask(task.case_id+'/donor',task.family,task.topology,
        tuple((tuple(x[:13]),tuple(y[:13])) for x,y in demos),task.query,tuple(future),graph,'evalmath_arg01',other)


def make_pairs(oracle,count,start,role,families,seen,encoding):
    result=[]
    for attempt in range(count*100):
        a=make_evalmath_case(oracle,start+attempt,case_id=f'scm19/{role}/{start+attempt}',
              family=families[attempt%len(families)],topology=TOPOLOGIES[attempt%4],graph=PROGRAMS[(17*(attempt//4))%64])
        b=counterfactual(a,oracle)
        if a.hidden_target==b.hidden_target or a.signature==b.signature:continue
        if a.signature in seen or b.signature in seen:continue
        xa,xb=features(a),features(b)
        if torch.equal(xa[...,:28] if encoding=='coefficient' else xa,
                       xb[...,:28] if encoding=='coefficient' else xb):continue
        seen.update((a.signature,b.signature));result.append((a,b))
        if len(result)==count:return result
    raise RuntimeError('structurally distinct pair generation exhausted')


class FiberAdapter(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden'];r=c['adapter_rank']
        self.read=nn.Parameter(torch.randn(h,r)*.05);self.write=nn.Parameter(torch.randn(r,h)*.05)
        self.router=nn.Linear(c['sae_width'],r,bias=False);nn.init.normal_(self.router.weight,std=.05)
        self.gate=nn.Parameter(torch.full((c['visits'],r),.6));self.scale=.15/math.sqrt(r)
        self.register_buffer('q',torch.zeros(h,c['head_dim']))
    def forward(self,h,code,visit):
        gate=torch.tanh(self.router(code))*torch.tanh(self.gate[visit])
        raw=((h@self.read)*gate)@self.write*self.scale
        return raw-(raw@self.q)@self.q.T


class Model(nn.Module):
    def __init__(self,c,encoding):
        super().__init__();self.config=c;self.encoding=encoding;self.hidden=c['hidden'];self.visits=c['visits']
        shared_rng=torch.get_rng_state()
        width=28 if encoding=='coefficient' else 28+23*23
        self.degree_weight=nn.Parameter(torch.randn(13,width,self.hidden)/math.sqrt(width))
        self.degree_bias=nn.Parameter(torch.zeros(13,self.hidden))
        # Common modules use the same initial RNG stream in both encoding arms.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(shared_rng)
            self.positions=nn.Parameter(torch.randn(27,self.hidden)*.02)
            self.core=TiedBlock(self.hidden,c['heads'])
            self.head=nn.Sequential(nn.Linear(self.hidden,c['head_dim'],bias=False),nn.Linear(c['head_dim'],64))
            self.sae=SharedTopKSAE(self.hidden,c['sae_width'],c['sae_k'])
            self.adapter=FiberAdapter(c)
        self.locked=False;self.tracker=None
    def lens(self,h):
        if self.tracker:self.tracker.work['readout_examples']+=len(h)
        return self.head(h)
    def forward(self,data,policy='base',capture=None,hook=None,initial=False):
        if data.shape[1:]!=(2,13,51):raise ValueError('public input must be [B,2,13,51]')
        if self.tracker:self.tracker.forward(len(data),27,self.visits)
        basic=data[...,:28]
        if self.encoding=='polynomial_gram':
            vector=data[...,28:];gram=(vector[...,None]*vector[...,None,:]).flatten(-2)
            basic=torch.cat((basic,gram),-1)
            if self.tracker:self.tracker.work['public_gram_multiplications']+=len(data)*26*529
        h=torch.einsum('bdnf,nfh->bdnh',basic,self.degree_weight)+self.degree_bias
        h=h.reshape(len(data),26,self.hidden);h=torch.cat((h.mean(1,keepdim=True),h),1)+self.positions
        initial_logits=self.lens(h[:,0]) if initial else None
        states=[];writes=[];energy=h.new_zeros(())
        for visit in range(self.visits):
            h=h+self.config['residual_scale']*(self.core(h)-h)
            delta=torch.zeros_like(h)
            if self.locked:
                code=self.sae.encode(F.layer_norm(h,(self.hidden,)))
                if policy in ('base','zero'):code=torch.zeros_like(code)
                elif policy!='adapter':raise ValueError('invalid policy')
                delta=self.adapter(h,code,visit);h=h+delta;energy+=delta.square().mean()
            if hook is not None:h=hook(visit,h)
            if capture:
                states.append((h if capture=='all' else h[:,0]).detach().cpu())
                writes.append(delta[:,0].detach().cpu())
        return {'logits':self.lens(h[:,0]),'initial_logits':initial_logits,
                'states':states,'writes':writes,'energy':energy}
    @torch.no_grad()
    def lock(self):
        weight=self.head[0].weight.detach().cpu().double()
        assert torch.linalg.matrix_rank(weight)==self.config['head_dim']
        q=torch.linalg.qr(weight.T).Q.float().to(self.adapter.q.device);self.adapter.q.copy_(q)
        self.locked=True
        for p in self.parameters():p.requires_grad_(False)


def set_loss(logits,mask):
    if not mask.any(-1).all():raise ValueError('empty correctness label set')
    return (torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),-1)).mean()


def grade_detail(oracle,task,pred):
    graph=PROGRAMS[int(pred)]
    prefix=oracle.check_candidate(task.oracle_case,graph)
    correct=bool(oracle.grade(task.oracle_case,{'graph':graph})['future_correct']) if prefix['accepted'] else False
    return {'correct':int(correct),'prefix_accepted':int(prefix['accepted']),
            'false_future_acceptance':int(prefix['accepted'] and not correct),'first_faults':prefix['first_faults']}


def grade_one(oracle,task,pred):return bool(grade_detail(oracle,task,pred)['correct'])


@torch.no_grad()
def score(model,cases,oracle,tracker,policy='base',initial=False):
    device=next(model.parameters()).device;rows=[];initial_rows=[]
    for start in range(0,len(cases),model.config['batch']):
        batch=cases[start:start+model.config['batch']]
        x=torch.stack([features(t) for t in batch]).to(device)
        out=model(x,policy=policy,initial=initial);pred=out['logits'].argmax(-1).cpu().tolist()
        # All grading is strictly after the complete neural prediction/argmax.
        tracker.event('prediction_selected',ids=[t.case_id for t in batch],predictions=pred)
        for task,p in zip(batch,pred):
            rows.append({'id':task.case_id,'signature':task.signature,'family':task.family,
                         'prediction':p,**grade_detail(oracle,task,p)})
        if initial:
            p0=out['initial_logits'].argmax(-1).cpu().tolist()
            initial_rows.extend(int(grade_one(oracle,t,p)) for t,p in zip(batch,p0))
    return {'n':len(rows),'correct':sum(v['correct'] for v in rows),
            'accuracy':sum(v['correct'] for v in rows)/len(rows),
            'prefix_acceptances':sum(v['prefix_accepted'] for v in rows),
            'false_future_acceptances':sum(v['false_future_acceptance'] for v in rows),
            'initial_cls_correct':sum(initial_rows) if initial else None,
            'inference_verifier_calls':0,'predictions_per_case':1,'records':rows}


def save_model(path,model,meta):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'meta':meta},path)
    return {'file':path.name,'file_sha256':sha(path),'model_sha256':digest(model),
            'sae_sha256':digest(model.sae),'encoding':model.encoding,**meta}


def load_model(path,c,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=Model(c,saved['meta']['encoding']);model.load_state_dict(saved['state'],strict=True)
    model.locked=saved['meta'].get('locked',False)
    for p in model.parameters():p.requires_grad_(False)
    return model.to(device).eval()
