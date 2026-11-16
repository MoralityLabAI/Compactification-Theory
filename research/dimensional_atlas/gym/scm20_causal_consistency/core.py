"""Differentiable native-write interchange and durable local accounting."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from ..scm19_clean_topone.core import (Model,Journal as OldJournal,OracleAudit,features,groups,correct_mask,
    make_pool,make_pairs,FAMILIES,OOD,PROGRAMS,set_loss,grade_detail,digest,sha,write_new,load_model,
    save_model,device_setup,release)
from ..scm19_clean_topone.campaign import recipe,restore,old_signatures

HERE=Path(__file__).resolve().parent


def named(task):
    name=task.case_id.replace('scm19/','scm20/',1)
    return replace(task,case_id=name,oracle_case=replace(task.oracle_case,identifier=name))


def pool(*args,**kwargs):return [named(t) for t in make_pool(*args,**kwargs)]
def pair_pool(*args,**kwargs):return [(named(a),named(b)) for a,b in make_pairs(*args,**kwargs)]


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.new.json')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
    os.replace(tmp,path)


class Journal(OldJournal):
    def event(self,event_name,**values):
        # Payload keys such as kind never collide with the event name.
        self.file.write(json.dumps({'event':event_name,'stage':self.stage,
            'elapsed':time.perf_counter()-self.started,**values},sort_keys=True,allow_nan=False)+'\n')
    def checkpoint(self,audit):
        atomic_json(Path(self.file.name).with_name('latest_state.json'),
            {'resources':self.report(),'oracle':audit.report(),'journal_bytes':self.file.tell()})


def row_loss(logits,mask):
    if not mask.any(-1).all():raise ValueError('empty correctness label')
    return torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),-1)


def run(model,data,tracker=None,patch=None,capture=False):
    """Same frozen SCM19 algebra, with live native writes for causal gradients.

    patch = (visit, donor_native_write, unused) replaces recipient native write
    on CLS with donor write. Other patches use (visit, delta).
    """
    if tracker:tracker.forward(len(data),27,model.visits)
    assert model.encoding=='coefficient' and model.locked
    h=torch.einsum('bdnf,nfh->bdnh',data[...,:28],model.degree_weight)+model.degree_bias
    h=h.reshape(len(data),26,model.hidden);h=torch.cat((h.mean(1,keepdim=True),h),1)+model.positions
    states=[];writes=[];energy=h.new_zeros(())
    for visit in range(model.visits):
        h=h+model.config['residual_scale']*(model.core(h)-h)
        code=model.sae.encode(F.layer_norm(h,(model.hidden,)))
        delta=model.adapter(h,code,visit);h=h+delta;energy+=delta.square().mean()
        writes.append(delta[:,0])
        if patch is not None and visit==patch[0]:
            d=patch[1]-delta[:,0] if len(patch)==3 else patch[1]
            h=h.clone();h[:,0]=h[:,0]+d
        if capture:states.append(h[:,0])
    if tracker:tracker.work['readout_examples']+=len(data)
    return {'logits':model.head(h[:,0]),'writes':writes,'states':states,'energy':energy}


def losses(a,b,patched,mask_a,mask_b,teacher_a,teacher_b,eligible,c):
    clean=(row_loss(a['logits'],mask_a)+row_loss(b['logits'],mask_b)).mean()/2
    kl=(F.kl_div(F.log_softmax(a['logits'],-1),F.softmax(teacher_a,-1),reduction='batchmean')+
        F.kl_div(F.log_softmax(b['logits'],-1),F.softmax(teacher_b,-1),reduction='batchmean'))/2
    causal=(row_loss(patched['logits'],mask_b)*eligible).sum()/eligible.sum().clamp_min(1)
    energy=(a['energy']+b['energy'])/2
    total=clean+c['clean_kl_weight']*kl+c['causal_weight']*causal+c['energy_weight']*energy
    return total,{'clean_nll':clean,'clean_kl':kl,'causal_nll':causal,'write_energy':energy,'total':total}


@torch.no_grad()
def score(model,cases,oracle,tracker,capture=False):
    rows=[];traces=[];device=next(model.parameters()).device
    for start in range(0,len(cases),model.config['batch']):
        batch=cases[start:start+model.config['batch']]
        value=run(model,torch.stack([features(t) for t in batch]).to(device),tracker,capture=capture)
        predictions=value['logits'].argmax(-1).cpu().tolist()
        tracker.event('prediction_selected',ids=[t.case_id for t in batch],predictions=predictions)
        for j,(task,pred) in enumerate(zip(batch,predictions)):
            rows.append({'id':task.case_id,'signature':task.signature,'family':task.family,'prediction':pred,
                **grade_detail(oracle,task,pred)})
            if capture:traces.append({'states':[h[j].cpu() for h in value['states']],
                'writes':[h[j].cpu() for h in value['writes']]})
    return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
        'prefix_acceptances':sum(r['prefix_accepted'] for r in rows),'false_future_acceptances':sum(r['false_future_acceptance'] for r in rows),
        'inference_verifier_calls':0,'predictions_per_case':1,'records':rows},traces


@torch.no_grad()
def pair_trace(model,pairs,oracle,tracker):
    # Stream roles separately, retaining only small CLS summaries on CPU.
    a,ta=score(model,[p[0] for p in pairs],oracle,tracker,True)
    b,tb=score(model,[p[1] for p in pairs],oracle,tracker,True)
    eligible=[bool(x['correct'] and y['correct']) for x,y in zip(a['records'],b['records'])]
    return {'recipient':a,'donor':b,'eligible':eligible,'trace_a':ta,'trace_b':tb}


def random_frame(model,seed):
    q=model.adapter.q.detach().cpu();rng=torch.Generator().manual_seed(seed+206000)
    z=torch.randn(model.hidden,4,generator=rng);z-=q@(q.T@z)
    return torch.linalg.qr(z.double()).Q.float()


@torch.no_grad()
def assay(model,pairs,trace,indices,oracle,tracker,methods,visits,frame):
    result=[];device=next(model.parameters()).device
    q=model.adapter.q
    for visit in visits:
        for method in methods:
            rows=[];norms=[]
            for offset in range(0,len(indices),model.config['batch']):
                ids=indices[offset:offset+model.config['batch']]
                d=torch.stack([trace['trace_b'][i]['states'][visit]-trace['trace_a'][i]['states'][visit] for i in ids]).to(device)
                wd=torch.stack([trace['trace_b'][i]['writes'][visit]-trace['trace_a'][i]['writes'][visit] for i in ids]).to(device)
                if method=='native_vpd_write':delta=wd
                elif method=='sham':delta=torch.zeros_like(d)
                elif method=='full_cls':delta=d
                elif method=='head_readable':delta=(d@q)@q.T
                elif method=='head_fiber':delta=d-(d@q)@q.T
                else:
                    basis=frame['random'] if method=='random_fiber4' else frame[method][visit]
                    basis=basis.to(device);delta=(d@basis)@basis.T
                    if method=='random_fiber4':
                        length=delta.norm(dim=-1,keepdim=True);fallback=basis[:,0][None].expand(len(ids),-1)
                        delta=torch.where(length>1e-12,delta/length.clamp_min(1e-12),fallback)*wd.norm(dim=-1,keepdim=True)
                x=torch.stack([features(pairs[i][0]) for i in ids]).to(device)
                pred=run(model,x,tracker,patch=(visit,delta))['logits'].argmax(-1).cpu().tolist()
                tracker.event('patched_prediction_selected',pair_ids=[pairs[i][0].case_id for i in ids],
                    predictions=pred,method=method,visit=visit)
                for i,p,norm in zip(ids,pred,delta.norm(dim=-1).cpu().tolist()):
                    correct=grade_detail(oracle,pairs[i][1],p)['correct'];norms.append(norm)
                    rows.append({'pair_id':pairs[i][0].case_id,'family':pairs[i][0].family,'prediction':p,'correct':correct})
            n=len(rows);k=sum(v['correct'] for v in rows)
            result.append({'method':method,'visit':visit,'n':n,'correct':k,'fidelity':k/n if n else None,'patch_norms':norms,'records':rows})
            tracker.event('mediation_cell_completed',method=method,visit=visit,n=n,correct=k)
    return result
