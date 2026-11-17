"""Durable, sequential SCM-19 acquisition, freeze and fresh semantic confirmation."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import (HERE,FAMILIES,OOD,PROGRAMS,Model,Journal,OracleAudit,features,groups,
    correct_mask,make_pool,make_pairs,set_loss,grade_one,score,save_model,load_model,
    canonical,digest,sha,write_new,device_setup,release)


def source_hashes():
    paths=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PROTOCOL.md',HERE.parent/'model.py',HERE.parent/'tasks.py',HERE.parent/'scm18_rtx3050/core.py']
    return {p.relative_to(HERE.parents[3]).as_posix():sha(p) for p in paths}


def recipe(task):
    return {'seed':int(task.case_id.rsplit('/',1)[-1]),'id':task.case_id,'family':task.family,
            'topology':task.topology,'graph':list(task.source_graph),'signature':task.signature}


def restore(oracle,recipes):
    from ..tasks import make_evalmath_case
    result=[]
    for row in recipes:
        task=make_evalmath_case(oracle,row['seed'],case_id=row['id'],family=row['family'],
                                topology=row['topology'],graph=tuple(row['graph']))
        assert task.signature==row['signature'];result.append(task)
    return result


def old_signatures():
    old=HERE.parent/'scm18_rtx3050/results'
    seen=set(json.loads((old/'freeze.json').read_text())['all_pre_final_signatures'])
    raw=json.loads((old/'final.json').read_text())
    for trial in raw['results']:
        for policy in trial['policies'].values():
            for dataset in policy.values():seen.update(v['signature'] for v in dataset['records'])
    return seen


def prepare(c,private,out,oracle,tracker):
    cache=private/'data.pt';receipt=out/'data.json'
    if receipt.exists():
        r=json.loads(receipt.read_text());assert sha(cache)==r['cache_sha256']
        return torch.load(cache,weights_only=True),r
    seen=old_signatures();tracker.stage='training_data'
    train=make_pool(oracle,c['train_count'],19100000,'train',FAMILIES,seen)
    grouping,group_meta=groups(oracle)
    x=[];mask=[]
    for i,task in enumerate(train):
        x.append(features(task));mask.append(correct_mask(task,oracle,grouping))
        tracker.event('training_case_labeled',index=i,oracle_counts=dict(oracle._scm19_audit.counts))
        if (i+1)%1024==0:print(json.dumps({'training_data_ready':i+1}),flush=True)
    data={'x':torch.stack(x),'mask':torch.stack(mask),'train_signatures':[t.signature for t in train]}
    for role,count,start,families in [('dev_id',c['dev_id_count'],19300000,FAMILIES),
            ('dev_ood',c['dev_ood_count'],19400000,OOD),('calibration',c['calibration_count'],19500000,FAMILIES)]:
        data[role]=[recipe(t) for t in make_pool(oracle,count,start,role,families,seen)]
    data['seen']=sorted(seen);data['group_meta']=group_meta
    torch.save(data,cache)
    r={'config_sha256':sha(HERE/'config.json'),'cache_sha256':sha(cache),'training_n':len(train),
       'public_signature_count':len(seen),'grouping':group_meta,
       'mean_verified_positive_programs':float(data['mask'].float().sum(-1).mean()),
       'split_signatures':{k:[v['signature'] for v in data[k]] for k in ('dev_id','dev_ood','calibration')},
       'training_signatures':data['train_signatures'],'oracle':oracle._scm19_audit.report()}
    write_new(receipt,r);del train,x,mask;gc.collect();return data,r


def brief(value):return {k:v for k,v in value.items() if k!='records'}


def atomic_save(path,payload):
    temporary=path.with_suffix('.new.pt');torch.save(payload,temporary);os.replace(temporary,path)


def base_train(c,encoding,seed,total,private,out,data,dev,oracle,tracker,device,extend_from=None):
    label=f'{encoding}_s{seed}_steps{total}';done=out/(label+'.json')
    model_file=private/(label+'.pt');resume=private/(label+'.resume.pt')
    if done.exists():
        result=json.loads(done.read_text());assert sha(model_file)==result['checkpoint']['file_sha256']
        return result
    torch.manual_seed(seed);model=Model(c,encoding).to(device);model.tracker=tracker
    params=[]
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith(('sae.','adapter.')))
        if p.requires_grad:params.append(p)
    opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'])
    rng=torch.Generator().manual_seed(seed+1);step=0;curve=[];losses=[]
    checkpoint=resume if resume.exists() else extend_from
    if checkpoint:
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
        model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng']);step=saved['step'];curve=saved['curve'];losses=saved['losses']
    initial=digest(model);before=Counter(tracker.work);started=time.perf_counter();model.train()
    for step in range(step+1,total+1):
        tracker.stage=label+'/train'
        ids=torch.randint(len(data['x']),(c['batch'],),generator=rng)
        xb=data['x'][ids].to(device);yb=data['mask'][ids].to(device)
        logits=model(xb)['logits'];loss=set_loss(logits,yb)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite acquisition loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,2.);opt.step()
        tracker.work['base_optimizer_steps']+=1
        tracker.event('optimizer_completed',step=step,loss=float(loss.detach()),**tracker.memory())
        if step%100==0:losses.append({'step':step,'correct_set_nll':float(loss.detach())})
        if step%c['dev_every']==0 or step==total:
            model.eval();entry={'step':step}
            for name,cases in dev.items():
                tracker.stage=label+'/'+name
                entry[name]=brief(score(model,cases,oracle,tracker))
            curve.append(entry);model.train()
            print(json.dumps({'model':label,'step':step,'loss':float(loss.detach()),'dev_id':entry['dev_id']['accuracy'],'dev_ood':entry['dev_ood']['accuracy']}),flush=True)
        if step%c['checkpoint_every']==0 or step==total:
            atomic_save(resume,{'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':opt.state_dict(),'rng':rng.get_state(),'step':step,'curve':curve,'losses':losses})
    model.lock();model.eval()
    ck=save_model(model_file,model,{'encoding':encoding,'seed':seed,'steps':total,'locked':True})
    result={'encoding':encoding,'seed':seed,'steps':total,'checkpoint':ck,'initial_sha256':initial,
            'parameters':sum(p.numel() for p in model.parameters()),'base_parameters':sum(p.numel() for p in params),
            'curve':curve,'losses':losses,'work':dict(tracker.work-before),'seconds':time.perf_counter()-started}
    write_new(done,result);del model,opt,params;release();return result


def fit_components(c,base,private,out,data,dev,oracle,tracker,device,fit_adapter):
    seed=base['seed'];name=f'frozen_s{seed}';path=out/(name+'.json')
    if path.exists():
        value=json.loads(path.read_text());assert sha(private/value['checkpoint']['file'])==value['checkpoint']['file_sha256'];return value
    model=load_model(private/base['checkpoint']['file'],c,device);model.tracker=tracker
    frozen=digest(model,('degree_','positions','core.','head.'))
    before=Counter(tracker.work);tracker.stage=name+'/activation_cache'
    chunks=[]
    with torch.no_grad():
        for start in range(0,min(256,len(data['x'])),c['batch']):
            output=model(data['x'][start:start+c['batch']].to(device),capture='all')
            chunks.extend(F.layer_norm(h,(c['hidden'],)).reshape(-1,c['hidden']) for h in output['states'])
    target=torch.cat(chunks);del chunks,output
    for p in model.sae.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.sae.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(seed+2)
    sae_losses=[]
    for step in range(c['sae_steps']):
        tracker.stage=name+'/sae';ids=torch.randint(len(target),(c['batch'],),generator=rng);x=target[ids].to(device)
        recon,_=model.sae(x);loss=F.mse_loss(recon,x)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite SAE loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();tracker.work['sae_optimizer_steps']+=1
        tracker.event('sae_update',step=step+1,loss=float(loss.detach()),**tracker.memory())
        if step%50==0 or step+1==c['sae_steps']:sae_losses.append({'step':step+1,'reconstruction':float(loss.detach())})
    for p in model.sae.parameters():p.requires_grad_(False)
    dictionary=digest(model.sae);del target,optimizer;release()
    adapter_losses=[]
    if fit_adapter:
        for p in model.adapter.parameters():p.requires_grad_(True)
        optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=c['adapter_lr'])
        rng=torch.Generator().manual_seed(seed+3)
        for step in range(c['adapter_steps']):
            tracker.stage=name+'/adapter';ids=torch.randint(len(data['x']),(c['batch'],),generator=rng)
            xb=data['x'][ids].to(device);mask=data['mask'][ids].to(device)
            with torch.no_grad():teacher=model(xb,policy='base')['logits']
            output=model(xb,policy='adapter');math_loss=set_loss(output['logits'],mask)
            distill=F.kl_div(F.log_softmax(output['logits'],-1),F.softmax(teacher,-1),reduction='batchmean')
            loss=math_loss+c['adapter_distillation']*distill+c['adapter_energy']*output['energy']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite adapter loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),2.);optimizer.step()
            tracker.work['adapter_optimizer_steps']+=1
            tracker.event('adapter_update',step=step+1,math=float(math_loss.detach()),distillation=float(distill.detach()),energy=float(output['energy'].detach()),**tracker.memory())
            if step%50==0 or step+1==c['adapter_steps']:adapter_losses.append({'step':step+1,'math':float(math_loss.detach()),'distillation':float(distill.detach()),'energy':float(output['energy'].detach()),'total':float(loss.detach())})
        del optimizer
    for p in model.parameters():p.requires_grad_(False)
    assert digest(model,('degree_','positions','core.','head.'))==frozen
    assert digest(model.sae)==dictionary
    tracker.stage=name+'/development'
    scores={policy:{kind:brief(score(model,cases,oracle,tracker,policy=policy)) for kind,cases in dev.items()}
            for policy in ('base','adapter') if policy=='base' or fit_adapter}
    ck=save_model(private/(name+'.pt'),model,{'encoding':model.encoding,'seed':seed,'steps':base['steps'],'locked':True,'adapter_fitted':fit_adapter})
    result={'seed':seed,'checkpoint':ck,'frozen_base_sha256':frozen,'sae_losses':sae_losses,
            'adapter_losses':adapter_losses,'adapter_fitted':fit_adapter,'development':scores,
            'work':dict(tracker.work-before),'base_unchanged':True,'dictionary_unchanged_during_adapter':True}
    write_new(path,result);del model;release();return result


@torch.no_grad()
def calibrate(c,trial,private,calib,pairs,tracker,device):
    receipt=HERE/'results'/f'frame_s{trial["seed"]}.json'
    if receipt.exists():
        existing=json.loads(receipt.read_text());assert sha(private/existing['file'])==existing['sha256'];return existing
    model=load_model(private/trial['checkpoint']['file'],c,device);model.tracker=tracker
    frames={};tracker.stage=f'calibration_s{trial["seed"]}'
    policies=('base','adapter') if trial['adapter_fitted'] else ('base',)
    for policy in policies:
        states=[[] for _ in range(c['visits'])];diff=[[] for _ in range(c['visits'])];usage=torch.zeros(c['sae_width'])
        for start in range(0,len(calib),c['batch']):
            out=model(torch.stack([features(t) for t in calib[start:start+c['batch']]]).to(device),policy=policy,capture='cls')
            for v,h in enumerate(out['states']):
                states[v].append(h)
                usage+=model.sae.encode(F.layer_norm(h.to(device),(c['hidden'],))).sum(0).cpu()
        for start in range(0,len(pairs),c['batch']):
            batch=pairs[start:start+c['batch']]
            a=model(torch.stack([features(t[0]) for t in batch]).to(device),policy=policy,capture='cls')
            b=model(torch.stack([features(t[1]) for t in batch]).to(device),policy=policy,capture='cls')
            for v in range(c['visits']):diff[v].append(b['states'][v]-a['states'][v])
        pca=[];semantic=[]
        for v in range(c['visits']):
            h=torch.cat(states[v]).double();_,_,vh=torch.linalg.svd(h-h.mean(0),full_matrices=False)
            pca.append(vh[:4].T.float())
            _,_,vh=torch.linalg.svd(torch.cat(diff[v]).double(),full_matrices=False);semantic.append(vh[:4].T.float())
        order=usage.argsort(descending=True)[:4]
        sae=torch.linalg.qr(model.sae.decoder.weight.detach().cpu()[:,order].double()).Q.float()
        q=model.adapter.q.detach().cpu();native=model.adapter.write.detach().cpu().T;native-=q@(q.T@native)
        u,s,_=torch.linalg.svd(native.double(),full_matrices=False);native=u[:,s>s.max()*1e-6].float()
        rng=torch.Generator().manual_seed(trial['seed']+8000);random_basis=torch.randn(c['hidden'],4,generator=rng)
        random_basis-=q@(q.T@random_basis);random_basis=torch.linalg.qr(random_basis.double()).Q.float()
        frames[policy]={'pca':pca,'semantic':semantic,'sae':sae,'sae_features':order,'native':native,'random':random_basis,'q':q}
    file=private/f'frames_s{trial["seed"]}.pt';torch.save(frames,file)
    result={'seed':trial['seed'],'file':file.name,'sha256':sha(file),'case_signatures':[t.signature for t in calib],
            'pair_signatures':[[a.signature,b.signature] for a,b in pairs]}
    write_new(receipt,result)
    del model;release();return result


@torch.no_grad()
def historical_score(c,seed,cases,private_root,oracle,tracker,device):
    from ..scm18_rtx3050.core import load_checkpoint
    from ..tasks import encode
    path=private_root/f's{seed}_final_r4_joint.pt'
    old_config=json.loads((HERE.parent/'scm18_rtx3050/config.json').read_text())
    model,_=load_checkpoint(path,old_config,device);model.meter=None;rows=[]
    before=Counter(tracker.work)
    for start in range(0,len(cases),c['batch']):
        batch=cases[start:start+c['batch']];x=torch.stack([encode(t) for t in batch]).to(device)
        tracker.forward(len(batch),4,16);pred=model(x,policy='tied').logits.argmax(-1).cpu().tolist()
        tracker.work['readout_examples']+=len(batch)
        for task,p in zip(batch,pred):rows.append({'id':task.case_id,'signature':task.signature,'family':task.family,'prediction':p,'correct':int(grade_one(oracle,task,p))})
    result={'seed':seed,'n':len(rows),'correct':sum(v['correct'] for v in rows),'accuracy':sum(v['correct'] for v in rows)/len(rows),
            'inference_verifier_calls':0,'records':rows,'work':dict(tracker.work-before)}
    del model;release();return result


@torch.no_grad()
def mediation(c,trial,frame_info,private,sets,oracle,tracker,device,assay_allowed=True):
    model=load_model(private/trial['checkpoint']['file'],c,device);model.tracker=tracker
    assert sha(private/frame_info['file'])==frame_info['sha256']
    frames=torch.load(private/frame_info['file'],weights_only=True);results={}
    for policy,frame_cpu in frames.items():
        frame={k:([v.to(device) for v in val] if isinstance(val,list) else val.to(device)) for k,val in frame_cpu.items()}
        results[policy]={}
        for kind,pairs in sets.items():
            tracker.stage=f'mediation_s{trial["seed"]}/{policy}/{kind}'
            eligible=[];counts=Counter()
            for start in range(0,len(pairs),c['batch']):
                batch=pairs[start:start+c['batch']]
                a=model(torch.stack([features(t[0]) for t in batch]).to(device),policy=policy,capture='cls')
                b=model(torch.stack([features(t[1]) for t in batch]).to(device),policy=policy,capture='cls')
                pa=a['logits'].argmax(-1).cpu().tolist();pb=b['logits'].argmax(-1).cpu().tolist()
                for j,(ta,tb) in enumerate(batch):
                    ca=grade_one(oracle,ta,pa[j]);cb=grade_one(oracle,tb,pb[j]);counts['clean_recipient_correct']+=ca;counts['clean_donor_correct']+=cb
                    if ca and cb:
                        counts['eligible']+=1
                        if len(eligible)<c['eligible_cap_per_stratum']:
                            eligible.append((ta,tb,[h[j] for h in a['states']],[h[j] for h in b['states']],
                                             [h[j] for h in a['writes']],[h[j] for h in b['writes']]))
            records=[];n=len(eligible) if assay_allowed else 0
            for visit in range(c['visits']):
                for method in c['mediation_methods']:
                    correct=0;norms=[];outcomes=[];logit_effects=[]
                    for start in range(0,n,c['batch']):
                        batch=eligible[start:start+c['batch']]
                        difference=torch.stack([b[3][visit]-b[2][visit] for b in batch]).to(device)
                        write_delta=torch.stack([b[5][visit]-b[4][visit] for b in batch]).to(device)
                        if method=='sham':delta=torch.zeros_like(difference)
                        elif method=='full_cls':delta=difference
                        elif method=='head_readable':delta=(difference@frame['q'])@frame['q'].T
                        elif method=='head_fiber':delta=difference-(difference@frame['q'])@frame['q'].T
                        elif method=='native_vpd_write':delta=write_delta
                        else:
                            basis={'pca4':frame['pca'][visit],'semantic_diff4':frame['semantic'][visit],
                                   'sae_group4':frame['sae'],'native_write4':frame['native'],
                                   'random_fiber4':frame['random']}[method]
                            delta=(difference@basis)@basis.T
                            if method=='random_fiber4':
                                den=delta.norm(dim=-1,keepdim=True);fallback=basis[:,0][None].expand(len(batch),-1)
                                direction=torch.where(den>1e-12,delta/den.clamp_min(1e-12),fallback)
                                delta=direction*write_delta.norm(dim=-1,keepdim=True)
                        def hook(at,h):
                            if at!=visit:return h
                            h=h.clone();h[:,0]+=delta;return h
                        x=torch.stack([features(b[0]) for b in batch]).to(device)
                        output=model(x,policy=policy,hook=hook)['logits'];pred=output.argmax(-1).cpu().tolist()
                        for b,p,norm in zip(batch,pred,delta.norm(dim=-1).cpu().tolist()):
                            success=grade_one(oracle,b[1],p);correct+=success;norms.append(norm)
                            outcomes.append({'pair_id':b[0].case_id,'family':b[0].family,'prediction':p,'counterfactual_correct':int(success)})
                    records.append({'visit':visit,'method':method,'valid_counterfactual_tests':n,
                        'correct_counterfactual_outcomes':correct,'causal_fidelity':correct/n if n else None,
                        'patch_norms':norms,'outcomes':outcomes,'intervention_token':'CLS_only'})
                    tracker.event('mediation_cell_completed',seed=trial['seed'],policy=policy,kind=kind,visit=visit,method=method,n=n,correct=correct)
            results[policy][kind]={'attempted_pairs':len(pairs),'eligible_pairs_total':counts['eligible'],
                'assayed_pairs':n,'eligibility':dict(counts),'records':records,'acquisition_gate_passed':assay_allowed}
    del model;release();return {'seed':trial['seed'],'policies':results}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True);parser.add_argument('--old-private',type=Path,required=True)
    args=parser.parse_args();c=json.loads((HERE/'config.json').read_text());out=HERE/'results'
    args.private.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
    registered=json.loads((out/'registration.json').read_text())
    assert registered['config_sha256']==sha(HERE/'config.json') and registered['source_sha256']==source_hashes()
    if (out/'completed.json').exists():
        result=json.loads((out/'completed.json').read_text());assert result['freeze_sha256']==sha(out/'freeze.json')
        print('SCM19_ALREADY_COMPLETE');return
    device=device_setup('cuda',c);tracker=Journal(args.private/'work.jsonl',c)
    audit=OracleAudit(args.oracle,tracker);oracle=audit.module;oracle._scm19_audit=audit
    started=time.perf_counter();data,data_receipt=prepare(c,args.private,out,oracle,tracker)
    dev={name:restore(oracle,data[name]) for name in ('dev_id','dev_ood')}
    if not (out/'historical_development.json').exists():
        tracker.stage='historical_development'
        write_new(out/'historical_development.json',{k:historical_score(c,181,v,args.old_private,oracle,tracker,device) for k,v in dev.items()})
    pilots=[base_train(c,encoding,c['seeds'][0],c['base_steps'],args.private,out,data,dev,oracle,tracker,device) for encoding in c['encodings']]
    def gate(value):return value['curve'][-1]['dev_id']['accuracy']>=c['dev_id_gate'] and value['curve'][-1]['dev_ood']['accuracy']>=c['dev_ood_gate']
    passing=[v for v in pilots if gate(v)]
    chosen=sorted(passing or pilots,key=lambda v:(-v['curve'][-1]['dev_id']['accuracy'],-v['curve'][-1]['dev_ood']['accuracy'],c['encodings'].index(v['encoding'])))[0]
    total=c['base_steps']
    if not any(gate(v) for v in pilots):
        previous=args.private/f'{chosen["encoding"]}_s{c["seeds"][0]}_steps{total}.resume.pt';total=c['extension_steps']
        chosen=base_train(c,chosen['encoding'],c['seeds'][0],total,args.private,out,data,dev,oracle,tracker,device,extend_from=previous)
    selection={'encoding':chosen['encoding'],'common_step_budget':total,'clean_gate_passed':gate(chosen),
        'selected_by':'fixed endpoint development ID, then OOD; no fresh held-out access',
        'endpoint':chosen['curve'][-1],'source_sha256':source_hashes()}
    if not (out/'selection.json').exists():write_new(out/'selection.json',selection)
    else:assert json.loads((out/'selection.json').read_text())==selection
    bases=[chosen]+[base_train(c,chosen['encoding'],seed,total,args.private,out,data,dev,oracle,tracker,device) for seed in c['seeds'][1:]]
    trials=[fit_components(c,b,args.private,out,data,dev,oracle,tracker,device,selection['clean_gate_passed']) for b in bases]
    seen=set(data['seen']);calib=restore(oracle,data['calibration'])
    calibration_pairs=make_pairs(oracle,c['calibration_pair_count'],19550000,'calibration_pair',FAMILIES,seen,selection['encoding'])
    frames=[calibrate(c,t,args.private,calib,calibration_pairs,tracker,device) for t in trials]
    frozen={'schema':'scm19.frozen.v1','registration_sha256':sha(out/'registration.json'),'selection_sha256':sha(out/'selection.json'),
        'source_sha256':source_hashes(),'oracle_sha256':sha(args.oracle),'data_sha256':sha(out/'data.json'),
        'trials':trials,'frames':frames,'pre_final_signatures':sorted(seen),'official_roster_calls_before_freeze':0}
    if not (out/'freeze.json').exists():write_new(out/'freeze.json',frozen)
    else:assert json.loads((out/'freeze.json').read_text())==frozen
    assert source_hashes()==registered['source_sha256']
    for trial in trials:assert sha(args.private/trial['checkpoint']['file'])==trial['checkpoint']['file_sha256']
    tracker.event('all_weights_and_directions_frozen',freeze_sha256=sha(out/'freeze.json'))
    # Fresh evaluation data become reachable only after the complete freeze.
    final_sets={'id':make_pool(oracle,c['clean_id_count'],19600000,'clean_final_id',FAMILIES,seen),
                'ood':make_pool(oracle,c['clean_ood_count'],19700000,'clean_final_ood',OOD,seen)}
    pair_sets={'id':make_pairs(oracle,c['mediation_id_pairs'],19800000,'mediation_id',FAMILIES,seen,selection['encoding']),
               'ood':make_pairs(oracle,c['mediation_ood_pairs'],19900000,'mediation_ood',OOD,seen,selection['encoding'])}
    gold={'clean':{k:[recipe(t) for t in cases] for k,cases in final_sets.items()},
          'mediation':{k:[{'recipient':recipe(a),'donor_graph':list(b.source_graph),'recipient_future':list(a.hidden_target),'donor_future':list(b.hidden_target)} for a,b in cases] for k,cases in pair_sets.items()}}
    if not (args.private/'fresh.gold.private.json').exists():write_new(args.private/'fresh.gold.private.json',gold)
    if not (out/'fresh.public.json').exists():write_new(out/'fresh.public.json',{'clean':{k:[t.public for t in cases] for k,cases in final_sets.items()},'mediation':{k:[[a.public,b.public] for a,b in cases] for k,cases in pair_sets.items()}})
    clean=[]
    for index,trial in enumerate(trials):
        name=f'clean_s{trial["seed"]}.json';path=out/name
        if path.exists():clean.append(json.loads(path.read_text()));continue
        model=load_model(args.private/trial['checkpoint']['file'],c,device);model.tracker=tracker
        tracker.stage=f'clean_final_s{trial["seed"]}'
        policies=('base','adapter') if trial['adapter_fitted'] else ('base',)
        row={'seed':trial['seed'],'scores':{p:{k:score(model,cases,oracle,tracker,policy=p,initial=True) for k,cases in final_sets.items()} for p in policies}}
        assert digest(model)==trial['checkpoint']['model_sha256'];del model;release()
        row['historical_reference']={k:historical_score(c,[181,1181,2181][index],cases,args.old_private,oracle,tracker,device) for k,cases in final_sets.items()}
        write_new(path,row);clean.append(row)
        print(json.dumps({'clean_final_seed':trial['seed'],'scores':{p:{k:v['accuracy'] for k,v in sets.items()} for p,sets in row['scores'].items()}}),flush=True)
    mediation_rows=[]
    for trial,frame in zip(trials,frames):
        path=out/f'mediation_s{trial["seed"]}.json'
        if path.exists():row=json.loads(path.read_text())
        else:
            row=mediation(c,trial,frame,args.private,pair_sets,oracle,tracker,device,selection['clean_gate_passed']);write_new(path,row)
        mediation_rows.append(row)
        print(json.dumps({'mediation_seed':trial['seed'],'eligible':{p:{k:v['eligible_pairs_total'] for k,v in sets.items()} for p,sets in row['policies'].items()}}),flush=True)
    receipt={'schema':'scm19.completed.v1','freeze_sha256':sha(out/'freeze.json'),'selection':selection,
        'clean':clean,'mediation':mediation_rows,'resources':tracker.report(),'oracle':audit.report(),
        'journal_sha256':sha(args.private/'work.jsonl'),'private_gold_sha256':sha(args.private/'fresh.gold.private.json'),
        'seconds':time.perf_counter()-started,'new_training_after_final':False}
    write_new(out/'completed.json',receipt);print('SCM19_COMPLETE',flush=True)


if __name__=='__main__':main()
