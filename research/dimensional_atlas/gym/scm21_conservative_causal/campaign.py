"""Preregister, train paired controls, freeze, then generate fresh confirmation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import (HERE,Journal,OracleAudit,Model,features,groups,correct_mask,pool,pair_pool,FAMILIES,OOD,
    row_loss,run,losses,score,pair_trace,assay,random_frame,recipe,restore,old_signatures,put,atomic_json,
    digest,sha,write_new,load_model,save_model,device_setup,release,Registry,encoder_hash,bound_parameter_step,MathTask)


def sources():
    files=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PROTOCOL.md']
    for path in ('model.py','tasks.py','scm18_rtx3050/core.py','scm19_clean_topone/core.py','scm19_clean_topone/campaign.py'):
        files.append(HERE.parent/path)
    return {p.relative_to(HERE.parents[3]).as_posix():sha(p) for p in files}


def old_seen():
    seen=old_signatures();root=HERE.parent/'scm19_clean_topone/results'
    seen.update(json.loads((root/'freeze.json').read_text())['pre_final_signatures'])
    public=json.loads((root/'fresh.public.json').read_text())
    for cases in public['clean'].values():
        for case in cases:seen.add(hashlib.sha256(json.dumps({k:v for k,v in case.items() if k!='case_id'},sort_keys=True,separators=(',',':')).encode()).hexdigest())
    for pairs in public['mediation'].values():
        for pair in pairs:
            for case in pair:seen.add(hashlib.sha256(json.dumps({k:v for k,v in case.items() if k!='case_id'},sort_keys=True,separators=(',',':')).encode()).hexdigest())
    root=HERE.parent/'scm20_causal_consistency/results'
    seen.update(json.loads((root/'freeze.json').read_text())['pre_final_signatures'])
    public=json.loads((root/'fresh.public.json').read_text())
    for cases in public['clean'].values():
        for case in cases:seen.add(hashlib.sha256(json.dumps({k:v for k,v in case.items() if k!='case_id'},sort_keys=True,separators=(',',':')).encode()).hexdigest())
    for pairs in public['pairs'].values():
        for pair in pairs:
            for case in pair:seen.add(hashlib.sha256(json.dumps({k:v for k,v in case.items() if k!='case_id'},sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return seen


def historical_registry(old,previous,oracle):
    registry=Registry(old_seen());catalog={}
    for label,path in [('scm19',old/'data.pt'),('scm20',previous/'data.pt')]:
        catalog[label+'_cache_sha256']=sha(path);cached=torch.load(path,weights_only=True,map_location='cpu')
        registry.encoded.update(encoder_hash(x) for x in cached['x'].reshape(-1,2,13,51))
        for role in ('dev_id','dev_ood','calibration'):
            registry.encoded.update(encoder_hash(features(t)) for t in restore(oracle,cached[role]))
        if label=='scm20':
            for role in ('dev_pair_id','dev_pair_ood','calibration_pair'):
                registry.encoded.update(encoder_hash(features(t)) for pair in restore_pairs(oracle,cached[role]) for t in pair)
        else:
            from ..scm19_clean_topone.core import make_pairs
            pairs=make_pairs(oracle,32,19550000,'calibration_pair',FAMILIES,set(cached['seen']),'coefficient')
            expected=json.loads((HERE.parent/'scm19_clean_topone/results/frame_s191.json').read_text())['pair_signatures']
            assert [[a.signature,b.signature] for a,b in pairs]==expected
            registry.encoded.update(encoder_hash(features(t)) for pair in pairs for t in pair)
        del cached
        root=HERE.parent/('scm19_clean_topone' if label=='scm19' else 'scm20_causal_consistency')/'results/fresh.public.json'
        public=json.loads(root.read_text());catalog[label+'_public_sha256']=sha(root)
        cases=[case for rows in public['clean'].values() for case in rows]
        cases += [case for rows in public['mediation' if label=='scm19' else 'pairs'].values() for pair in rows for case in pair]
        for case in cases:
            task=MathTask(case['case_id'],'reference',case['topology'],tuple((tuple(d['input']),tuple(d['output'])) for d in case['demos']),tuple(case['query']),(),('I','I','I'),case['backend'])
            registry.encoded.add(encoder_hash(features(task)))
    return registry,catalog


def prepare(c,private,out,oracle,journal,old,previous):
    receipt=out/'data.json';cache=private/'data.pt'
    if receipt.exists():
        value=json.loads(receipt.read_text());assert sha(cache)==value['cache_sha256']
        return torch.load(cache,weights_only=True),value
    journal.stage='historical_registry';seen,catalog=historical_registry(old,previous,oracle)
    historical_encoded=len(seen.encoded);journal.stage='training_data'
    train=pair_pool(oracle,c['train_pair_count'],210000000,'train',FAMILIES,seen,'coefficient')
    grouping,metadata=groups(oracle);x=[];mask=[]
    for i,pair in enumerate(train):
        x.append(torch.stack([features(t) for t in pair]));mask.append(torch.stack([correct_mask(t,oracle,grouping) for t in pair]))
        if (i+1)%512==0:print(json.dumps({'training_pairs_labeled':i+1}),flush=True)
    data={'x':torch.stack(x),'mask':torch.stack(mask),'train_recipes':[recipe(a) for a,b in train],
        'train_signatures':[[a.signature,b.signature] for a,b in train]}
    del train,x,mask;gc.collect()
    for kind,count,start,families in [('dev_id',c['dev_id_count'],220000000,FAMILIES),('dev_ood',c['dev_ood_count'],230000000,OOD),
            ('calibration',c['calibration_count'],260000000,FAMILIES)]:
        data[kind]=[recipe(t) for t in pool(oracle,count,start,kind,families,seen)]
    for kind,count,start,families in [('dev_pair_id',c['dev_id_pairs'],240000000,FAMILIES),('dev_pair_ood',c['dev_ood_pairs'],250000000,OOD),
            ('calibration_pair',c['calibration_pair_count'],270000000,FAMILIES)]:
        data[kind]=[recipe(a) for a,b in pair_pool(oracle,count,start,kind,families,seen,'coefficient')]
    data['seen']=sorted(seen);data['encoded_seen']=sorted(seen.encoded);torch.save(data,cache)
    receipt_value={'schema':'scm21.data.v1','cache_sha256':sha(cache),'train_pair_n':c['train_pair_count'],
        'training_signatures':data['train_signatures'],'old_signature_count':len(old_seen()),'seen_signatures':data['seen'],
        'split_signatures':{k:[r['signature'] for r in data[k]] for k in ('dev_id','dev_ood','calibration','dev_pair_id','dev_pair_ood','calibration_pair')},
        'mean_positive_programs':float(data['mask'].float().sum(-1).mean()),'grouping':metadata,
        'historical_catalog':catalog,'historical_encoded_count':historical_encoded,'encoded_signature_count':len(seen.encoded),'generation_rejections':dict(seen.rejections)}
    put(receipt,receipt_value);return data,receipt_value


def restore_pairs(oracle,recipes):
    from ..scm19_clean_topone.core import counterfactual
    return [(a,counterfactual(a,oracle)) for a in restore(oracle,recipes)]


def warm_path(old,seed):return old/f'frozen_s{seed}.pt'


def protected(model):return digest(model,('degree_','positions','core.','head.','sae.','adapter.q'))


def resolve_model(info,private,old):return (private if info.get('local',True) else old)/info['file']


@torch.no_grad()
def teacher(c,seed,data,old,private,oracle,journal,device):
    path=private/f'teacher_s{seed}.pt';receipt=HERE/'results'/f'teacher_s{seed}.json'
    if receipt.exists():
        row=json.loads(receipt.read_text());assert sha(path)==row['sha256'];return torch.load(path,weights_only=True),row
    model=load_model(warm_path(old,seed),c,device);journal.stage=f'teacher_s{seed}'
    logits=[];eligible=[];pairs=restore_pairs(oracle,data['train_recipes'])
    for i in range(0,len(data['x']),c['pair_batch']):
        x=data['x'][i:i+c['pair_batch']];flat=x.flatten(0,1).to(device)
        z=run(model,flat,journal)['logits'].cpu().reshape(len(x),2,64);logits.append(z)
        pred=z.argmax(-1).tolist()
        for pair,ps in zip(pairs[i:i+c['pair_batch']],pred):
            correct=[bool(grade_detail_local(oracle,t,p)) for t,p in zip(pair,ps)]
            eligible.append(all(correct))
    value={'logits':torch.cat(logits),'eligible':torch.tensor(eligible,dtype=torch.float32)}
    torch.save(value,path);row={'seed':seed,'file':path.name,'sha256':sha(path),'warm_sha256':sha(warm_path(old,seed)),
        'causal_training_eligible_pairs':int(value['eligible'].sum()),'training_pairs':len(eligible)}
    put(receipt,row);del model,pairs;release();return value,row


def grade_detail_local(oracle,task,pred):
    from ..scm19_clean_topone.core import grade_detail
    return grade_detail(oracle,task,pred)['correct']


@torch.no_grad()
def warm_development(c,seed,dev,dev_pairs,old,out,journal,oracle,device):
    path=out/f'warm_development_s{seed}.json'
    if path.exists():return json.loads(path.read_text())
    model=load_model(warm_path(old,seed),c,device);journal.stage=f'warm_development_s{seed}'
    clean={k:score(model,v,oracle,journal)[0] for k,v in dev.items()};eligibility={}
    for kind,pairs in dev_pairs.items():
        trace=pair_trace(model,pairs,oracle,journal);eligibility[kind]=trace['eligible'];del trace
    value={'seed':seed,'clean':clean,'eligibility':eligibility};put(path,value)
    del model;release();return value


@torch.no_grad()
def development(c,model,seed,dev,dev_pairs,warm,journal,oracle):
    clean={k:score(model,v,oracle,journal)[0] for k,v in dev.items()};mediation={}
    for kind,pairs in dev_pairs.items():
        trace=pair_trace(model,pairs,oracle,journal)
        ids=[i for i,(a,b) in enumerate(zip(trace['eligible'],warm['eligibility'][kind])) if a and b][:c['dev_cap']]
        cells=assay(model,pairs,trace,ids,oracle,journal,['native_vpd_write','random_fiber4'],[c['primary_mediation_visit']],{'random':random_frame(model,seed)})
        mediation[kind]={'eligible_common':sum(a and b for a,b in zip(trace['eligible'],warm['eligibility'][kind])),
            'n':len(ids),'native':cells[0]['fidelity'],'random':cells[1]['fidelity']}
        del trace
    preserved=all(clean[k]['accuracy']>=warm['clean'][k]['accuracy']-c['development_clean_drop'] for k in clean)
    return {'clean':{k:{kk:vv for kk,vv in v.items() if kk!='records'} for k,v in clean.items()},
        'mediation':mediation,'clean_preserved':preserved}


def atomic_torch(path,value):
    temp=path.with_suffix('.new.pt');torch.save(value,temp);os.replace(temp,path)


def train(c,seed,weight,total,data,teacher_value,old,private,out,journal,audit,dev,dev_pairs,warm,device,pilot=False):
    label=f's{seed}_lambda{weight:g}';done=out/(label+'.json');resume=private/(label+'.resume.pt')
    if done.exists():return json.loads(done.read_text())
    model=load_model(warm_path(old,seed),c,device);base_sha=protected(model)
    for p in model.adapter.parameters():p.requires_grad_(True)
    params=list(model.adapter.parameters());opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'])
    rng=torch.Generator().manual_seed(seed+200000);start=0;curves=[];checkpoints={};loss_log=[]
    if resume.exists():
        state=torch.load(resume,weights_only=True,map_location='cpu');model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
        rng.set_state(state['rng']);start=state['step'];curves=state['curves'];checkpoints=state['checkpoints'];loss_log=state['loss_log']
    objective={**c,'causal_weight':weight};before=Counter(journal.work)
    for step in range(start+1,total+1):
        journal.stage=label+'/train';ids=torch.randint(len(data['x']),(c['pair_batch'],),generator=rng)
        x=data['x'][ids].to(device);m=data['mask'][ids].to(device);t=teacher_value['logits'][ids].to(device);valid=teacher_value['eligible'][ids].to(device)
        ra=run(model,x[:,0],journal);rb=run(model,x[:,1],journal)
        patched=run(model,x[:,0],journal,patch=(c['primary_mediation_visit'],rb['writes'][c['primary_mediation_visit']],None))
        total_loss,parts=losses(ra,rb,patched,m[:,0],m[:,1],t[:,0],t[:,1],valid,objective)
        if not torch.isfinite(total_loss):raise RuntimeError('nonfinite causal consistency loss')
        opt.zero_grad(set_to_none=True);total_loss.backward();gnorm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'])
        before_params=[p.detach().clone() for p in params];opt.step()
        step_metrics=bound_parameter_step(params,before_params,c['max_parameter_step_l2'])
        journal.work['adapter_optimizer_steps']+=1
        values={k:float(v.detach()) for k,v in parts.items()}
        journal.event('optimizer_completed',step=step,causal_weight=weight,eligible_in_batch=int(valid.sum()),gradient_norm=float(gnorm),**step_metrics,**values,**journal.memory())
        if step%100==0:loss_log.append({'step':step,**step_metrics,**values})
        wanted=step in c['selection_steps'] if pilot else step==total
        if wanted:
            assert protected(model)==base_sha
            ck=save_model(private/f'{label}_step{step}.pt',model,{'encoding':'coefficient','seed':seed,'steps':step,'locked':True,'causal_weight':weight})
            checkpoints[str(step)]=ck
            if pilot and weight>0:
                journal.stage=label+f'/development_step{step}'
                value=development(c,model,seed,dev,dev_pairs,warm,journal,audit.module)
                curves.append({'step':step,**value})
                print(json.dumps({'pilot':label,'step':step,'clean_preserved':value['clean_preserved'],
                    'clean':{k:v['accuracy'] for k,v in value['clean'].items()},'mediation':value['mediation']}),flush=True)
        if step%c['checkpoint_every']==0 or step==total:
            atomic_torch(resume,{'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                'rng':rng.get_state(),'step':step,'curves':curves,'checkpoints':checkpoints,'loss_log':loss_log})
            journal.checkpoint(audit)
    assert protected(model)==base_sha
    row={'seed':seed,'causal_weight':weight,'steps':total,'checkpoints':checkpoints,'development':curves,'losses':loss_log,
        'protected_sha256':base_sha,'base_and_sae_unchanged':True,'work':dict(journal.work-before)}
    put(done,row);del model,opt,params;release();return row


def choose(c,pilots):
    candidates=[]
    for row in pilots:
        if not row['causal_weight']:continue
        for point in row['development']:
            if not point['clean_preserved']:continue
            if any(v['eligible_common']<c['min_eligible_'+k] for k,v in point['mediation'].items()):continue
            ds=[v['native']-v['random'] for v in point['mediation'].values() if v['native'] is not None]
            if len(ds)!=2:continue
            candidates.append((min(ds),sum(ds)/2,-row['causal_weight'],-point['step'],row,point))
    if not candidates:return {'promoted':False,'causal_weight':0.,'step':0,'reason':'no development-clean-preserving causal checkpoint'}
    *_,row,point=max(candidates,key=lambda v:v[:4])
    return {'promoted':True,'causal_weight':row['causal_weight'],'step':point['step'],'development':point,
        'checkpoint':row['checkpoints'][str(point['step'])],'selected_by':c['selection']}


@torch.no_grad()
def frames(c,seed,models,calib,calib_pairs,private,old,out,oracle,journal,device):
    receipt=out/f'frames_s{seed}.json';path=private/f'frames_s{seed}.pt'
    if receipt.exists():
        r=json.loads(receipt.read_text());assert sha(path)==r['sha256'];return r
    values={}
    for name,info in models.items():
        model=load_model(resolve_model(info,private,old),c,device)
        journal.stage=f'calibration_s{seed}/{name}';_,traces=score(model,calib,oracle,journal,True)
        pairs=pair_trace(model,calib_pairs,oracle,journal);pca=[];semantic=[];usage=torch.zeros(c['sae_width'])
        for visit in range(c['visits']):
            h=torch.stack([r['states'][visit] for r in traces]);_,_,vh=torch.linalg.svd((h-h.mean(0)).double(),full_matrices=False);pca.append(vh[:4].T.float())
            d=torch.stack([b['states'][visit]-a['states'][visit] for a,b in zip(pairs['trace_a'],pairs['trace_b'])]);_,_,vh=torch.linalg.svd(d.double(),full_matrices=False);semantic.append(vh[:4].T.float())
            usage+=model.sae.encode(F.layer_norm(h.to(device),(c['hidden'],))).sum(0).cpu()
        order=usage.argsort(descending=True)[:4];sae=torch.linalg.qr(model.sae.decoder.weight.detach().cpu()[:,order].double()).Q.float()
        q=model.adapter.q.cpu();native=model.adapter.write.detach().cpu().T;native=native-q@(q.T@native)
        u,s,_=torch.linalg.svd(native.double(),full_matrices=False);native=u[:,s>s.max()*1e-6].float()
        values[name]={'pca4':pca,'semantic_diff4':semantic,'sae_group4':[sae]*4,'native_write4':[native]*4,
            'random':random_frame(model,seed),'sae_features':order,'q':q}
        del model,traces,pairs;release()
    torch.save(values,path);r={'seed':seed,'file':path.name,'sha256':sha(path),
        'calibration_signatures':[t.signature for t in calib],'pair_signatures':[[a.signature,b.signature] for a,b in calib_pairs]}
    put(receipt,r);return r


def model_map(c,seed,selection,training,old,private):
    warm=warm_path(old,seed)
    baseline={'file':warm.name,'local':False,'file_sha256':sha(warm),'model_sha256':digest(load_model(warm,c,'cpu')),'encoding':'coefficient'}
    if selection['promoted']:
        step=str(selection['step']);control=training[0.]['checkpoints'][step];causal=training[selection['causal_weight']]['checkpoints'][step]
    else:control=causal=baseline
    return {'warm':baseline,'control':control,'causal':causal}


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--old-private',type=Path,required=True);p.add_argument('--previous-private',type=Path,required=True)
    args=p.parse_args();c=json.loads((HERE/'config.json').read_text());out=HERE/'results'
    registration=json.loads((out/'registration.json').read_text());assert sources()==registration['source_sha256']
    assert sha(args.oracle)==registration['oracle_sha256']
    for name,wanted in registration['historical_signature_source_sha256'].items():assert sha(HERE.parent/name)==wanted
    if (out/'completed.json').exists():print('SCM21_ALREADY_COMPLETE');return
    args.private.mkdir(parents=True,exist_ok=True);device=device_setup('cuda',c);journal=Journal(args.private/'work.jsonl',c)
    audit=OracleAudit(args.oracle,journal);oracle=audit.module;oracle._scm19_audit=audit
    journal.event('process_started',source_sha256=sources())
    try:
        for info in registration['warm_starts']:assert sha(warm_path(args.old_private,info['seed']))==info['file_sha256']
        data,receipt=prepare(c,args.private,out,oracle,journal,args.old_private,args.previous_private);journal.checkpoint(audit)
        assert receipt['historical_catalog']==registration['historical_catalog']
        dev={k:restore(oracle,data['dev_'+k]) for k in ('id','ood')};dp={k:restore_pairs(oracle,data['dev_pair_'+k]) for k in ('id','ood')}
        first=c['seeds'][0];warm=warm_development(c,first,dev,dp,args.old_private,out,journal,oracle,device)
        teachers,tinfo=teacher(c,first,data,args.old_private,args.private,oracle,journal,device)
        pilots=[train(c,first,w,c['training_steps'],data,teachers,args.old_private,args.private,out,journal,audit,dev,dp,warm,device,True) for w in c['causal_weights']]
        selection=choose(c,pilots);put(out/'selection.json',selection) if not (out/'selection.json').exists() else None
        assert json.loads((out/'selection.json').read_text())==selection
        print(json.dumps({'selection':selection}),flush=True);all_models=[]
        for seed in c['seeds']:
            if seed==first:trained={r['causal_weight']:r for r in pilots}
            elif selection['promoted']:
                warm=warm_development(c,seed,dev,dp,args.old_private,out,journal,oracle,device)
                teachers,_=teacher(c,seed,data,args.old_private,args.private,oracle,journal,device)
                trained={w:train(c,seed,w,selection['step'],data,teachers,args.old_private,args.private,out,journal,audit,dev,dp,warm,device)
                    for w in (0.,selection['causal_weight'])}
            else:trained={}
            all_models.append({'seed':seed,'models':model_map(c,seed,selection,trained,args.old_private,args.private)})
        calib=restore(oracle,data['calibration']);cp=restore_pairs(oracle,data['calibration_pair'])
        frame_receipts=[frames(c,t['seed'],t['models'],calib,cp,args.private,args.old_private,out,oracle,journal,device) for t in all_models]
        frozen={'schema':'scm21.freeze.v1','source_sha256':sources(),'registration_sha256':sha(out/'registration.json'),
            'selection_sha256':sha(out/'selection.json'),'data_sha256':sha(out/'data.json'),'models':all_models,
            'frames':frame_receipts,'pre_final_signatures':data['seen'],'pre_final_encoded_signatures':data['encoded_seen'],'official_roster_calls':0}
        if not (out/'freeze.json').exists():put(out/'freeze.json',frozen)
        else:assert json.loads((out/'freeze.json').read_text())==frozen
        assert sources()==registration['source_sha256'];journal.event('all_weights_and_directions_frozen',freeze_sha256=sha(out/'freeze.json'));journal.checkpoint(audit)
        # No final data are created or scored until every model/direction freezes.
        seen=Registry(set(data['seen']),data['encoded_seen']);clean={'id':pool(oracle,c['clean_id_count'],280000000,'fresh_clean_id',FAMILIES,seen),
            'ood':pool(oracle,c['clean_ood_count'],290000000,'fresh_clean_ood',OOD,seen)}
        pairs={'id':pair_pool(oracle,c['mediation_id_pairs'],300000000,'fresh_pair_id',FAMILIES,seen,'coefficient'),
            'ood':pair_pool(oracle,c['mediation_ood_pairs'],310000000,'fresh_pair_ood',OOD,seen,'coefficient')}
        gold={'clean':{k:[recipe(t) for t in v] for k,v in clean.items()},'pairs':{k:[recipe(a) for a,b in v] for k,v in pairs.items()}}
        if not (args.private/'fresh.gold.private.json').exists():put(args.private/'fresh.gold.private.json',gold)
        public={'clean':{k:[t.public for t in v] for k,v in clean.items()},'pairs':{k:[[a.public,b.public] for a,b in v] for k,v in pairs.items()}}
        if not (out/'fresh.public.json').exists():put(out/'fresh.public.json',public)
        final_receipts=[]
        for trial,fr in zip(all_models,frame_receipts):
            seed=trial['seed'];path=out/f'final_s{seed}.json'
            if path.exists():final_receipts.append({'seed':seed,'file':path.name,'sha256':sha(path)});continue
            frame=torch.load(args.private/fr['file'],weights_only=True);scores={};traces={};checks={}
            for name,info in trial['models'].items():
                model=load_model(resolve_model(info,args.private,args.old_private),c,device)
                assert digest(model)==info['model_sha256'];journal.stage=f'fresh_clean_s{seed}/{name}'
                scores[name]={k:score(model,v,oracle,journal)[0] for k,v in clean.items()}
                journal.stage=f'fresh_eligibility_s{seed}/{name}'
                traces[name]={k:pair_trace(model,v,oracle,journal) for k,v in pairs.items()};del model;release()
            common={k:[i for i in range(len(v)) if all(traces[n][k]['eligible'][i] for n in traces)] for k,v in pairs.items()}
            for name,info in trial['models'].items():
                model=load_model(resolve_model(info,args.private,args.old_private),c,device)
                checks[name]={}
                for kind,v in pairs.items():
                    journal.stage=f'fresh_mediation_s{seed}/{name}/{kind}'
                    ids=common[kind][:c['eligible_cap_per_stratum']]
                    cells=assay(model,v,traces[name][kind],ids,oracle,journal,c['mediation_methods'],range(c['visits']),frame[name])
                    checks[name][kind]={'attempted_pairs':len(v),'own_eligible':sum(traces[name][kind]['eligible']),
                        'common_eligible':len(common[kind]),'assayed_pairs':len(ids),'common_ids':[v[i][0].case_id for i in ids],
                        'clean_recipient':traces[name][kind]['recipient'],'clean_donor':traces[name][kind]['donor'],'cells':cells}
                assert digest(model)==info['model_sha256'];del model;release()
            value={'seed':seed,'clean':scores,'mediation':checks};put(path,value)
            final_receipts.append({'seed':seed,'file':path.name,'sha256':sha(path)})
            print(json.dumps({'final_seed':seed,'clean':{n:{k:v['accuracy'] for k,v in sets.items()} for n,sets in scores.items()},
                'common_eligible':{k:len(v) for k,v in common.items()}}),flush=True)
            del traces;gc.collect();journal.checkpoint(audit)
        journal.event('campaign_completed');journal.checkpoint(audit)
        completed={'schema':'scm21.completed.v1','registration_sha256':sha(out/'registration.json'),'freeze_sha256':sha(out/'freeze.json'),
            'selection':selection,'final_receipts':final_receipts,'resources':journal.report(),'oracle':audit.report(),
            'journal_sha256':sha(args.private/'work.jsonl'),'private_gold_sha256':sha(args.private/'fresh.gold.private.json'),
            'training_after_final':False,'official_roster_calls':0}
        put(out/'completed.json',completed);print('SCM21_COMPLETE',flush=True)
    except BaseException:
        journal.event('process_exception',traceback=traceback.format_exc());journal.checkpoint(audit);raise


if __name__=='__main__':main()
