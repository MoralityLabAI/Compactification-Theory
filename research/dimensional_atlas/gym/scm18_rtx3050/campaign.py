"""Sequential local SCM-18 training, freeze, and separately launched final readout."""
from __future__ import annotations
import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F

from .core import (HERE, MathModel, Meter, canonical, checkpoint, device_setup,
                   digest, environment, evaluate, fit_dictionary, load_checkpoint,
                   pool, release, replace_adapter, sha, tensors, train_adapter,
                   train_base, write_new)
from ..tasks import (EXPECTED_EVALMATH_SHA256, PROGRAMS, MathTask, encode,
                     load_evalmath, make_evalmath_case, verify_order)
from ..scm16.landscape import neighbors


class OracleAudit:
    """Count actual source calls, including grade's internal acceptance check."""
    def __init__(self, module, allow_final=False):
        self.module, self.counts = module, Counter()
        self.allow_final = allow_final
        for name in ('check_candidate','grade','execute','make_case','roster'):
            original = getattr(module,name)
            def wrapped(*args,_name=name,_original=original,**kwargs):
                if _name=='roster' and not self.allow_final:
                    raise RuntimeError('official roster sealed until a separately verified freeze')
                self.counts[_name] += 1
                return _original(*args,**kwargs)
            setattr(module,name,wrapped)


def make_pools(oracle,c):
    seen=set(); families=('signed','product','sparse')
    data={}
    for name,n,start,ff in (
        ('train',c['train_count'],1810000,families),
        ('discovery',c['discovery_count'],1820000,families),
        ('calibration',c['calibration_count'],1830000,families),
        ('confirmation',c['confirmation_count'],1840000,('rational','squared','lacunary','twisted_product')),
        ('development',c['development_count'],1850000,families)):
        data[name]=pool(oracle,n,start,name,ff,seen)
    return data,seen


def mine(oracle,cases):
    all_rows=[]
    for task in cases:
        if not task.case_id.startswith('discovery:'):
            raise ValueError('discovery-only mining')
        values={}
        details={}
        for graph in PROGRAMS:
            result=oracle.check_candidate(task.oracle_case,graph)
            values[graph]=min(13 if f is None else f for f in result['first_faults'])
            details[graph]=result
        maxima=[]
        for graph in PROGRAMS:
            adjacent=list(neighbors(graph))
            if values[graph]>=max(values[g] for g in adjacent):
                maxima.append({'graph':list(graph),'score':values[graph],
                               'neighbor_scores':[values[g] for g in adjacent],
                               'first_faults':details[graph]['first_faults'],
                               'prefix_accepted':details[graph]['accepted']})
        deceptive=[m for m in maxima if tuple(m['graph'])!=task.source_graph]
        if deceptive:
            chosen=sorted(deceptive,key=lambda r:(-r['score'],r['graph']))[0]
            all_rows.append({'id':task.case_id,'signature':task.signature,'family':task.family,
                             'topology':task.topology,'source_graph':list(task.source_graph),
                             'maximum':chosen,'maxima_count':len(maxima)})
    cells=set();selected=[]
    for row in sorted(all_rows,key=lambda r:(-r['maximum']['score'],r['id'])):
        key=(row['family'],row['topology'],tuple(row['source_graph']))
        if key not in cells:
            cells.add(key);selected.append(row)
        if len(selected)==3:
            break
    return {'definition':'>= all nine one-edit neighbors, including plateau maxima',
            'tasks':len(cases),'paid_prefix_checks':64*len(cases), 'rows':all_rows,'selected':selected}


def replay_cases(oracle,mining,seen):
    cases=[]
    for i,row in enumerate(mining['selected']):
        for j in range(2):
            case=make_evalmath_case(oracle,1880000+i*100+j,case_id=f'replay:{i}:{j}',
                                    family=row['family'],topology=row['topology'],graph=tuple(row['source_graph']))
            if case.signature in seen:
                raise ValueError('replay overlaps public signature')
            seen.add(case.signature);cases.append(case)
    return cases


def sources():
    paths=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PROTOCOL.md',
        HERE.parent/'model.py',HERE.parent/'tasks.py',HERE.parent/'scm16/fiber.py']
    root=HERE.parents[3]
    return {str(p.relative_to(root)).replace('\\','/'):sha(p) for p in paths}


def train(args,c,device,oracle,audit):
    out=args.output
    if (out/'freeze.json').exists() or (out/'training.json').exists():
        raise FileExistsError('refuse a second training campaign in the same output directory')
    args.private.mkdir(parents=True,exist_ok=True)
    data,seen=make_pools(oracle,c)
    mining=mine(oracle,data['discovery'])
    replay=replay_cases(oracle,mining,seen)
    write_new(args.private/'replay.gold.private.json',[
        {'id':t.case_id,'signature':t.signature,'graph':t.source_graph,'future':t.hidden_target} for t in replay])
    write_new(out/'replay.public.json',[t.public for t in replay])
    write_new(out/'registration.json',{'config':c,'config_sha256':sha(HERE/'config.json'),
        'sources':sources(),'splits':{name:[t.signature for t in ts] for name,ts in data.items()},
        'replay_signatures':[t.signature for t in replay],'mining':mining,'oracle_sha256':sha(args.oracle)})
    records=[];screen=[];manifest=[]
    # Screen with clean tied baselines only; no official or final roster access.
    for visits in c['screen_visits']:
        torch.manual_seed(c['seeds'][0])
        model=MathModel(c,visits).to(device);meter=Meter(device,c);model.meter=meter
        init=digest(model,('embed.','positions','core.','head.'))
        report=train_base(model,data['train'],c,c['seeds'][0])
        dev=evaluate(model,data['development'],oracle,c['budget'])
        item=checkpoint(args.private/f'screen_v{visits}.pt',model,
                        {'visits':visits,'rank':4,'seed':c['seeds'][0],'objective':'final','untied':False})
        screen.append({'visits':visits,'development':dev,'base_training':report,
                       'initial_base_sha256':init,'checkpoint':item,'resources':meter.report()})
        print(json.dumps({'screen_visits':visits,'development_accuracy':dev['overall']['accuracy'],
                          'resources':meter.report()}),flush=True)
        del model,meter;release()
    selected_visits=sorted(screen,key=lambda r:(-r['development']['overall']['accuracy'],r['visits']))[0]['visits']
    selected_rank=None
    for seed in c['seeds']:
        for objective in c['objectives']:
            meter=Meter(device,c)
            if seed==c['seeds'][0] and objective=='final':
                model,_=load_checkpoint(args.private/f'screen_v{selected_visits}.pt',c,device)
                base_report={'reused_screen':True,'steps':0}
            else:
                torch.manual_seed(seed)
                model=MathModel(c,selected_visits).to(device);model.meter=meter
                base_report=train_base(model,data['train'],c,seed,objective)
            model.meter=meter
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                unfit=MathModel(c,selected_visits)
                random_dictionary={k:v.detach().cpu().clone() for k,v in unfit.sae.state_dict().items()}
                del unfit
            dictionary=fit_dictionary(model,data['train'],c,seed)
            fixed_base=digest(model,('embed.','positions','core.','head.'))
            base_dev=evaluate(model,data['development'],oracle,c['budget'])
            ranks=c['ranks'] if objective=='final' else [selected_rank]
            candidates=[]
            for rank in ranks:
                for sector in ('joint','fiber_only'):
                    replace_adapter(model,c,rank,seed)
                    fit=train_adapter(model,data['train']+replay,c,seed,sector,objective,oracle)
                    dev=evaluate(model,data['development'],oracle,c['budget'],policy=sector)
                    name=f's{seed}_{objective}_r{rank}_{sector}.pt'
                    meta={'visits':selected_visits,'rank':rank,'seed':seed,'objective':objective,
                          'sector':sector,'untied':False,'random_dictionary':False}
                    ck=checkpoint(args.private/name,model,meta);manifest.append(ck)
                    candidates.append({'rank':rank,'sector':sector,'development':dev,
                                       'training':fit,'checkpoint':ck})
            if selected_rank is None:
                selected_rank=sorted([r for r in candidates if r['sector']=='joint'],
                                     key=lambda r:(-r['development']['overall']['accuracy'],r['rank']))[0]['rank']
            # Same shape, optimizer budget and base with an unfitted dictionary.
            model.sae.load_state_dict(random_dictionary)
            replace_adapter(model,c,selected_rank,seed)
            random_fit=train_adapter(model,data['train']+replay,c,seed,'joint',objective,oracle)
            random_dev=evaluate(model,data['development'],oracle,c['budget'],policy='joint')
            random_ck=checkpoint(args.private/f's{seed}_{objective}_random.pt',model,
                {'visits':selected_visits,'rank':selected_rank,'seed':seed,'objective':objective,
                 'sector':'joint','untied':False,'random_dictionary':True})
            manifest.append(random_ck)
            assert digest(model,('embed.','positions','core.','head.'))==fixed_base
            records.append({'seed':seed,'objective':objective,'base_training':base_report,
                            'dictionary':dictionary,'base_development':base_dev,
                            'base_sha256':fixed_base,'candidates':candidates,
                            'random_dictionary':{'training':random_fit,'development':random_dev,'checkpoint':random_ck},
                            'resources':meter.report()})
            print(json.dumps({'seed':seed,'objective':objective,'selected_rank':selected_rank,
                              'base_dev':base_dev['overall']['accuracy'],'resources':meter.report()}),flush=True)
            del model,meter,random_dictionary;release()
        # Untied comparator: identical cloned initial core and the same expanded depth.
        torch.manual_seed(seed)
        untied=MathModel(c,selected_visits,untied=True).to(device)
        meter=Meter(device,c);untied.meter=meter
        fit=train_base(untied,data['train'],c,seed)
        ck=checkpoint(args.private/f's{seed}_untied.pt',untied,
                      {'visits':selected_visits,'rank':4,'seed':seed,'objective':'final','untied':True})
        manifest.append(ck)
        records.append({'seed':seed,'objective':'untied','base_training':fit,'checkpoint':ck,
                        'development':evaluate(untied,data['development'],oracle,c['budget']),
                        'parameter_count':sum(p.numel() for p in untied.parameters()),'resources':meter.report()})
        del untied,meter;release()
    # Calibrate every direction/compression basis now, before final-data access.
    from .diagnostics import calibrate
    primary=args.private/f's{c["seeds"][0]}_final_r{selected_rank}_joint.pt'
    model,meta=load_checkpoint(primary,c,device);meter=Meter(device,c);model.meter=meter
    frame=calibrate(model,data['calibration'])
    frame_file=args.private/'frame.pt'
    torch.save(frame,frame_file)
    frame_resources=meter.report()
    del model,meter;release()
    training={'config_sha256':sha(HERE/'config.json'),'environment':environment(),
              'screen':screen,'selected_visits':selected_visits,'selected_rank':selected_rank,
              'records':records,'oracle_source_calls':dict(audit.counts),'frame_resources':frame_resources,
              'official_roster_calls':audit.counts['roster']}
    write_new(out/'training.json',training)
    freeze={'schema':'scm18.frozen.v1','config_sha256':sha(HERE/'config.json'),
            'registration_sha256':sha(out/'registration.json'),'training_sha256':sha(out/'training.json'),
            'oracle_sha256':sha(args.oracle),'sources':sources(),'models':manifest,
            'selected_visits':selected_visits,'selected_rank':selected_rank,
            'primary_checkpoint':primary.name,'frame_sha256':sha(frame_file),
            'official_roster_calls_before_freeze':audit.counts['roster'],
            'all_pre_final_signatures':sorted(seen)}
    assert freeze['official_roster_calls_before_freeze']==0
    write_new(out/'freeze.json',freeze)
    print(json.dumps({'status':'all_models_and_selection_frozen','manifest':str(out/'freeze.json')}),flush=True)


def final(args,c,device,oracle,audit):
    frozen=json.loads((args.output/'freeze.json').read_text())
    assert sha(HERE/'config.json')==frozen['config_sha256']
    assert sha(args.output/'registration.json')==frozen['registration_sha256']
    assert sha(args.output/'training.json')==frozen['training_sha256']
    assert sources()==frozen['sources'], 'source changed after training freeze'
    assert sha(args.oracle)==frozen['oracle_sha256']==EXPECTED_EVALMATH_SHA256
    assert sha(args.private/'frame.pt')==frozen['frame_sha256']
    for ck in frozen['models']:
        assert sha(args.private/ck['file'])==ck['file_sha256']
    # This is the first reachable official-roster access in the campaign.
    audit.allow_final=True
    from ..scm16.campaign import load_public_evalmath
    official=load_public_evalmath(oracle)
    seen=set(frozen['all_pre_final_signatures'])
    assert not ({t.signature for t in official}&seen)
    fresh=pool(oracle,c['fresh_final_count'],1860000,'final',
               ('rational','squared','lacunary','twisted_product'),seen)
    registration=json.loads((args.output/'registration.json').read_text())
    challenges=[]
    for i,row in enumerate(registration['mining']['selected']):
        for j in range(2):
            t=make_evalmath_case(oracle,1890000+i*100+j,case_id=f'challenge:{i}:{j}',
                  family=row['family'],topology=row['topology'],graph=tuple(row['source_graph']))
            assert t.signature not in seen;seen.add(t.signature);challenges.append(t)
    write_new(args.private/'challenge.gold.private.json',[
        {'id':t.case_id,'graph':t.source_graph,'future':t.hidden_target} for t in challenges])
    write_new(args.output/'challenges.public.json',[t.public for t in challenges])
    results=[]
    for ck in frozen['models']:
        # Rank sweep outside the selected rank is retained as development evidence.
        if not ck.get('untied') and ck['rank']!=frozen['selected_rank']:
            continue
        model,meta=load_checkpoint(args.private/ck['file'],c,device)
        assert digest(model)==ck['model_sha256']
        meter=Meter(device,c);model.meter=meter
        policies=(['tied'] if meta.get('untied') else ['joint'] if meta.get('random_dictionary') else
                  ['fiber_only'] if meta['sector']=='fiber_only' else
                  ['tied','joint','fiber_only','visible_only','shuffled_visit','zero_codes'])
        scores={}
        for name in policies:
            scores[name]={label:evaluate(model,ts,oracle,c['budget'],policy=name)
                          for label,ts in [('official',official),('fresh',fresh),('challenges',challenges)]}
        assert digest(model)==ck['model_sha256']
        results.append({'checkpoint':ck,'policies':scores,'resources':meter.report()})
        print(json.dumps({'final_checkpoint':ck['file'],'official':{k:v['official']['overall'] for k,v in scores.items()}}),flush=True)
        del model,meter;release()
    symbolic={}
    for budget in (12,64):
        rows=[verify_order(t,list(range(64)),budget=budget,oracle=oracle) for t in official]
        symbolic[str(budget)]={'n':len(rows),'correct':sum(r['future_correct'] for r in rows),
                              'paid_queries':sum(r['queries'] for r in rows)}
    write_new(args.output/'final.json',{'freeze_sha256':sha(args.output/'freeze.json'),
        'results':results,'symbolic':symbolic,'official_count':len(official),
        'official_unique_signatures':len({t.signature for t in official}),
        'fresh_count':len(fresh),'oracle_source_calls':dict(audit.counts),
        'challenge_gold_sha256':sha(args.private/'challenge.gold.private.json')})


def diagnostics(args,c,device,oracle,audit):
    from .diagnostics import (compactification, fiber_transport, geometry, interchange,
                              topology_diagnostics)
    f=json.loads((args.output/'freeze.json').read_text())
    model,meta=load_checkpoint(args.private/f['primary_checkpoint'],c,device)
    frame=torch.load(args.private/'frame.pt',weights_only=True)
    assert sha(args.private/'frame.pt')==f['frame_sha256']
    data,_=make_pools(oracle,c)
    meter=Meter(device,c);model.meter=meter
    result={'geometry':geometry(model,data['confirmation'],frame,c,oracle),
            'fiber_transport':fiber_transport(model,data['confirmation'][:4]),
            'compactification':compactification(model,data['confirmation'][:4],frame,c,oracle),
            'interchange':interchange(model,oracle,frame,c),
            'topology':topology_diagnostics(model,data['confirmation'][0],frame,args.certificates),
            'frame':{'axes':frame['axes'].tolist(),'feature_order':frame['feature_order'].tolist(),
                     'usage':frame['usage'].tolist()},
            'resources':meter.report(),'oracle_source_calls':dict(audit.counts)}
    del model,meter;release()
    untied_ck=next(ck for ck in f['models'] if ck.get('untied') and ck['seed']==c['seeds'][0])
    untied,_=load_checkpoint(args.private/untied_ck['file'],c,device)
    untied_meter=Meter(device,c);untied.meter=untied_meter
    result['untied_geometry']=geometry(untied,data['confirmation'],frame,c,oracle,policy='tied')
    result['untied_geometry_note']='same calibration-selected intervention axes, examples, visits and differences; untied has more parameters'
    result['untied_resources']=untied_meter.report()
    result['oracle_source_calls']=dict(audit.counts)
    write_new(args.output/'diagnostics.json',result)
    print(json.dumps({'diagnostics_completed':True,'resources':result['resources']}),flush=True)


def parity(args,c,device,oracle,audit):
    """Original SCM-15 optimizer, same CPU initial state and pinned math examples."""
    from .. import run as old
    from ..model import LoopedMathTransformer
    train=old._make_pool(24,110000,backend='evalmath',oracle=oracle,split='train')
    discovery=old._make_pool(12,220000,backend='evalmath',oracle=oracle,split='discovery')
    torch.manual_seed(15)
    model=LoopedMathTransformer(hidden=16,heads=4,visits=4,sae_width=20,sae_k=3,vpd_rank=2)
    initial=digest(model)
    model.to(device)
    meter=Meter(device,c)
    handle=model.register_forward_pre_hook(lambda m,a:meter.forward(len(a[0]),m.visits))
    original=old.tensorize
    old.tensorize=lambda tasks:original(tasks,device=device)
    reports={}
    try:
        for i,(stage,steps) in enumerate([('base',3),('sae',2),('vpd',3)]):
            reports[stage]=old.optimizer_steps(model,train,stage=stage,steps=steps,batch=4,seed=16+i,lr=.002)
        arms={}
        with torch.no_grad():
            for name,kwargs in [('tied',{}),('joint',{'use_vpd':True}),
                ('shuffled',{'use_vpd':True,'visit_order':(3,2,1,0)}),
                ('zero',{'use_vpd':True,'code_override':'zero'})]:
                rows=[]
                for task in discovery:
                    logits=model(encode(task)[None].to(device),**kwargs).logits[0]
                    rows.append(verify_order(task,logits.argsort(descending=True).cpu().tolist(),budget=12,oracle=oracle))
                arms[name]={'correct':sum(r['future_correct'] for r in rows),'n':len(rows),
                            'paid_queries':sum(r['queries'] for r in rows)}
        write_new(args.output/f'parity_{device.type}.json',{'initial_sha256':initial,
            'example_sha256':canonical([t.public for t in train+discovery]),'oracle_sha256':sha(args.oracle),
            'phases':reports,'arms':arms,'frozen_model_sha256':digest(model),
            'optimizer_steps':8,'resources':meter.report(),'oracle_source_calls':dict(audit.counts),
            'scope':'minimal original SCM-15 optimizer on pinned EvalMath discovery; not final benchmark'})
    finally:
        old.tensorize=original;handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('parity','train','final','diagnostics'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--certificates',type=Path)
    args=p.parse_args()
    c=json.loads((HERE/'config.json').read_text())
    device=device_setup(args.device,c)
    oracle=load_evalmath(args.oracle)
    audit=OracleAudit(oracle)
    {'train':train,'parity':parity,'final':final,'diagnostics':diagnostics}[args.stage](args,c,device,oracle,audit)


if __name__=='__main__':
    main()
