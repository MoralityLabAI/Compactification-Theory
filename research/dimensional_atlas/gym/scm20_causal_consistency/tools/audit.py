"""CPU-only independent exact replay of the frozen confirmation receipts."""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from ..core import HERE,PROGRAMS,features,sha,digest,load_model
from ..campaign import sources,protected,restore,restore_pairs,resolve_model
from ...tasks import load_evalmath


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--old-private',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);args=p.parse_args();out=HERE/'results'
    read=lambda name:json.loads((out/name).read_text());reg=read('registration.json');c=reg['config'];done=read('completed.json');frozen=read('freeze.json')
    assert reg['source_sha256']==sources() and reg['config_sha256']==sha(HERE/'config.json')
    assert sha(args.oracle)==reg['oracle_sha256']
    for key,name in [('registration_sha256','registration.json'),('freeze_sha256','freeze.json')]:assert done[key]==sha(out/name)
    assert frozen['selection_sha256']==sha(out/'selection.json') and frozen['data_sha256']==sha(out/'data.json')
    assert sha(args.private/'fresh.gold.private.json')==done['private_gold_sha256']
    assert sha(args.private/'work.jsonl')==done['journal_sha256']
    data=read('data.json');assert sha(args.private/'data.pt')==data['cache_sha256'];checks=Counter()
    oracle=load_evalmath(args.oracle);verification_requests=Counter();verification_caches={}
    for name in ('unary','convolution','reciprocal','execute'):
        original=getattr(oracle,name)
        @functools.lru_cache(maxsize=32768)
        def cache(*xs,_f=original):return _f(*xs)
        verification_caches[name]=cache
        def wrapped(*xs,_f=cache,_name=name):
            verification_requests[_name]+=1
            return _f(*(tuple(v) if isinstance(v,list) else v for v in xs))
        setattr(oracle,name,wrapped)
    for name in ('make_case','check_candidate','grade'):
        original=getattr(oracle,name)
        def counted(*xs,_f=original,_name=name,**kwargs):
            verification_requests[_name]+=1
            return _f(*xs,**kwargs)
        setattr(oracle,name,counted)
    oracle.roster=lambda **kwargs:(_ for _ in ()).throw(AssertionError('official roster forbidden'))
    gold=json.loads((args.private/'fresh.gold.private.json').read_text());public=read('fresh.public.json')
    clean={k:restore(oracle,v) for k,v in gold['clean'].items()};pairs={k:restore_pairs(oracle,v) for k,v in gold['pairs'].items()}
    cases={};signatures=[];pair_index={}
    for kind,tasks in clean.items():
        assert [t.public for t in tasks]==public['clean'][kind]
        for t in tasks:cases[t.case_id]=t.oracle_case;signatures.append(t.signature)
    for kind,objects in pairs.items():
        assert [[a.public,b.public] for a,b in objects]==public['pairs'][kind]
        for a,b in objects:
            assert a.query==b.query and a.source_graph[1:]==b.source_graph[1:]
            assert {'I':'N','N':'S','S':'R','R':'I'}[a.source_graph[0]]==b.source_graph[0]
            assert a.hidden_target!=b.hidden_target and a.signature!=b.signature
            assert not torch.equal(features(a)[...,:28],features(b)[...,:28])
            assert [x for x,y in a.oracle_case.demos]==[x for x,y in b.oracle_case.demos]
            pair_index[a.case_id]=(a.oracle_case,b.oracle_case);signatures.extend((a.signature,b.signature))
            checks['semantic_pairs_reconstructed']+=1
    assert len(signatures)==len(set(signatures)) and not set(signatures)&set(frozen['pre_final_signatures'])
    checks['fresh_unique_signatures']=len(signatures)
    # Public signatures include query coefficients, but inference encodes only
    # demonstrations. Quantify actual encoded-observation reuse separately;
    # this diagnostic never changes the registered sample or primary endpoint.
    def encoded(tensor):return hashlib.sha256(tensor[...,:28].contiguous().numpy().tobytes()).hexdigest()
    reference=set();reference_counts={}
    for label,path in [('scm19',args.old_private/'data.pt'),('scm20',args.private/'data.pt')]:
        cached=torch.load(path,weights_only=True,map_location='cpu')
        seen_inputs={encoded(x) for x in cached['x'].reshape(-1,2,13,51)}
        reference.update(seen_inputs);reference_counts[label+'_training_unique']=len(seen_inputs)
        for role in ('dev_id','dev_ood','calibration'):
            tasks=restore(oracle,cached[role]);values={encoded(features(t)) for t in tasks}
            reference.update(values);reference_counts[label+'_'+role+'_unique']=len(values)
        if label=='scm20':
            for role in ('dev_pair_id','dev_pair_ood','calibration_pair'):
                objects=restore_pairs(oracle,cached[role]);values={encoded(features(t)) for pair in objects for t in pair}
                reference.update(values);reference_counts[label+'_'+role+'_unique']=len(values)
        del cached
    encoded_overlap={'reference_unique_inputs':len(reference),'reference_role_counts':reference_counts,'clean':{},'pairs':{},
        'scope':'SCM19 training/development/clean calibration and all SCM20 pre-final roles; actual coefficient encoder input, excluding query; post-hoc diagnostic only'}
    for kind,tasks in clean.items():
        overlap=[t.case_id for t in tasks if encoded(features(t)) in reference]
        encoded_overlap['clean'][kind]={'n':len(tasks),'overlap_n':len(overlap),'ids':overlap}
    for kind,objects in pairs.items():
        flags=[(encoded(features(a)) in reference,encoded(features(b)) in reference) for a,b in objects]
        encoded_overlap['pairs'][kind]={'n':len(objects),'either_overlap_n':sum(a or b for a,b in flags),
            'both_overlap_n':sum(a and b for a,b in flags),'ids':[pair[0].case_id for pair,flag in zip(objects,flags) if any(flag)]}
    def grade(case,pred):
        accepted=bool(oracle.check_candidate(case,PROGRAMS[pred])['accepted'])
        correct=bool(oracle.grade(case,{'graph':PROGRAMS[pred]})['future_correct']) if accepted else False
        checks['recorded_predictions_regraded']+=1
        return int(correct),int(accepted),int(accepted and not correct)
    finals=[]
    for receipt in done['final_receipts']:
        assert sha(out/receipt['file'])==receipt['sha256'];trial=read(receipt['file']);finals.append(trial)
        for name,sets in trial['clean'].items():
            for kind,result in sets.items():
                count=0
                for row in result['records']:
                    actual=grade(cases[row['id']],row['prediction'])
                    assert actual==(row['correct'],row['prefix_accepted'],row['false_future_acceptance']);count+=actual[0]
                assert count==result['correct'] and result['accuracy']==count/result['n'] and result['n']==len(result['records'])
                assert result['inference_verifier_calls']==0
        for kind,objects in pairs.items():
            eligible={}
            for name,sets in trial['mediation'].items():
                assay=sets[kind];rec=assay['clean_recipient'];donor=assay['clean_donor']
                for rows,role in ((rec['records'],0),(donor['records'],1)):
                    for row,(a,b) in zip(rows,objects):
                        case=(a,b)[role];assert row['id']==case.case_id
                        actual=grade(case.oracle_case,row['prediction'])
                        assert actual==(row['correct'],row['prefix_accepted'],row['false_future_acceptance'])
                eligible[name]=[bool(a['correct'] and b['correct']) for a,b in zip(rec['records'],donor['records'])]
                assert sum(eligible[name])==assay['own_eligible']
            ids=[i for i in range(len(objects)) if all(v[i] for v in eligible.values())]
            chosen=ids[:c['eligible_cap_per_stratum']];expected=[objects[i][0].case_id for i in chosen]
            for name,sets in trial['mediation'].items():
                assay=sets[kind];assert assay['common_ids']==expected and assay['common_eligible']==len(ids)
                assert assay['attempted_pairs']==len(objects) and assay['assayed_pairs']==len(chosen)
                for cell in assay['cells']:
                    assert len(cell['records'])==len(cell['patch_norms'])==cell['n']==len(chosen)
                    assert [r['pair_id'] for r in cell['records']]==expected
                    correct=0
                    for row in cell['records']:
                        actual=grade(pair_index[row['pair_id']][1],row['prediction'])[0]
                        assert actual==row['correct'];correct+=actual
                    assert correct==cell['correct'] and cell['fidelity']==(correct/cell['n'] if cell['n'] else None)
                    if cell['n'] and cell['method']=='sham':assert correct==0
                    if cell['n'] and cell['visit']==3 and cell['method']=='full_cls':assert correct==cell['n']
                    if cell['n'] and cell['visit']==3 and cell['method'] in ('head_fiber','native_vpd_write','random_fiber4'):assert correct==0
                    checks['mediation_cells_checked']+=1
                for visit in range(4):
                    rows={r['method']:r for r in assay['cells'] if r['visit']==visit}
                    for x,y in zip(rows['native_vpd_write']['patch_norms'],rows['random_fiber4']['patch_norms']):
                        assert abs(x-y)<=1e-5+1e-4*max(x,y)
    torch.set_num_threads(1);residuals=[]
    for trial,fr,warm in zip(frozen['models'],frozen['frames'],reg['warm_starts']):
        assert sha(args.private/fr['file'])==fr['sha256'];frames=torch.load(args.private/fr['file'],weights_only=True)
        old=args.old_private/warm['file'];assert sha(old)==warm['file_sha256']
        for name,info in trial['models'].items():
            path=resolve_model(info,args.private,args.old_private);assert sha(path)==info['file_sha256']
            model=load_model(path,c,'cpu');assert digest(model)==info['model_sha256'] and protected(model)==warm['protected_sha256']
            q=model.adapter.q;rng=torch.Generator().manual_seed(trial['seed']+207000)
            z=torch.randn(16,c['hidden'],generator=rng);z=z-(z@q)@q.T
            error=float((model.head[0].weight@z.T).abs().max());assert error<1e-4
            residuals.append({'seed':trial['seed'],'model':name,'terminal_fiber_first_readout_max':error})
            torch.testing.assert_close(frames[name]['random'],frames['warm']['random'],atol=0,rtol=0)
            checks['frozen_models_verified']+=1;del model
    counts=Counter();forward=Counter();by_stage={};frozen_seen=False;memory={'allocated_mb':0.,'reserved_mb':0.,'minimum_free_mb':float('inf')};oracle_calls=Counter()
    with (args.private/'work.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line);event=e['event'];counts[event]+=1
            if event=='all_weights_and_directions_frozen':
                assert e['freeze_sha256']==sha(out/'freeze.json');frozen_seen=True
            if event=='optimizer_completed':
                assert not frozen_seen
                s=by_stage.setdefault(e['stage'],Counter());s['updates']+=1
                expected=e['clean_nll']+c['clean_kl_weight']*e['clean_kl']+e['causal_weight']*e['causal_nll']+c['energy_weight']*e['write_energy']
                assert abs(expected-e['total'])<1e-4*(1+abs(expected))
            if event=='forward_called':
                assert e['batch']<=8 and e['tokens']==27 and e['visits']==4
                forward.update(model_calls=1,forward_examples=e['batch'],recurrent_example_visits=e['batch']*e['visits'],recurrent_token_visits=e['batch']*e['tokens']*e['visits'],readout_examples=e['batch'])
                if e['stage'].endswith('/train'):by_stage.setdefault(e['stage'],Counter())['forwards']+=1
            if event=='oracle_call':oracle_calls[e['name']]+=1
            if 'allocated_mb' in e:
                for key in ('allocated_mb','reserved_mb'):memory[key]=max(memory[key],e[key])
                if e.get('free_mb') is not None:memory['minimum_free_mb']=min(memory['minimum_free_mb'],e['free_mb'])
    assert counts['process_started']==counts['all_weights_and_directions_frozen']==counts['campaign_completed']==1 and counts['process_exception']==0
    for k,v in forward.items():assert done['resources']['work'][k]==v
    assert counts['optimizer_completed']==done['resources']['work']['adapter_optimizer_steps']
    for row in by_stage.values():assert row['forwards']==3*row['updates']
    for k,v in oracle_calls.items():assert done['oracle']['source_requests'][k]==v
    assert memory['reserved_mb']<=3000 and memory['minimum_free_mb']>=500
    latest=json.loads((args.private/'latest_state.json').read_text());assert latest['oracle']==done['oracle']
    result={'schema':'scm20.audit.v1','passed':True,'checks':dict(checks),'journal_events':dict(counts),
        'model_work_replayed':dict(forward),'training_stages':{k:dict(v) for k,v in by_stage.items()},'memory':memory,
        'terminal_fiber_checks':residuals,'official_roster_calls':0,'training_after_final':False,
        'verification_oracle_requests':dict(verification_requests),
        'verification_physical_caches':{k:v.cache_info()._asdict() for k,v in verification_caches.items()},
        'encoded_observation_overlap':encoded_overlap,
        'completed_sha256':sha(out/'completed.json'),'registration_commit':'179c01d'}
    with (out/'audit.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
