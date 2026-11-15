"""Independently replay recorded predictions, without retraining or selecting models."""
from __future__ import annotations
import argparse
import functools
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from ..core import HERE,PROGRAMS,features,load_model,sha,digest,canonical
from ..campaign import source_hashes,restore
from ...tasks import load_evalmath,MathTask


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);a=p.parse_args()
    out=HERE/'results';read=lambda name:json.loads((out/name).read_text())
    reg=read('registration.json');done=read('completed.json');freeze=read('freeze.json')
    assert reg['source_sha256']==source_hashes()
    assert sha(a.oracle)==reg['oracle_sha256']
    assert done['freeze_sha256']==sha(out/'freeze.json')
    assert freeze['registration_sha256']==sha(out/'registration.json')
    assert freeze['selection_sha256']==sha(out/'selection.json')
    assert done['journal_sha256']==sha(a.private/'work.jsonl')
    assert done['private_gold_sha256']==sha(a.private/'fresh.gold.private.json')
    checks=Counter();oracle=load_evalmath(a.oracle)
    # Cache pure integer functions only. The grader below is independent of the
    # campaign's grade_detail and score aggregation implementations.
    for name in ('unary','convolution','reciprocal','execute'):
        original=getattr(oracle,name)
        @functools.lru_cache(maxsize=32768)
        def cached(*args,_original=original):return _original(*args)
        def wrapped(*args,_cached=cached):return _cached(*(tuple(v) if isinstance(v,list) else v for v in args))
        setattr(oracle,name,wrapped)
    def forbidden(**kwargs):raise AssertionError('official roster is forbidden')
    oracle.roster=forbidden
    gold=json.loads((a.private/'fresh.gold.private.json').read_text())
    data=read('data.json');assert sha(a.private/'data.pt')==data['cache_sha256']
    split_signatures=[data['training_signatures'],*data['split_signatures'].values()]
    all_pre=[s for group in split_signatures for s in group]
    assert len(all_pre)==len(set(all_pre))
    assert set(all_pre)<=set(freeze['pre_final_signatures'])
    checks['train_dev_calibration_unique_signatures']=len(all_pre)
    public=read('fresh.public.json');cases={};signatures=[]
    for kind,rows in gold['clean'].items():
        tasks=restore(oracle,rows)
        assert [t.public for t in tasks]==public['clean'][kind]
        for t in tasks:cases[t.case_id]=t.oracle_case;signatures.append(t.signature)
    pairs={}
    for kind,rows in gold['mediation'].items():
        objects=[]
        for row in rows:
            t=restore(oracle,[row['recipient']])[0];graph=tuple(row['donor_graph']);case=t.oracle_case
            assert graph[1:]==t.source_graph[1:]
            assert graph[0]=={'I':'N','N':'S','S':'R','R':'I'}[t.source_graph[0]]
            donor=replace(case,identifier=case.identifier+'/donor',source_graph=graph,
                demos=tuple((x,oracle.execute(x,t.topology,graph)) for x,_ in case.demos),
                future=oracle.execute(case.query,t.topology,graph))
            other=MathTask(t.case_id+'/donor',t.family,t.topology,
                tuple((tuple(x[:13]),tuple(y[:13])) for x,y in donor.demos),t.query,
                donor.future,graph,'evalmath_arg01',donor)
            assert list(case.future)==row['recipient_future'] and list(donor.future)==row['donor_future']
            assert case.future!=donor.future and t.signature!=other.signature
            x,y=features(t),features(other)
            if done['selection']['encoding']=='coefficient':x,y=x[...,:28],y[...,:28]
            assert not x.equal(y)
            signatures.extend((t.signature,other.signature));pairs[t.case_id]=(case,donor)
            objects.append([t.public,other.public]);checks['semantic_pairs_reconstructed']+=1
        assert objects==public['mediation'][kind]
    assert len(signatures)==len(set(signatures))
    assert not set(signatures)&set(freeze['pre_final_signatures'])
    checks['fresh_unique_public_signatures']=len(signatures)
    def grade(case,pred):
        prefix=oracle.check_candidate(case,PROGRAMS[pred]);accepted=bool(prefix['accepted'])
        correct=bool(oracle.grade(case,{'graph':PROGRAMS[pred]})['future_correct']) if accepted else False
        checks['recorded_predictions_regraded']+=1
        return int(correct),int(accepted),int(accepted and not correct)
    for trial in done['clean']:
        for policy,sets in {**trial['scores'],'historical':trial['historical_reference']}.items():
            for kind,score in sets.items():
                actual=[]
                for row in score['records']:
                    value=grade(cases[row['id']],row['prediction']);assert value[0]==row['correct'];actual.append(value)
                    if 'prefix_accepted' in row:assert value[1:]==(row['prefix_accepted'],row['false_future_acceptance'])
                assert len(actual)==score['n'] and sum(v[0] for v in actual)==score['correct']
                assert score['accuracy']==score['correct']/score['n']
                assert score['inference_verifier_calls']==0
    for trial in done['mediation']:
        for policy,sets in trial['policies'].items():
            for kind,assay in sets.items():
                assert assay['assayed_pairs']<=64 and assay['assayed_pairs']<=assay['eligible_pairs_total']
                for cell in assay['records']:
                    n=cell['valid_counterfactual_tests'];correct=0
                    assert len(cell['outcomes'])==len(cell['patch_norms'])==n
                    for row in cell['outcomes']:
                        value=grade(pairs[row['pair_id']][1],row['prediction'])[0]
                        assert value==row['counterfactual_correct'];correct+=value
                    assert correct==cell['correct_counterfactual_outcomes']
                    assert cell['causal_fidelity']==(correct/n if n else None)
                    if n and cell['method']=='sham':assert correct==0
                    if n and cell['method']=='full_cls' and cell['visit']==3:assert correct==n
                    if n and cell['method'] in ('head_fiber','native_vpd_write','random_fiber4') and cell['visit']==3:assert correct==0
                    checks['mediation_cells_checked']+=1
                for visit in range(4):
                    rows={v['method']:v for v in assay['records'] if v['visit']==visit}
                    for x,y in zip(rows['native_vpd_write']['patch_norms'],rows['random_fiber4']['patch_norms']):
                        assert abs(x-y)<=1e-5+1e-4*max(x,y)
    c=reg['config'];numerical=[]
    import torch
    torch.set_num_threads(1)
    for trial,frame in zip(freeze['trials'],freeze['frames']):
        path=a.private/trial['checkpoint']['file'];assert sha(path)==trial['checkpoint']['file_sha256']
        assert sha(a.private/frame['file'])==frame['sha256']
        model=load_model(path,c,'cpu');assert digest(model)==trial['checkpoint']['model_sha256']
        assert digest(model.sae)==trial['checkpoint']['sae_sha256']
        base=digest(model,('degree_','positions','core.','head.'));assert base==trial['frozen_base_sha256']
        base_receipt=read(f"{done['selection']['encoding']}_s{trial['seed']}_steps{done['selection']['common_step_budget']}.json")
        base_path=a.private/base_receipt['checkpoint']['file'];assert sha(base_path)==base_receipt['checkpoint']['file_sha256']
        original=load_model(base_path,c,'cpu');assert digest(original,('degree_','positions','core.','head.'))==base
        del original
        q=model.adapter.q
        rng=torch.Generator().manual_seed(trial['seed']+999)
        z=torch.randn(16,c['hidden'],generator=rng);z=z-(z@q)@q.T
        error=float((model.head[0].weight@z.T).abs().max())
        assert error<1e-4
        numerical.append({'seed':trial['seed'],'terminal_first_readout_fiber_residual_max':error})
        checks['frozen_models_and_frames_verified']+=1
        del model
    counts=Counter();forward=Counter();memory={'allocated_mb':0.,'reserved_mb':0.,'minimum_free_mb':float('inf')}
    frozen_seen=False;frozen_events=0;readouts=0;gram=0
    with (a.private/'work.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line);kind=e['event'];counts[kind]+=1
            if kind=='all_weights_and_directions_frozen':
                assert e['freeze_sha256']==sha(out/'freeze.json')
                frozen_seen=True;frozen_events+=1
            if kind in ('optimizer_completed','sae_update','adapter_update'):assert not frozen_seen
            if kind=='forward_called':
                forward.update(model_calls=1,forward_examples=e['batch'],recurrent_example_visits=e['batch']*e['visits'],
                    recurrent_token_visits=e['batch']*e['tokens']*e['visits'])
                readouts+=e['batch']*(2 if e['stage'].startswith('clean_final_') and e['tokens']==27 else 1)
                if e['stage'].startswith('polynomial_gram_'):gram+=e['batch']*26*529
            if 'allocated_mb' in e:
                memory['allocated_mb']=max(memory['allocated_mb'],e['allocated_mb'])
                memory['reserved_mb']=max(memory['reserved_mb'],e['reserved_mb'])
                if e.get('free_mb') is not None:memory['minimum_free_mb']=min(memory['minimum_free_mb'],e['free_mb'])
    amendment=read('recovery_registration.json') if (out/'recovery_registration.json').exists() else None
    assert frozen_events==(2 if amendment else 1)
    if amendment:
        import hashlib
        with (a.private/'work.jsonl').open('rb') as stream:
            assert hashlib.sha256(stream.read(amendment['prior_journal_bytes'])).hexdigest()==amendment['prior_journal_sha256']
        assert amendment['recovery_source_sha256']==sha(HERE/'tools/resume.py')
    expected=done['resources']['work']
    for k,v in forward.items():assert expected[k]==v
    assert readouts==expected['readout_examples'] and gram==expected['public_gram_multiplications']
    for name in ('check_candidate','grade'):
        with (a.private/'work.jsonl').open() as stream:
            count=sum(1 for line in stream if (lambda e:e['event']=='oracle_call' and e['name']==name)(json.loads(line)))
        assert count==done['oracle']['source_requests'][name]
    for event,key in [('optimizer_completed','base_optimizer_steps'),('sae_update','sae_optimizer_steps'),('adapter_update','adapter_optimizer_steps')]:
        assert counts[event]==expected.get(key,0)
    assert memory['minimum_free_mb']>=500 and memory['reserved_mb']<=3000
    result={'schema':'scm19.independent_audit.v1','passed':True,'checks':dict(checks),'journal_events':dict(counts),
        'journal_forward_totals':dict(forward),'journal_memory':memory,'terminal_numerical_checks':numerical,
        'official_roster_calls':0,'training_after_final':False,'inference_verifier_calls':0,
        'registration_commit':'a94d749','completed_sha256':sha(out/'completed.json')}
    (out/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
