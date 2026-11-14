"""Durable recovery of the same frozen final matrix after host-tool interruption.

Uses the original evaluator unchanged, including batch-one FP32 inference and
paid candidate budgets. Only pure integer oracle.execute results are memoized;
logical requests, physical cache misses and cache hits are recorded separately.
Completed dataset/policy receipts survive a process/tool restart. No training,
new selection, direction fitting or oracle source edits occur here.
"""
import argparse
import functools
import json
import time
from collections import Counter
from pathlib import Path
import torch
from research.dimensional_atlas.gym.tasks import EXPECTED_EVALMATH_SHA256,load_evalmath,make_evalmath_case,verify_order
from research.dimensional_atlas.gym.scm16.campaign import load_public_evalmath
from research.dimensional_atlas.gym.scm18_rtx3050.core import (
    HERE,Meter,canonical,device_setup,digest,evaluate,load_checkpoint,pool,release,sha,write_new)
from research.dimensional_atlas.gym.scm18_rtx3050.campaign import OracleAudit,sources


def keep_existing(path,value):
    normalized=json.loads(json.dumps(value))
    if path.exists():
        assert json.loads(path.read_text())==normalized,'existing artifact differs: '+str(path)
    else:write_new(path,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True)
    args=parser.parse_args();out=HERE/'results';c=json.loads((HERE/'config.json').read_text())
    frozen=json.loads((out/'freeze.json').read_text());frozen_hash=sha(out/'freeze.json')
    assert not (out/'final.json').exists(),'completed final receipt already exists'
    assert sha(HERE/'config.json')==frozen['config_sha256']
    assert sha(out/'registration.json')==frozen['registration_sha256']
    assert sha(out/'training.json')==frozen['training_sha256']
    assert sources()==frozen['sources']
    assert sha(args.oracle)==frozen['oracle_sha256']==EXPECTED_EVALMATH_SHA256
    assert sha(args.private/'frame.pt')==frozen['frame_sha256']
    for ck in frozen['models']:assert sha(args.private/ck['file'])==ck['file_sha256']
    device=device_setup('cuda',c)
    oracle=load_evalmath(args.oracle)
    exact_execute=oracle.execute
    cached=functools.lru_cache(maxsize=65536)(exact_execute)
    oracle.execute=lambda a,topology,graph:cached(tuple(a),topology,tuple(graph))
    audit=OracleAudit(oracle)
    progress=args.private/'recovery_progress';progress.mkdir(exist_ok=True)
    source_hash=sha(__file__)
    keep_existing(out/'recovery_amendment.json',{
        'status':'performance_and_durability_only_after_interrupted_final_attempt',
        'freeze_sha256':frozen_hash,'driver_sha256':source_hash,
        'unchanged':['all models/dictionaries/frames','model/rank/objective selection','batch-one FP32 evaluator','case identities','paid verifier budgets','exact oracle source'],
        'changes':['memoize pure exact-integer execute results on CPU','persist each completed policy/dataset receipt'],
        'cache_limit_entries':65536,
        'interrupted_attempt':{'combined_receipt_written':False,
            'completed_checkpoint_summaries_observed':32,
            'total_attempted_forwards_and_verifier_calls':'unrecoverable_from_terminated_process',
            'interpretation':'This failed attempt is excluded from completed-matrix work totals; its additional compute is explicitly unaccounted, not zero.'}})
    # First official access follows the same complete freeze checks as the original.
    audit.allow_final=True
    official=load_public_evalmath(oracle);seen=set(frozen['all_pre_final_signatures'])
    assert not ({t.signature for t in official}&seen)
    fresh=pool(oracle,c['fresh_final_count'],1860000,'final',('rational','squared','lacunary','twisted_product'),seen)
    registration=json.loads((out/'registration.json').read_text());challenges=[]
    for i,row in enumerate(registration['mining']['selected']):
        for j in range(2):
            task=make_evalmath_case(oracle,1890000+i*100+j,case_id=f'challenge:{i}:{j}',
                family=row['family'],topology=row['topology'],graph=tuple(row['source_graph']))
            assert task.signature not in seen;seen.add(task.signature);challenges.append(task)
    keep_existing(args.private/'challenge.gold.private.json',[
        {'id':t.case_id,'graph':t.source_graph,'future':t.hidden_target} for t in challenges])
    keep_existing(out/'challenges.public.json',[t.public for t in challenges])
    results=[];all_deltas=Counter();reused=0;created=0
    all_cache_hits=all_cache_misses=0
    selected=[ck for ck in frozen['models'] if ck.get('untied') or ck['rank']==frozen['selected_rank']]
    for ix,ck in enumerate(selected):
        model,meta=load_checkpoint(args.private/ck['file'],c,device)
        assert digest(model)==ck['model_sha256']
        meter=Meter(device,c);model.meter=meter
        policies=(['tied'] if meta.get('untied') else ['joint'] if meta.get('random_dictionary') else
            ['fiber_only'] if meta['sector']=='fiber_only' else
            ['tied','joint','fiber_only','visible_only','shuffled_visit','zero_codes'])
        scores={};work=Counter();pieces=[]
        for policy in policies:
            scores[policy]={}
            for label,cases in [('official',official),('fresh',fresh),('challenges',challenges)]:
                key=canonical([frozen_hash,source_hash,ck['model_sha256'],policy,label,[t.signature for t in cases],c['budget']])
                path=progress/(key+'.json')
                if path.exists():
                    piece=json.loads(path.read_text());assert piece['key']==key;reused+=1
                else:
                    before=Counter(meter.counts);before_oracle=Counter(audit.counts);before_cache=cached.cache_info()
                    score=evaluate(model,cases,oracle,c['budget'],policy=policy)
                    after_cache=cached.cache_info()
                    piece={'key':key,'checkpoint':ck['file'],'policy':policy,'dataset':label,
                        'score':score,'work':dict(meter.counts-before),
                        'oracle_source_requests':dict(audit.counts-before_oracle),
                        'execute_cache_hits':after_cache.hits-before_cache.hits,
                        'execute_physical_misses':after_cache.misses-before_cache.misses,
                        'resources':meter.report()}
                    write_new(path,piece);created+=1
                scores[policy][label]=piece['score'];work.update(piece['work']);pieces.append(piece)
                all_deltas.update(piece['oracle_source_requests'])
                all_cache_hits+=piece['execute_cache_hits'];all_cache_misses+=piece['execute_physical_misses']
        assert digest(model)==ck['model_sha256']
        resource={'work':dict(work),'seconds':sum(v['score']['seconds'] for v in pieces),
            'peak_allocated_mb':max(v['resources']['peak_allocated_mb'] for v in pieces),
            'peak_reserved_mb':max(v['resources']['peak_reserved_mb'] for v in pieces),
            'minimum_sampled_free_mb':min(v['resources']['minimum_sampled_free_mb'] for v in pieces),
            'device':'cuda','memory_sampling':'every_model_call_and_dataset_receipt',
            'timing_note':'sum of persisted dataset evaluations; model loading and cache reuse excluded'}
        results.append({'checkpoint':ck,'policies':scores,'resources':resource})
        status={'completed_checkpoints':ix+1,'total_checkpoints':len(selected),'last_checkpoint':ck['file'],
                'new_dataset_receipts':created,'reused_dataset_receipts':reused}
        (progress/'status.json').write_text(json.dumps(status),encoding='utf-8')
        print(json.dumps(status),flush=True)
        del model,meter;release()
    symbolic={}
    for budget in (12,64):
        rows=[verify_order(t,list(range(64)),budget=budget,oracle=oracle) for t in official]
        symbolic[str(budget)]={'n':len(rows),'correct':sum(r['future_correct'] for r in rows),'paid_queries':sum(r['queries'] for r in rows)}
    write_new(out/'final.json',{'freeze_sha256':frozen_hash,'results':results,'symbolic':symbolic,
        'official_count':len(official),'official_unique_signatures':len({t.signature for t in official}),
        'fresh_count':len(fresh),'oracle_source_calls':dict(all_deltas),
        'challenge_gold_sha256':sha(args.private/'challenge.gold.private.json'),
        'recovery':{'amendment_sha256':sha(out/'recovery_amendment.json'),'driver_sha256':source_hash,
            'logical_evaluation_source_requests':dict(all_deltas),'evaluation_execute_cache_hits':all_cache_hits,
            'evaluation_execute_physical_misses':all_cache_misses,
            'this_process_all_source_requests_including_generation_and_symbolic':dict(audit.counts),
            'this_process_all_execute_cache':cached.cache_info()._asdict(),
            'dataset_receipts_new':created,'dataset_receipts_reused':reused,
            'interrupted_attempt_compute_incomplete':True}})
    print('RECOVERED_FINAL_COMPLETE',flush=True)


if __name__=='__main__':main()
