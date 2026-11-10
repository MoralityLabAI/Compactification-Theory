"""RRC-25: exact CPU replay, source-locked model proposals and real wall clock."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import statistics
import time

try:
    from . import compound25 as b, model25 as m, train25 as tr
except ImportError:
    import compound25 as b
    import model25 as m
    import train25 as tr

HERE=Path(__file__).resolve().parent
KINDS=('recurrent','feedforward','shuffled')
ARMS=('recurrent_1','recurrent_4','recurrent_16','feedforward_4','shuffled_4',
      'fixed_4','preview_1','preview_4','complete_64')
SOURCES=('compound25.py','model25.py','train25.py','bench25.py','delay24.py','graph11.py')


def hashes():return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCES}


def checkpoints():
    expected=tr.source_hashes()
    ck={};dig={}
    for kind in KINDS:
        path=HERE/'results'/f'rrc25_{kind}.json'
        raw=path.read_bytes();item=json.loads(raw)
        if item['kind']!=kind or item['parameters']!=m.parameters(kind):
            raise AssertionError('checkpoint shape or kind changed: '+kind)
        if item['source_sha256']!=expected or item['steps']!=tr.STEPS:
            raise AssertionError('checkpoint training source changed: '+kind)
        ck[kind]=item;dig[kind]=sha256(raw).hexdigest()
    return ck,dig


def predict(case,arm,models):
    if arm=='complete_64':return b.complete_search(case)
    kind='recurrent' if arm.startswith('recurrent') else 'feedforward' if arm.startswith('feedforward') else 'shuffled' if arm.startswith('shuffled') else None
    order='fixed' if arm.startswith('fixed') else 'preview' if arm.startswith('preview') else kind
    budget=int(arm.rsplit('_',1)[1])
    return b.search(case,order,models[kind] if kind else None,budget)


def summaries(rows):
    n=len(rows)
    if n<1:raise ValueError('empty benchmark group')
    return {'cases':n,'accepted':sum(z['accepted'] for z in rows),
        'future_correct':sum(z['future_correct'] for z in rows),
        'false_accept':sum(z['false_accept'] for z in rows),
        'abstentions':n-sum(z['accepted'] for z in rows),
        'mean_macro_checks':round(sum(z['full_checks'] for z in rows)/n,4),
        'mean_public_base_executions':round(sum(z['header_base_executions'] for z in rows)/n,4),
        'mean_paid_template_previews':round(sum(z['preview_pairs'] for z in rows)/n,4),
        'mean_nominal_work':round(sum(z['nominal_work'] for z in rows)/n,4),
        'mean_dense_model_macs':round(sum(z['model_macs'] for z in rows)/n,4)}


def score_case(case,arm,models):
    prediction=predict(case,arm,models)
    if prediction['full_checks']!=len(prediction['trace']):raise AssertionError('trace length mismatch')
    if len(set(prediction['trace']))!=len(prediction['trace']):raise AssertionError('duplicate paid template check')
    budget=int(arm.rsplit('_',1)[1])
    if prediction['full_checks']>budget:raise AssertionError('macro budget exceeded')
    assert 4<=prediction['header_base_executions']<=8
    result=b.grade(case,prediction['program'])
    if prediction['program'] is not None and not result['accepted']:
        raise AssertionError('unchecked program claimed')
    return {**result,**prediction,'nominal_work':b.total_work(prediction)}


def deterministic_receipt(*,test_per=b.TEST_PER):
    models,dig=checkpoints()
    groups=b.roster(test_per=test_per)
    seen={z.visible for z in groups['train']}
    aliases={k:sum(z.visible in seen for z in items) for k,items in groups.items() if k!='train'}
    if any(aliases.values()):raise AssertionError('train/test observed inputs aliased')
    scores={};traces={};pairs={};header_stats={};ambiguity={}
    for split,cases in groups.items():
        if split in ('train','dev'):continue
        per_arm={arm:[] for arm in ARMS};trace_log={arm:[] for arm in ARMS}
        header_stats[split]={str(k):sum(len(b.possible_headers(c))==k for c in cases) for k in range(5)}
        cardinalities=[]
        for case in cases:
            count=b.visible_template_count(case)
            cardinalities.append(count)
        ambiguity[split]={'unique_visible_programs':sum(z==1 for z in cardinalities),
                          'multiple_visible_programs':sum(z>1 for z in cardinalities),
                          'no_visible_programs':sum(z==0 for z in cardinalities),
                          'max_programs':max(cardinalities),
                          'distribution':{str(v):cardinalities.count(v) for v in sorted(set(cardinalities))}}
        for case in cases:
            for arm in ARMS:
                item=score_case(case,arm,models)
                per_arm[arm].append(item)
                trace_log[arm].append((case.identifier,item['trace'],item['program']))
        scores[split]={arm:summaries(rows) for arm,rows in per_arm.items()}
        traces[split]={arm:sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
                       for arm,values in trace_log.items()}
        if split=='collision':
            pairs={arm:{'identical_trace_and_proposal':all(trace_log[arm][i][1:]==trace_log[arm][i+1][1:]
                            for i in range(0,len(cases),2)),
                         'false_future_acceptances':scores[split][arm]['false_accept']}
                   for arm in ARMS}
    for family in b.TEST_FAMILIES:
        if scores[family]['complete_64']['accepted']!=len(groups[family]):
            raise AssertionError('complete factorized grammar missed positive '+family)
    if scores['negative']['complete_64']['accepted']!=0:
        raise AssertionError('invalid out-of-grammar negatives')
    if not all(x['identical_trace_and_proposal'] for x in pairs.values()):
        raise AssertionError('visible-equal future pair produced different policy traces')
    if pairs['complete_64']['false_future_acceptances']!=b.COLLISION_PAIRS:
        raise AssertionError('future-pair bound invalid')
    return {'schema':'rrc25.deterministic.v1','status':'finite_delayed_two_branch_composition_cpu',
       'source_sha256':hashes(),'checkpoint_sha256':dig,
       'roster_sha256':{k:b.fingerprint(v) for k,v in groups.items()},
       'visible_train_test_aliases':aliases,
       'parameters':{'seed':b.SEED,'observed_degree':b.OBSERVED,'future_degree':b.FUTURE,
           'model_input_window':b.WINDOW,'delays':list(b.DELAYS),'spacings':list(b.SPACINGS),
           'amplitudes':list(b.AMPLITUDES),'base_ops':list(b.OPS),
           'template_pairs':16,'full_literal_grammar':len(b.OPS)*len(b.DELAYS)*len(b.SPACINGS)*len(b.AMPLITUDES)*16,
           'training_examples':len(groups['train']),'development_examples':len(groups['dev']),
           'model_training_steps':tr.STEPS,'model_training_batch':tr.BATCH,
           'full_prefix_base_executions_upper_bound':8,
           'recurrent_parameters':m.parameters('recurrent'),
           'feedforward_parameters':m.parameters('feedforward'),
           'strict_macro_budget':4},
       'scores':scores,'header_counts':header_stats,'visible_program_ambiguity':ambiguity,
       'trace_sha256':traces,
       'future_pairs':pairs,
       'limitations':'Model receives a public exact header scan that already reveals base operation and both delay coordinates; it only ranks 16 U/V templates. The explicit interpreters, headers, exact previews, and trained models are different resources. Source family labels/future outputs are excluded from inference; finite prefix matching cannot prove unbounded equivalence.'}


def machine_profile(*,test_per=8,rounds=2):
    models,_=checkpoints();data=b.roster(test_per=test_per)
    cases=[z for fam in b.TEST_FAMILIES for z in data[fam][:min(4,test_per)]]
    arms=('recurrent_4','feedforward_4','fixed_4','preview_4','complete_64')
    results={}
    for arm in arms:
        elapsed=[];correct=0
        for case in cases[:2]:predict(case,arm,models)
        for run in range(rounds):
            batch=cases.copy();random.Random(b.SEED+run).shuffle(batch)
            for case in batch:
                t0=time.perf_counter_ns();response=predict(case,arm,models)
                elapsed.append((time.perf_counter_ns()-t0)/1e6)
                correct+=b.grade(case,response['program'])['future_correct']
        results[arm]={'cases_per_pass':len(cases),'passes':rounds,
             'successes_over_passes':correct,
             'median_wall_ms':round(statistics.median(elapsed),5),
             'total_wall_ms':round(sum(elapsed),4),
             'p90_wall_ms':round(sorted(elapsed)[int(.9*(len(elapsed)-1))],5)}
    return {'schema':'rrc25.machine_profile.v1',
            'python':platform.python_version(),
            'cpu':next((z.split(':',1)[1].strip() for z in Path('/proc/cpuinfo').read_text().splitlines() if z.startswith('model name')),'unknown'),
            'source_sha256':hashes(),'roster_count':len(cases),
            'profile':results,'measured_wall_times_not_replay_asserted':True,
            'scope':'Warm Python CPU inference and exact observed-prefix program checks. Training, roster construction and future grading excluded.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--test-per',type=int,default=b.TEST_PER)
    p.add_argument('--cases',type=int,default=8)
    p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    if args.profile:
        obj=machine_profile(test_per=args.cases,rounds=args.rounds)
        path=args.out or HERE/'results'/'rrc25_reference_profile.json'
    else:
        obj=deterministic_receipt(test_per=args.test_per)
        path=args.out or HERE/'results'/'rrc25_pilot.json'
    payload=json.dumps(obj,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not path.is_file() or path.read_text()!=payload:
            raise SystemExit('RRC-25 receipt missing/stale')
        print('RRC-25 complete exact receipt verified')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload)
        if args.profile:print({k:v['median_wall_ms'] for k,v in obj['profile'].items()})
        else:print({k:{a:obj['scores'][k][a]['future_correct'] for a in ARMS} for k in b.TEST_FAMILIES})

if __name__=='__main__':main()
