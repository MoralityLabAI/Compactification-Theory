"""RRC-24 exact long-prefix search receipt and separately measured CPU timings."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
import os
import platform
from pathlib import Path
import random
import statistics
import time

try:
    from . import delay24 as d, model24 as m, train24 as tr
except ImportError:
    import delay24 as d
    import model24 as m
    import train24 as tr

HERE=Path(__file__).resolve().parent
MODEL_KINDS=('recurrent','feedforward','shuffled')
ARMS=('recurrent_1','recurrent_2','recurrent_4','feedforward_2','shuffled_2',
      'static_2','prefix_exact_2','prefix_exact_4','exhaustive_4')
SOURCE_FILES=('delay24.py','model24.py','train24.py','bench24.py','graph11.py')


def hashes():return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCE_FILES}


def load_models():
    locked=tr.source_hashes();models={};digests={}
    for kind in MODEL_KINDS:
        path=HERE/'results'/f'rrc24_{kind}.json'
        raw=path.read_bytes();obj=json.loads(raw)
        if obj['kind']!=kind or obj['parameter_count']!=m.params(kind):raise AssertionError('checkpoint identity or dimension')
        if obj['source_sha256']!=locked or obj['steps']!=tr.STEPS:
            raise AssertionError('training source mismatch')
        models[kind]=obj;digests[kind]=sha256(raw).hexdigest()
    return models,digests


def infer(case,arm,models):
    if arm.startswith('recurrent'):order=m.order(case,models['recurrent'])
    elif arm=='feedforward_2':order=m.order(case,models['feedforward'])
    elif arm=='shuffled_2':order=m.order(case,models['shuffled'])
    elif arm=='static_2' or arm=='exhaustive_4':order=d.OPS
    elif arm.startswith('prefix_exact'):order=d.base_order_exact_prefix(case)
    else:raise ValueError(arm)
    budget=int(arm.rsplit('_',1)[1])
    return order,budget


def predict(case,arm,models):
    order,budget=infer(case,arm,models)
    out=d.search(case,order,budget)
    out['model_macs']=(m.macs('recurrent') if arm.startswith('recurrent') else
                       m.macs('feedforward') if arm=='feedforward_2' else
                       m.macs('shuffled') if arm=='shuffled_2' else 0)
    out['prefix_queries']=8 if arm.startswith('prefix_exact') else 0
    out['prefix_work']=sum(2*d.operation_work(op,d.MODEL_PREFIX) for op in d.OPS) if out['prefix_queries'] else 0
    return out


def flat_catalogue(order):
    return tuple((op,k,c) for op in order for k in d.DELAYS for c in d.AMPLITUDES)+tuple((op,None,0) for op in order)


def flat_search(case,order,*,budget=12,degree=d.OBSERVED):
    """Literal 132-program search, every attempted stage execution is charged."""
    if budget<1:raise ValueError('invalid syntax budget')
    stages=(d.MODEL_PREFIX,64,128,degree)
    trace=[];execs=0;work=0;found=None;stage_counts={str(k):0 for k in stages}
    for program in flat_catalogue(order):
        if len(trace)>=budget:break
        trace.append(program)
        good=True
        for stage in stages:
            for f,y in case.demos:
                x=d.execute(f[:stage+1],program) if program[1] is None or program[1]<=stage else d.operate(f[:stage+1],program[0])
                if tuple(x)!=tuple(y[:stage+1]):good=False
                execs+=1
            stage_counts[str(stage)]+=1
            work+=2*d.operation_work(program[0],stage)
            if not good:break
        if good:found=program;break
    return {'program':found,'queries':len(trace),'trace':tuple(trace),
            'demo_stage_executions':execs,'nominal_work':work,'stage_counts':stage_counts}


def summarize(rows):
    n=len(rows)
    if not n:raise ValueError('empty results')
    sums={k:sum(row[k] for row in rows) for k in
          ('accepted','future_correct','false_accept','queries','approx_ops','model_macs','prefix_work')}
    return {'cases':n,**sums,
            'mean_queries':round(sums['queries']/n,4),
            'mean_exact_work':round((sums['approx_ops']+sums['prefix_work'])/n,3),
            'mean_model_macs':round(sums['model_macs']/n,3)}



def extra_seed_checks(models):
    """Post-primary, frozen-weight robustness batches; test-only generation."""
    output={}
    for offset in (16000, 27000):
        results={name:{arm:0 for arm in ('recurrent_2','feedforward_2','static_2','prefix_exact_2','exhaustive_4')}
                 for name in d.TEST_FAMILIES}
        hashes={}
        for idx,family in enumerate(d.TEST_FAMILIES):
            rng=random.Random(d.SEED+offset+idx*777)
            cases=[d._sample_case(rng,family,f'rep-{offset}-{family}-{i:03d}') for i in range(24)]
            hashes[family]=d.fingerprint(cases)
            for case in cases:
                for arm in results[family]:
                    z=predict(case,arm,models)
                    results[family][arm]+=d.grade(case,z['program'])['future_correct']
            if results[family]['exhaustive_4']!=24:
                raise AssertionError('complete factorized search misses replication')
        output[str(d.SEED+offset)]={'cases_per_family':24,'roster_sha256':hashes,'correct_future':results}
    return output

def deterministic_receipt():
    models,digests=load_models()
    groups=d.dataset()
    tr_inputs={x.visible for x in groups['train']}
    aliases={k:sum(x.visible in tr_inputs for x in v) for k,v in groups.items() if k!='train'}
    if any(aliases.values()):raise AssertionError('source leaked or train/test collided')
    scores={};traces={};pairing={};by_case={};eligible=[]
    for split,cases in groups.items():
        if split in ('train','dev'):continue
        rows={arm:[] for arm in ARMS};logs={arm:[] for arm in ARMS}
        for case in cases:
            for arm in ARMS:
                outcome=predict(case,arm,models)
                grade=d.grade(case,outcome['program'])
                if outcome['queries']!=len(outcome['trace']):raise AssertionError('query trace mismatch')
                if outcome['queries']>int(arm.rsplit('_',1)[1]):raise AssertionError('budget overrun')
                if outcome['approx_ops']!=sum(2*d.operation_work(op,d.OBSERVED) for op in outcome['trace']):
                    raise AssertionError('nominal work mismatch')
                if outcome['program'] is not None and not grade['accepted']:
                    raise AssertionError('unchecked proposal accepted')
                rows[arm].append({**grade,**outcome})
                logs[arm].append((case.identifier,list(outcome['trace']),outcome['program']))
        scores[split]={arm:summarize(arr) for arm,arr in rows.items()}
        traces[split]={arm:sha256(json.dumps(arr,separators=(',',':')).encode()).hexdigest() for arm,arr in logs.items()}
        if split=='collision':
            pairing={arm:{'identical_visible_proposals':all(logs[arm][i][1:]==logs[arm][i+1][1:]
                            for i in range(0,len(logs[arm]),2)),
                'false_future_acceptances':scores[split][arm]['false_accept']}
                for arm in ARMS}
        if split=='signed':
            by_case={'recurrent_only_vs_ff':sum(a['future_correct'] and not b['future_correct'] for a,b in
                      zip(rows['recurrent_2'],rows['feedforward_2'])),
                      'feedforward_only_vs_recurrent':sum(b['future_correct'] and not a['future_correct'] for a,b in
                      zip(rows['recurrent_2'],rows['feedforward_2']))}
    for split in d.TEST_FAMILIES:
        if scores[split]['exhaustive_4']['accepted']!=len(groups[split]):
            raise AssertionError('full factorized search missed positive')
    if scores['negative']['exhaustive_4']['accepted']!=0:raise AssertionError('invalid out-of-grammar roster')
    if scores['collision']['exhaustive_4']['false_accept']!=d.COLLISION_PAIRS:
        raise AssertionError('collision information gate')
    if not all(z['identical_visible_proposals'] for z in pairing.values()):raise AssertionError('future leakage')
    # One complete syntactic catalogue check on a fixed 12-case subset, with
    # candidate count 132. This is an explicitly higher-resource diagnostic.
    sampled=tuple(groups['signed'][:4]+groups['product'][:4]+groups['late_source'][:4])
    syntactic={}
    for budget in (12,132):
        arr=[]
        for case in sampled:
            out=flat_search(case,d.OPS,budget=budget)
            grade=d.grade(case,out['program'])
            arr.append({**grade,**out})
        syntactic[str(budget)]={'cases':len(sampled),'accepted':sum(z['accepted'] for z in arr),
            'future_correct':sum(z['future_correct'] for z in arr),
            'total_syntax_queries':sum(z['queries'] for z in arr),
            'total_stage_demo_executions':sum(z['demo_stage_executions'] for z in arr),
            'total_nominal_work':sum(z['nominal_work'] for z in arr),
            'trace_sha256':sha256(json.dumps([[list(x) for x in z['trace']] for z in arr],separators=(',',':')).encode()).hexdigest()}
    if syntactic['132']['accepted']!=len(sampled):raise AssertionError('complete literal DSL failed')
    return {'schema':'rrc24.deterministic.v1','status':'exact_late_divergence_with_learned_base_ranking',
            'source_sha256':hashes(),'checkpoint_sha256':digests,
            'roster_sha256':{k:d.fingerprint(v) for k,v in groups.items()},
            'train_test_aliases':aliases,
            'configuration':{'seed':d.SEED,'model_input_degree':d.MODEL_PREFIX,'observed_degree':d.OBSERVED,
                'future_degree':d.FUTURE,'delays':list(d.DELAYS),'amplitudes':list(d.AMPLITUDES),
                'base_ops':list(d.OPS),'full_syntax_candidates':4*(len(d.DELAYS)*len(d.AMPLITUDES)+1),
                'train_families':list(d.TRAIN_FAMILIES),'test_families':list(d.TEST_FAMILIES),
                'train_examples':len(groups['train']),'dev_examples':len(groups['dev']),
                'base_macro_budget':4,'strict_budget':2,'flat_catalogue_case_sample':len(sampled)},
            'scores':scores,'trace_sha256':traces,'paired_future_collisions':pairing,
            'paired_signed_results':by_case,'flat_syntax_diagnostic':syntactic,
            'post_primary_robustness':extra_seed_checks(models),
            'claims_boundary':'Learned model ranks ONLY the four base operators using coefficients through degree 16. A separate exact suffix solver recovers K,c from degree-192 public data; it is a privileged test-time algebraic method. Verified observed prefixes do not prove unseen future coefficients.'}


def physical_profile(*,num_cases=32,rounds=2):
    """Actual warm CPU timing; no exact nanosecond thresholds in CI."""
    models,_=load_models();groups=d.dataset()
    cases=tuple((groups['signed']+groups['product']+groups['late_source'])[:num_cases])
    # Spread across generator classes, with a fixed slice per family.
    n=max(1,num_cases//3)
    cases=tuple((groups['signed'][:n]+groups['product'][:n]+groups['late_source'][:n]))
    result={};arms=('recurrent_2','feedforward_2','static_2','prefix_exact_2','prefix_exact_4','exhaustive_4')
    for arm in arms:
        timings=[];router=[];verifier=[];correct=0
        for c in cases[:6]:predict(c,arm,models)
        for rep in range(rounds):
            batch=list(cases);random.Random(d.SEED+rep*199).shuffle(batch)
            for c in batch:
                a=time.perf_counter_ns();order,budget=infer(c,arm,models);b=time.perf_counter_ns()
                out=d.search(c,order,budget);e=time.perf_counter_ns()
                timings.append((e-a)/1e6);router.append((b-a)/1e6);verifier.append((e-b)/1e6)
                correct+=d.grade(c,out['program'])['future_correct']
        result[arm]={'evaluations':len(timings),'correct_futures_over_runs':correct,
            'median_total_ms':round(statistics.median(timings),4),
            'p90_total_ms':round(sorted(timings)[min(len(timings)-1,math.ceil(.9*len(timings))-1)],4),
            'sum_total_ms':round(sum(timings),3),
            'median_route_ms':round(statistics.median(router),4),
            'median_exact_ms':round(statistics.median(verifier),4)}
    return {'schema':'rrc24.machine_profile.v1',
        'platform':platform.platform(),'python':platform.python_version(),
        'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        'cases':len(cases),'rounds':rounds,'timings':result,
        'timing_scope':'Warm Python CPU, two demo factorized checks; model and exact routing cost included. No separate future grader or training in timing.',
        'physical_times_are_machine_specific':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true')
    group.add_argument('--profile',action='store_true')
    parser.add_argument('--cases',type=int,default=32)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.profile:
        rec=physical_profile(num_cases=args.cases,rounds=args.rounds)
        out=args.out or HERE/'results'/'rrc24_reference_profile.json'
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:v['median_total_ms'] for k,v in rec['timings'].items()},indent=2));return
    rec=deterministic_receipt();txt=json.dumps(rec,sort_keys=True,indent=2)+'\n'
    out=args.out or HERE/'results'/'rrc24_pilot.json'
    if args.check:
        if not out.is_file() or out.read_text()!=txt:raise SystemExit('RRC-24 deterministic receipt drift')
        print('RRC-24 exact receipt verified')
    else:
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(txt)
        print(json.dumps({k:{arm:rec['scores'][k][arm]['future_correct'] for arm in ARMS}
                         for k in (*d.TEST_FAMILIES,'collision','negative')},indent=2))
if __name__=='__main__':main()
