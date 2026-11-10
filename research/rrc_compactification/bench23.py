"""RRC-23: fixed-model scaling study, deterministic exact receipts and CPU clock profile.

Frozen RRC-21 router receives only degree-12 prefixes. Every expanded-horizon
candidate gets independently checked on both degree-T public demonstrations.
Timing is physical evidence and deliberately excluded from the exact CI receipt.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

try:
    from . import scale23 as s, bench22 as prev, learn18 as l, graph11 as g
    from . import meter22, topology16 as top, bench21
except ImportError:
    import scale23 as s
    import bench22 as prev
    import learn18 as l
    import graph11 as g
    import meter22
    import topology16 as top
    import bench21

HERE=Path(__file__).resolve().parent
SOURCE_NAMES=('scale23.py','bench23.py','bench22.py','bench21.py','meter22.py',
              'select21.py','risk21.py','learn18.py','model18.py','graph11.py',
              'graph12.py','graph13.py','graph14.py','topology15.py','topology16.py',
              'certify18.py','family19.py','family20.py','gate20.py')
ARM_SET=('symbolic_full','symbolic_progressive','selective_full','selective_progressive','syntax64')
EXT_ARMS=('extended_mass','extended_neural12','extended_selective','extended_syntax256')


def source_hashes():
    return {f:sha256((HERE/f).read_bytes()).hexdigest() for f in SOURCE_NAMES}


def checkpoints():
    models,cal,locks=bench21.checkpoints()
    return models,cal,locks


def orders(case,models,cal,classes):
    full=l.mass_order(case.topology,classes)
    chosen,budget,route,uses_model,deferred=prev.decide(case,'selective_full',models,cal,classes)
    return full,tuple(chosen),budget,route,uses_model,deferred


def choose(case,arm,models,cal,classes):
    if arm in ('symbolic_full','symbolic_progressive','syntax64'):
        order=g.GRAPHS if arm=='syntax64' else l.mass_order(case.topology,classes)
        return tuple(order),len(order),'symbolic',False
    if arm not in ARM_SET:raise ValueError('unknown arm')
    chosen,budget,route,uses_model,_=prev.decide(case,'selective_full',models,cal,classes)
    return tuple(chosen),budget,route,uses_model


def predict(case,arm,models,cal,classes,degree):
    order,budget,route,uses_model=choose(case,arm,models,cal,classes)
    out=s.verify(case,order,degree=degree,budget=budget,
                 progressive=arm.endswith('progressive'))
    return {**out,'route':route,'uses_model':uses_model}


def extended_predict(case,arm,models,cal,classes,degree):
    if arm not in EXT_ARMS:raise ValueError('unknown extended arm')
    mass=l.mass_order(case.topology,classes)
    if arm=='extended_syntax256':
        order=s.extended_order(g.GRAPHS,kind='mass');budget=len(order)
        neural=False;route='syntax256'
    elif arm=='extended_mass':
        order=s.extended_order(mass,kind='mass');budget=len(order)
        neural=False;route='mass4'
    else:
        proposals,_,route,neural=choose(case,'selective_full',models,cal,classes)
        if arm=='extended_neural12':
            # Preserve the only learned ranking and explore all four new prefix ops.
            logits=bench21.neural_logits(case,models)
            ranking=prev.r.order(case,logits,classes)
            order=s.extended_order(ranking,kind='round_robin');budget=12
            route='neural_fourway';neural=True
        else:
            # Full fallback expands its old certified class catalogue by 4.
            if route=='neural':
                order=s.extended_order(proposals,kind='round_robin');budget=12
            else:
                order=s.extended_order(mass,kind='mass');budget=len(order)
    out=s.verify(case,order,degree=degree,budget=budget)
    return {**out,'route':route,'uses_model':neural}


def aggregate(rows):
    n=len(rows)
    if not n:raise ValueError('empty sample')
    return {'cases':n,
            'accepted':sum(z['accepted'] for z in rows),
            'correct_future':sum(z['future_correct'] for z in rows),
            'false_accept':sum(z['false_accept'] for z in rows),
            'queries':sum(z['queries'] for z in rows),
            'demo_executions':sum(z['demo_executions'] for z in rows),
            'nominal_work':sum(z['nominal_work'] for z in rows),
            'model_cases':sum(z['uses_model'] for z in rows),
            'mean_queries':round(sum(z['queries'] for z in rows)/n,5),
            'mean_work':round(sum(z['nominal_work'] for z in rows)/n,3)}


def group_long_aliases(cases,classes,horizon):
    """Empirical stress of prior *degree-12* quotient under larger horizons.

    This is not a symbolic proof at a higher degree.
    """
    aliases=0;comparisons=0;examples=[]
    for topo in top.TOPOLOGIES:
        probes=[case.demos[0][0][:horizon+1] for case in cases if case.topology==topo][:4]
        for group in classes[topo]['classes']:
            if len(group)==1:continue
            for p in group[1:]:
                for probe in probes:
                    comparisons+=1
                    if s.execute(probe,p,topo)!=s.execute(probe,group[0],topo):
                        aliases+=1
                        if len(examples)<5:examples.append([topo,list(group[0]),list(p)])
    return {'candidate_equivalence_comparisons':comparisons,'disagreements':aliases,
            'examples':examples,'scope':'independent coefficient samples only; degree-12 class proof remains bounded'}


def deterministic_receipt(*,per_cell=s.PER_CELL,extended_per=s.EXTENDED_PER_TOPOLOGY,
                          horizons=s.HORIZONS,extended_degrees=s.EXTENDED_DEGREES):
    models,cal,locks=checkpoints()
    classes=l.class_map()
    originals=s.dataset(per_cell=per_cell)
    expanded=s.extended_dataset(per_topology=extended_per)
    results={};trace_hashes={};routes={};paired={};span={}
    for degree in horizons:
        bucket={arm:[] for arm in ARM_SET}
        traces={arm:[] for arm in ARM_SET}
        route_counts={arm:{} for arm in ARM_SET}
        for original in originals:
            case=s.observed(original,degree)
            for arm in ARM_SET:
                prediction=predict(case,arm,models,cal,classes,degree)
                evaluation=s.grade(case,prediction['program'])
                row={**prediction,**evaluation}
                bucket[arm].append(row)
                traces[arm].append((case.identifier,[list(p) for p in row['trace']],
                                    None if row['program'] is None else list(row['program'])))
                route_counts[arm][row['route']]=route_counts[arm].get(row['route'],0)+1
                if row['queries']!=len(row['trace']):raise AssertionError('query trace drift')
                if row['demo_executions']!=2*sum(row['stage_counts'].values()):raise AssertionError('free stage execution')
                if arm in ('symbolic_full','symbolic_progressive') and row['queries']>len(classes[case.topology]['classes']):
                    raise AssertionError('symbolic budget drift')
        for one,two in (('symbolic_full','symbolic_progressive'),('selective_full','selective_progressive')):
            if any(a['program']!=b['program'] or a['queries']!=b['queries']
                   for a,b in zip(bucket[one],bucket[two])):
                raise AssertionError('progressive parity failure')
        results[str(degree)]={arm:aggregate(rows) for arm,rows in bucket.items()}
        trace_hashes[str(degree)]={arm:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                                   for arm,v in traces.items()}
        routes[str(degree)]=route_counts
        # Record semantic forecast disagreements between certified T12 and all 64 spellings.
        paired[str(degree)]={'class_vs_64_accepted_mismatch':sum(a['accepted']!=b['accepted']
                     for a,b in zip(bucket['symbolic_full'],bucket['syntax64'])),
                     'class_vs_64_forecast_mismatch':sum(a['future_correct']!=b['future_correct']
                     for a,b in zip(bucket['symbolic_full'],bucket['syntax64']))}
        span[str(degree)]=group_long_aliases(originals,classes,degree)
    gram_results={};gram_traces={}
    for degree in extended_degrees:
        groups={arm:[] for arm in EXT_ARMS};traces={arm:[] for arm in EXT_ARMS}
        for original,pre in expanded:
            case=s.observed(original,degree)
            for arm in EXT_ARMS:
                item=extended_predict(case,arm,models,cal,classes,degree)
                grade=s.grade(case,item['program'])
                groups[arm].append({**grade,**item})
                traces[arm].append((case.identifier,[list(p) for p in item['trace']],
                                    None if item['program'] is None else list(item['program'])))
        if groups['extended_syntax256'] and aggregate(groups['extended_syntax256'])['accepted']!=len(expanded):
            raise AssertionError('expanded syntax search must cover every generated sample')
        gram_results[str(degree)]={arm:aggregate(rows) for arm,rows in groups.items()}
        gram_traces[str(degree)]={arm:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                                 for arm,v in traces.items()}
    # Second, post-primary long-horizon stress: just eight fresh cases.
    stressed=s.stress_dataset()
    stress_rows={arm:[] for arm in ARM_SET}
    for original in stressed:
        case=s.observed(original,s.STRESS_DEGREE)
        for arm in ARM_SET:
            output=predict(case,arm,models,cal,classes,s.STRESS_DEGREE)
            stress_rows[arm].append({**output,**s.grade(case,output['program'])})
    stress_result={arm:aggregate(rows) for arm,rows in stress_rows.items()}
    # Costed integer operand widths, sampled only; arithmetic parity checked separately.
    bit_sample={}
    for degree in horizons:
        selected=s.observed(originals[0],degree)
        meter=meter22.Meter()
        for f,y in selected.demos:
            result=meter22.execute(f,originals[0].hidden_graph,selected.topology,meter)
            if result!=y:raise AssertionError('meter parity failed')
        bit_sample[str(degree)]=meter.summary()
    return {
        'schema':'rrc23.deterministic.v1','status':'frozen_router_extended_horizon_and_syntax_cpu',
        'source_sha256':source_hashes(),'checkpoint_sha256':locks,
        'parent_rrc22_semantic_sha256':sha256((HERE/'results'/'rrc22_deterministic.json').read_bytes()).hexdigest(),
        'input_sha256':s.digest(originals),'expanded_input_sha256':s.digest([x for x,_ in expanded]),
        'parameters':{'seed':s.SEED,'horizons':list(horizons),'future_span':24,
                      'families':list(s.FAMILIES),'topologies':list(s.TOPOLOGIES),'per_cell':per_cell,
                      'cases':len(originals),'extended_degrees':list(extended_degrees),
                      'extended_cases':len(expanded),'extended_syntax_programs':len(s.PRE_OPS)*len(g.GRAPHS),
                      'fixed_frozen_model_observation_degree':12,'fixed_frozen_candidate_budget':12,
                      'candidate_execution_demos':2,'progressive_stages':'4,12,T with repeated charged executions',
                      'frozen_training_updates':0},
        'horizon_scores':results,'horizon_trace_sha256':trace_hashes,'horizon_routes':routes,
        'long_horizon_alias_stress':span,'paired_vs_full_syntax64':paired,
        'grammar_scores':gram_results,'grammar_trace_sha256':gram_traces,
        'metered_operand_sample':bit_sample,
        'long_horizon_stress':{'degree':s.STRESS_DEGREE,'cases':len(stressed),
                               'roster_sha256':s.digest(stressed),'scores':stress_result},
        'limitations':'Degree-12 certified equivalences are tested empirically at longer horizons. The four-token grammar is hand-supplied. Neural router and older OOD detector observe only degree 12; no new model training. Warm timing belongs in a separate machine profile.'}


def _stats(values):
    values=sorted(values)
    return {'median_ms':round(statistics.median(values)*1000,5),
            'p90_ms':round(values[min(len(values)-1,math.ceil(.9*len(values))-1)]*1000,5),
            'total_ms':round(sum(values)*1000,3)}


def profile_horizon(degree,cases,models,cal,classes,*,rounds=2):
    if rounds<1:raise ValueError('positive rounds')
    methods=('symbolic_full','symbolic_progressive','selective_full','selective_progressive','syntax64')
    answer={}
    for arm in methods:
        wall=[];routing=[];verify=[];queries=[];correct=0
        for c in cases[:min(6,len(cases))]:
            x=s.observed(c,degree);predict(x,arm,models,cal,classes,degree)
        for rep in range(rounds):
            ordering=list(cases)
            random.Random(s.SEED+degree*41+rep).shuffle(ordering)
            for original in ordering:
                case=s.observed(original,degree)
                start=time.perf_counter()
                order,budget,route,neural=choose(case,arm,models,cal,classes)
                split=time.perf_counter()
                found=s.verify(case,order,degree=degree,budget=budget,
                               progressive=arm.endswith('progressive'))
                end=time.perf_counter()
                wall.append(end-start);routing.append(split-start);verify.append(end-split)
                queries.append(found['queries'])
                correct+=s.grade(case,found['program'])['future_correct']
        answer[arm]={'end_to_end_wall':_stats(wall),'routing_wall':_stats(routing),
                     'verifier_wall':_stats(verify),'mean_graph_queries':round(statistics.mean(queries),5),
                     'future_correct_across_rounds':correct,'total_measurements':len(wall)}
    return answer


def profile_grammar(degree,cases,models,cal,classes,*,rounds=2):
    records={}
    for arm in EXT_ARMS:
        wall=[];routing=[];verification=[];cost=[]
        for original,pre in cases[:min(4,len(cases))]:
            item=s.observed(original,degree)
            extended_predict(item,arm,models,cal,classes,degree)
        for rep in range(rounds):
            order_cases=list(cases)
            random.Random(s.SEED+degree*103+rep).shuffle(order_cases)
            for original,pre in order_cases:
                item=s.observed(original,degree)
                started=time.perf_counter()
                # These calls intentionally time the complete method, including
                # any frozen model and the actual larger-program verification.
                result=extended_predict(item,arm,models,cal,classes,degree)
                ended=time.perf_counter()
                wall.append(ended-started)
                cost.append(result['nominal_work'])
        records[arm]={'end_to_end_wall':_stats(wall),
                      'mean_nominal_work':round(statistics.mean(cost),3),
                      'measurements':len(wall)}
    return records


def physical_profile(*,rounds=2,per_cell=s.PER_CELL,horizons=s.HORIZONS):
    start=time.perf_counter()
    classes=l.class_map()
    no_model_setup_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    models,cal,locks=checkpoints()
    model_setup_ms=(time.perf_counter()-start)*1000
    cases=s.dataset(per_cell=per_cell)
    records={str(degree):profile_horizon(degree,cases,models,cal,classes,rounds=rounds)
             for degree in horizons}
    # Physical syntax-256 measurements are a separate, declared small sample.
    # Complete 40-case correctness already appears in the deterministic receipt.
    extended_all=s.extended_dataset(per_topology=2 if per_cell==1 else s.EXTENDED_PER_TOPOLOGY)
    extended=tuple(next((x for x in extended_all if x[0].topology==topology and
                         x[0].identifier.endswith(f'-{index:03d}')), None)
                   for topology in s.TOPOLOGIES for index in range(2))
    if any(x is None for x in extended):
        # IDs are global within the 40-case roster, so select by topology.
        extended=tuple(x for topo in s.TOPOLOGIES for x in
                       [item for item in extended_all if item[0].topology==topo][:2])
    gram={str(degree):profile_grammar(degree,extended,models,cal,classes,rounds=1)
          for degree in (12,48)}
    stress=s.stress_dataset()
    long_profile=profile_horizon(s.STRESS_DEGREE,stress,models,cal,classes,rounds=1)
    return {'schema':'rrc23.physical.v1','machine':{'system':platform.system(),
                   'python':platform.python_version(),'processor':platform.processor(),
                   'cpu_model':next((line.split(':',1)[1].strip() for line in
                                Path('/proc/cpuinfo').read_text().splitlines()
                                if line.startswith('model name')),'unknown'),
                   'available_cpus':os.cpu_count()},
            'parameters':{'rounds':rounds,'per_cell':per_cell,'cases':len(cases),
                          'horizons':list(horizons),'warmup_cases_per_arm':min(6,len(cases)),
                          'physical_timing_not_replay_asserted':True,
                          'future_scoring_excluded_from_policy_clock':True},
            'source_sha256':source_hashes(),
            'setup':{'symbolic_class_compilation_ms':round(no_model_setup_ms,3),
                     'neural_checkpoint_and_calibration_load_ms':round(model_setup_ms,3)},
            'results':records,'expanded_grammar_timing_cases':len(extended),
            'expanded_grammar_timing_rounds':1,
            'expanded_grammar_results':gram,
            'long_horizon_stress':{'degree':s.STRESS_DEGREE,'cases':len(stress),
                                  'rounds':1,'results':long_profile},
            'scope':'Pure Python standard-library model inference. No raw instrumented meters in clock path. Warm, within-process sequential arm batches; variation possible.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--profile',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--per-cell',type=int,default=s.PER_CELL)
    p.add_argument('--extended-per',type=int,default=s.EXTENDED_PER_TOPOLOGY)
    p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    if args.profile or args.smoke:
        obj=physical_profile(rounds=1 if args.smoke else args.rounds,
                             per_cell=1 if args.smoke else args.per_cell,
                             horizons=(12,48) if args.smoke else s.HORIZONS)
        out=args.out
        if out:out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
        for h,block in obj['results'].items():
            print(h,{arm:rec['end_to_end_wall']['median_ms'] for arm,rec in block.items()})
        return
    obj=deterministic_receipt(per_cell=args.per_cell,extended_per=args.extended_per)
    text=json.dumps(obj,indent=2,sort_keys=True)+'\n'
    out=args.out or HERE/'results'/'rrc23_deterministic.json'
    if args.check:
        if not out.is_file() or out.read_text()!=text:raise SystemExit('RRC-23 deterministic receipt missing/stale')
        print('RRC-23 source-bound exact semantic and operation replay verified')
    else:
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text)
        print('wrote',out)
        print('horizon summary',[(h,{k:v['correct_future'] for k,v in block.items()})
                                for h,block in obj['horizon_scores'].items()])
        print('grammar summary',[(h,{k:v['correct_future'] for k,v in block.items()})
                                for h,block in obj['grammar_scores'].items()])

if __name__=='__main__':main()
