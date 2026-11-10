"""RRC-22: deterministic correctness/operation receipts and physical CPU profiling.

Commands (repository root):
  python research/rrc_compactification/bench22.py --check
  python research/rrc_compactification/bench22.py --profile --out /tmp/rrc22_cpu.json
  python research/rrc_compactification/bench22.py --smoke --cases 4

Only deterministic semantic receipts are compared byte-for-byte in CI.
Wall-clock timings/RSS are recorded as environment-specific observations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

try:
    from . import bench21 as b, select21 as d, risk21 as r, learn18 as l
    from . import topology16 as t, meter22, model18 as m
except ImportError:
    import bench21 as b
    import select21 as d
    import risk21 as r
    import learn18 as l
    import topology16 as t
    import meter22
    import model18 as m

HERE=Path(__file__).resolve().parent
SOURCE_FILES=('bench22.py','meter22.py','bench21.py','risk21.py','select21.py',
              'train21.py','learn18.py','model18.py','gate20.py','family19.py',
              'family20.py','topology16.py','topology15.py','certify18.py',
              'graph11.py','graph12.py','graph13.py','graph14.py',
              'bench19.py','train19.py')
ARMS=('neural12','selective12','selective_full','ood_only_full','full_certified')
EXPERIMENT_GROUPS=d.KNOWN+d.NOVEL+('negative','collision')
POSITIVE_GROUPS=d.KNOWN+d.NOVEL


def source_hashes():
    return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCE_FILES}


def setup(arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    a=time.perf_counter_ns()
    # Symbolic-only execution never deserializes model checkpoints or calibration.
    models=cal=None
    if arm!='full_certified':
        models,cal,_=b.checkpoints()
    classes=l.class_map()
    data=d.dataset()
    cases=[c for key in EXPERIMENT_GROUPS for c in data[key]]
    setup_ns=time.perf_counter_ns()-a
    return models,cal,classes,cases,setup_ns,data


def decide(case,arm,models,cal,classes):
    """Reproduce RRC-21 deployable routing, with strict OOD short circuit.

    Search never sees future data/hidden graph and receives a fixed ordered list.
    """
    if arm=='full_certified':
        order=l.mass_order(case.topology,classes)
        return order,len(order),'full',False,False
    if arm=='neural12':
        logit=b.neural_logits(case,models)
        return r.order(case,logit,classes),d.BUDGET,'neural',True,False
    distance,p=r.calibrated_distance(cal,case)
    if arm in ('selective_full','selective12') and p<=cal['ood_alpha']:
        order=l.mass_order(case.topology,classes)
        budget=len(order) if arm=='selective_full' else d.BUDGET
        return order,budget,'ood',False,True
    if arm=='ood_only_full' and p<=cal['ood_alpha']:
        order=l.mass_order(case.topology,classes)
        return order,len(order),'ood',False,True
    logit=b.neural_logits(case,models)
    if arm in ('selective_full','selective12'):
        h=r.entropy(case,logit,classes)
        if h>cal['entropy_threshold']:
            order=l.mass_order(case.topology,classes)
            budget=len(order) if arm=='selective_full' else d.BUDGET
            return order,budget,'entropy',True,True
    return r.order(case,logit,classes),d.BUDGET,'neural',True,False


def evaluate(case,arm,models,cal,classes):
    order,budget,route,uses_model,deferred=decide(case,arm,models,cal,classes)
    result=l.search(case,order,budget=budget)
    return {**result,'route':route,'uses_model':uses_model,'deferred':deferred}


def expected_scores():
    return json.loads((HERE/'results'/'rrc21_pilot.json').read_text())


def digest(rows):
    return sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def deterministic_receipt():
    models,cal,classes,cases,_,data=setup('selective_full')
    expected=expected_scores()
    counters={a:{key:{'cases':0,'accepted':0,'future_correct':0,'false_accept':0,
                      'queries':0,'model_cases':0,'deferred':0,'integer_loop_proxy':0,
                      'ops':{'multiplications':0,'additions':0,'sign_changes':0,
                             'copies':0,'integer_comparisons':0,
                             'mul_operand_bit_product':0,'add_operand_max_bits':0},
                      'max_operand_bits':0,'max_intermediate_bits':0,
                      'comparison_max_bits':0} for key in EXPERIMENT_GROUPS} for a in ARMS}
    traces={a:[] for a in ARMS}
    max_source_bits=0
    split_labels=[key for key in EXPERIMENT_GROUPS for _ in data[key]]
    for split,case in zip(split_labels,cases):
        max_source_bits=max(max_source_bits,*(v.bit_length() for f,_ in case.demos for v in f[:d.PREFIX+1]))
        for arm in ARMS:
            if arm=='full_certified':
                output=evaluate(case,arm,None,None,classes)
            else:output=evaluate(case,arm,models,cal,classes)
            path=output['trace']
            metered,arith=meter22.metered_trace(case,path)
            if metered!=output['graph']:
                raise AssertionError('instrumented integer interpreter disagrees with standard verifier')
            if output['integer_loop_proxy']!=sum(2*t.graph_cost(graph,case.topology) for graph in path):
                raise AssertionError('declared nominal integer work drift')
            grade=t.certificate(case,output['graph'])
            metrics=counters[arm][split]
            for key in ('accepted','future_correct','false_accept'):
                metrics[key]+=grade[key]
            metrics['cases']+=1
            metrics['queries']+=len(path)
            metrics['integer_loop_proxy']+=output['integer_loop_proxy']
            metrics['model_cases']+=int(output['uses_model'])
            metrics['deferred']+=int(output['deferred'])
            for key in metrics['ops']:
                metrics['ops'][key]+=arith.summary()[key]
            for key in ('max_operand_bits','max_intermediate_bits','comparison_max_bits'):
                metrics[key]=max(metrics[key],arith.summary()[key])
            traces[arm].append((case.identifier,[list(graph) for graph in path],
                                None if output['graph'] is None else list(output['graph']),output['route']))
    for arm in ARMS:
        for fam in EXPERIMENT_GROUPS:
            actual=counters[arm][fam]
            ref=expected['scores'][fam][{'neural12':'neural12','selective12':'selective12',
                 'selective_full':'selective_full','ood_only_full':'ood_only_full',
                 'full_certified':'full_certified'}[arm]]
            if (actual['accepted'],actual['future_correct'],actual['false_accept'])!=(ref['accepted'],ref['future_correct'],ref['false_future_acceptance']):
                raise AssertionError(f'RRC-21 score parity failed: {arm}/{fam}')
            if actual['model_cases']>actual['cases'] or actual['deferred']>actual['cases']:
                raise AssertionError('invalid resource case counts')
    for a in ARMS:
        if counters[a]['collision']['false_accept']>d.COLLISION_PAIRS:
            raise AssertionError('future collision mismatch')
    return {'schema':'rrc22.deterministic.v1','status':'instrumented_exact_cpu_semantic_receipt',
            'source_sha256':source_hashes(),
            'parent_receipt_sha256':sha256((HERE/'results'/'rrc21_pilot.json').read_bytes()).hexdigest(),
            'calibration_sha256':sha256((HERE/'results'/'rrc21_calibration.json').read_bytes()).hexdigest(),
            'input_sha256':{k:d.digest(v) for k,v in data.items()},
            'configuration':{'degree':d.PREFIX,'future_degree':d.HORIZON,
                             'strict_budget':d.BUDGET,'experiment_groups':list(EXPERIMENT_GROUPS),
                             'candidate_comparisons_per_query':2*(d.PREFIX+1),
                             'max_source_coefficient_bits':max_source_bits},
            'per_arm_family':counters,
            'trace_sha256':{k:digest(v) for k,v in traces.items()},
            'meter_definition':{'mul_operand_bit_product':'sum(bit_length(left)*bit_length(right))',
                 'add_operand_max_bits':'sum(max(bit_length(left),bit_length(right)))',
                 'scope':'Measured on all executed degree-12 candidate graph intermediates and coefficients, excluding Python runtime and training.'},
            'limits':'Existing RRC-21 roster is reused for systems metering. Exact bit-width counts are operation proxies, not measured CPU cycles. Future query execution used only for independent grading.'}


def machine():
    cpumodel='unknown'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpumodel=line.split(':',1)[1].strip();break
    except OSError:pass
    return {'system':platform.system(),'machine':platform.machine(),
            'processor_model':cpumodel,'python':platform.python_version(),
            'cpu_count_available':os.cpu_count(),'pid':os.getpid(),
            'environment':{'PYTHONHASHSEED':os.getenv('PYTHONHASHSEED','unset'),
                           'OPENBLAS_NUM_THREADS':os.getenv('OPENBLAS_NUM_THREADS','unset')},
            'timing_clock':'time.perf_counter_ns',
            'memory_clock':'resource.RUSAGE_SELF.ru_maxrss (Linux KiB); tracemalloc Python allocations'}


def percentile_ns(values,fraction):
    seq=sorted(values)
    index=(len(seq)-1)*fraction
    lo=math.floor(index);hi=math.ceil(index)
    return int(round(seq[lo]+(seq[hi]-seq[lo])*(index-lo)))


def profile_arm(arm,cases,models,cal,classes,*,rounds=3,seed=2026111022):
    if rounds<1:raise ValueError('rounds')
    # Every approach sees the identical set of cases and gets a cold-cache warmup.
    for c in cases[:min(12,len(cases))]:evaluate(c,arm,models,cal,classes)
    wall=[];cpu=[];stage_route=[];stage_verify=[];checks=[]
    for rep in range(rounds):
        ordering=list(range(len(cases)))
        random.Random(seed+rep).shuffle(ordering)
        for idx in ordering:
            c=cases[idx]
            t0=time.perf_counter_ns();u0=time.process_time_ns()
            order,budget,route,uses_model,deferred=decide(c,arm,models,cal,classes)
            t1=time.perf_counter_ns()
            found=l.search(c,order,budget=budget)
            t2=time.perf_counter_ns();u2=time.process_time_ns()
            stage_route.append(t1-t0);stage_verify.append(t2-t1)
            wall.append(t2-t0);cpu.append(u2-u0)
            checks.append(len(found['trace']))
    def stats(x):return {'median_ms':round(statistics.median(x)/1e6,6),
                         'p90_ms':round(percentile_ns(x,.90)/1e6,6),
                         'p99_ms':round(percentile_ns(x,.99)/1e6,6),
                         'total_ms':round(sum(x)/1e6,3)}
    return {'arm':arm,'rounds':rounds,'cases_per_round':len(cases),'n_measurements':len(wall),
            'end_to_end_wall':stats(wall),'end_to_end_process_cpu':stats(cpu),
            'route_and_neural_wall':stats(stage_route),'exact_verifier_wall':stats(stage_verify),
            'mean_exact_checks':round(statistics.mean(checks),6)}


def current_rss_kib():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except OSError:pass
    return None


def memory_worker(arm,*,cases_limit=None):
    models,cal,classes,cases,setup_ns,_=setup(arm)
    if cases_limit is not None:cases=cases[:cases_limit]
    base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    initial_resident=current_rss_kib()
    n_traced=min(len(cases),64)
    chosen={int(j*(len(cases)-1)/max(1,n_traced-1)) for j in range(n_traced)}
    tracemalloc.start()
    for idx in sorted(chosen):evaluate(cases[idx],arm,models,cal,classes)
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for idx,case in enumerate(cases):
        if idx not in chosen:evaluate(case,arm,models,cal,classes)
    final_resident=current_rss_kib()
    return {'arm':arm,'case_count':len(cases),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'peak_rss_before_loop_kib':base,'resident_after_setup_kib':initial_resident,
            'resident_after_run_kib':final_resident,'peak_traced_python_bytes':peak,'tracemalloc_sample_cases':len(chosen),
            'setup_ms':round(setup_ns/1e6,3)}


def physical_profile(*,smoke=False,rounds=3,arms=ARMS):
    # Setup time measured separately for each arm, and memory in independent processes.
    elapsed_start=time.perf_counter_ns()
    run={}
    for arm in arms:
        models,cal,classes,cases,setup_ns,_=setup(arm)
        if smoke:cases=cases[:4]
        timing=profile_arm(arm,cases,models,cal,classes,rounds=(1 if smoke else rounds))
        timing['cold_setup_ms']=round(setup_ns/1e6,3)
        run[arm]=timing
    memory={}
    for arm in arms:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--memory-worker',arm]
        if smoke:cmd+=['--cases','4']
        output=subprocess.run(cmd,capture_output=True,text=True,check=True,env={**os.environ,'PYTHONHASHSEED':'0'},timeout=120)
        memory[arm]=json.loads(output.stdout)
    return {'schema':'rrc22.physical_profile.v1','machine':machine(),
            'protocol':{'primary_cases':480,'total_cases_including_controls':508,
                        'rounds':1 if smoke else rounds,'warmup_cases':12,'tracemalloc_memory_sample_max_cases':64,
                        'measurement_type':'real elapsed/CPU clocks, independent-process peak RSS, traced Python peak',
                        'physical_times_not_CI_asserted':True},
            'source_sha256':source_hashes(),
            'parent_receipt_sha256':sha256((HERE/'results'/'rrc21_pilot.json').read_bytes()).hexdigest(),
            'timings':run,'memory':memory,
            'total_profile_wall_ms':round((time.perf_counter_ns()-elapsed_start)/1e6,3),
            'interpretation':'Machine-dependent reference timings; serialized class compilation/model loading is separately timed, inference warm. Grading/future oracle is excluded from timed path; one arm per process for RSS. Measured unit cost and end-to-end timing depend on the environment.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true')
    group.add_argument('--profile',action='store_true')
    group.add_argument('--smoke',action='store_true')
    group.add_argument('--memory-worker',choices=ARMS)
    p.add_argument('--cases',type=int)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.memory_worker:
        print(json.dumps(memory_worker(args.memory_worker,cases_limit=args.cases)))
        return
    if args.profile or args.smoke:
        record=physical_profile(smoke=args.smoke,rounds=args.rounds,arms=tuple(args.arms))
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
        print(json.dumps({a:{'wall_median_ms':z['end_to_end_wall']['median_ms'],
                              'queries_mean':z['mean_exact_checks'],
                              'rss_kib':record['memory'][a]['peak_rss_kib']}
                          for a,z in record['timings'].items()},indent=2))
        return
    output=deterministic_receipt()
    payload=json.dumps(output,sort_keys=True,indent=2)+'\n'
    path=args.out or HERE/'results'/'rrc22_deterministic.json'
    if args.check:
        if not path.is_file() or path.read_text()!=payload:
            raise SystemExit('RRC-22 semantic operation receipt missing/stale')
        print('RRC-22 exact semantic and integer-bit audit verified')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload)
        print('RRC-22 semantic receipt saved',path)

if __name__=='__main__':main()
