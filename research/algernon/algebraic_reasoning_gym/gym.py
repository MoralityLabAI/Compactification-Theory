"""ARG-01 Algernon: frozen recurrent class proposals against independent EvalMath.

A frozen 6,240-parameter *task-trained RRC-19 GRU* is the currently runnable
recurrent backbone. This does NOT represent an external foundation model or an
internal Qwen graft. Future Qwen/looped checkpoints use `external.py` receipts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time
from typing import Sequence

from spiral_ln import algebraic_reasoning_oracle as oracle
from research.rrc_compactification import learn18, model18

HERE=Path(__file__).resolve().parent
RRCHERE=HERE.parents[1]/'rrc_compactification'
CHECKPOINT_ROLES={
    'frozen_gru':'balanced_recurrent',
    'skewed_gru':'skewed_recurrent',
    'frozen_ff':'balanced_feedforward',
    'shuffled_gru':'shuffled_recurrent',
}
BUDGET=12
ARMS=('frozen_gru','skewed_gru','frozen_ff','shuffled_gru',
      'certified_mass12','certified_mass_full','exhaustive_64')
MODEL_MACS={'frozen_gru':53216,'skewed_gru':53216,'frozen_ff':6120,
            'shuffled_gru':53216}


def file_digest(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def load_pin():
    pin=json.loads((HERE/'evalmath_pin.json').read_text())
    if pin['schema']!='algernon.arg01.pin.v1':raise ValueError('pin schema drift')
    source=Path(oracle.__file__).resolve()
    if file_digest(source)!=pin['oracle_source_sha256']:
        raise AssertionError('EvalMath oracle source differs from frozen pin')
    if pin['oracle_version']!=oracle.VERSION:raise AssertionError('oracle interface drift')
    # A source-only local pin is explicit until the matching EvalMath commit
    # exists. A remote revision, when supplied, must be a full commit SHA.
    rev=pin.get('evalmath_commit_sha')
    if rev is not None and (not isinstance(rev,str) or len(rev)!=40 or any(c not in '0123456789abcdef' for c in rev)):
        raise AssertionError('invalid EvalMath repository revision')
    return pin


def checkpoint_path(role):
    if role not in CHECKPOINT_ROLES:raise ValueError('unsupported model role')
    return RRCHERE/'results'/f'rrc19_{CHECKPOINT_ROLES[role]}.json'


def checkpoints():
    pin=load_pin()
    saved={}
    for role in CHECKPOINT_ROLES:
        path=checkpoint_path(role)
        content=path.read_bytes()
        if sha256(content).hexdigest()!=pin['checkpoint_sha256'][role]:
            raise AssertionError(f'checkpoint differs: {role}')
        obj=json.loads(content)
        expected_kind='feedforward' if role=='frozen_ff' else 'recurrent'
        if obj['kind']!=expected_kind or obj['parameter_count']!=model18.parameter_count(obj['kind']):
            raise AssertionError(f'model schema / parameters differ: {role}')
        saved[role]=obj
    return saved


def compile_classes():
    # RRC-18 supplied exact degree-12 symbolic class compiler. This is an
    # explicit common resource, not part of the neural checkpoint.
    return learn18.class_map()


def model_order(case, checkpoint, classes):
    raw_logits=model18.logits(case,checkpoint)
    return learn18.candidates(raw_logits,case.topology,classes,count=BUDGET)


def order(case,arm,models,classes):
    if arm in CHECKPOINT_ROLES:
        return model_order(case,models[arm],classes)
    if arm in ('certified_mass12','certified_mass_full'):
        return learn18.mass_order(case.topology,classes)
    if arm=='exhaustive_64':return oracle.PROGRAMS
    raise ValueError('unknown experiment arm')


def score_case(case,arm,models,classes):
    start=time.perf_counter_ns()
    candidates=order(case,arm,models,classes)
    ordering_ns=time.perf_counter_ns()-start
    budget=len(candidates) if arm in ('certified_mass_full','exhaustive_64') else BUDGET
    start=time.perf_counter_ns()
    result=oracle.search(case,candidates,budget=budget)
    verification_ns=time.perf_counter_ns()-start
    quality=oracle.grade(case,result)
    return {'grade':quality,'trace':result['trace'],'queries':result['queries'],
            'integer_loop_proxy':result['integer_loop_proxy'],
            'model_macs':MODEL_MACS.get(arm,0),
            'order_ns':ordering_ns,'verify_ns':verification_ns}


def aggregate(rows):
    n=len(rows)
    if not n:raise ValueError('empty score group')
    keys=['accepted','future_correct','false_future_acceptance']
    sums={k:sum(z['grade'][k] for z in rows) for k in keys}
    return {'cases':n,**sums,'abstentions':n-sums['accepted'],
            'total_queries':sum(z['queries'] for z in rows),
            'total_demo_executions':2*sum(z['queries'] for z in rows),
            'total_integer_loop_proxy':sum(z['integer_loop_proxy'] for z in rows),
            'total_model_macs':sum(z['model_macs'] for z in rows)}


def source_hashes():
    paths=['gym.py','external.py','test_gym.py']
    return {path:file_digest(HERE/path) for path in paths}


def run(*,with_timing=False):
    pin=load_pin()
    st=time.perf_counter_ns()
    models=checkpoints()
    ck_load_ns=time.perf_counter_ns()-st
    st=time.perf_counter_ns()
    classes=compile_classes()
    compile_ns=time.perf_counter_ns()-st
    rosters=oracle.roster()
    public_collisions=[];scores={};traces={};times={}
    for split,cases in rosters.items():
        if split=='calibration':continue  # unused calibration remains blinded
        split_rows={arm:[] for arm in ARMS}
        trace_log={arm:[] for arm in ARMS}
        for case in cases:
            for arm in ARMS:
                result=score_case(case,arm,models,classes)
                split_rows[arm].append(result)
                trace_log[arm].append((case.identifier,[list(p) for p in result['trace']],result['grade']))
                if result['queries']>BUDGET and arm not in ('certified_mass_full','exhaustive_64'):
                    raise AssertionError('exact-verifier budget exceeded')
        scores[split]={arm:aggregate(rows) for arm,rows in split_rows.items()}
        if with_timing:times[split]={arm:{'median_ms':round(statistics.median((z['order_ns']+z['verify_ns'])/1e6 for z in rows),6),
                            'total_ms':round(sum((z['order_ns']+z['verify_ns'])/1e6 for z in rows),6),
                            'route_ms':round(sum(z['order_ns']/1e6 for z in rows),6),
                            'verify_ms':round(sum(z['verify_ns']/1e6 for z in rows),6)}
                            for arm,rows in split_rows.items()}
        traces[split]={arm:sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
                       for arm,rows in trace_log.items()}
        if split=='collision':
            for i in range(0,len(cases),2):
                left,right=cases[i:i+2]
                if left.signature!=right.signature or left.future==right.future:
                    raise AssertionError('paired collision contract')
                for arm in ARMS:
                    if trace_log[arm][i][1]!=trace_log[arm][i+1][1]:
                        raise AssertionError('future leak into model decisions')
                public_collisions.append(left.pair_id)
    positive=oracle.ALL_FAMILIES
    for split in positive:
        n=len(rosters[split])
        if scores[split]['exhaustive_64']['accepted']!=n:
            raise AssertionError('exact complete DSL search failed')
        if scores[split]['certified_mass_full']['accepted']!=n:
            raise AssertionError('symbolic classes disagree with independent EvalMath')
    if scores['negative']['exhaustive_64']['accepted']!=0:raise AssertionError('constructed negative invalid')
    if scores['collision']['exhaustive_64']['false_future_acceptance']!=len(public_collisions):
        raise AssertionError('paired future ambiguity not retained')
    result={
        'schema':'algernon.arg01.receipt.v1',
        'status':'CPU frozen task-trained recurrent benchmark; Qwen/looped model unrun',
        'model_type':'RRC-19 supervised 6,240-parameter GRU, trained before this evaluation',
        'source_sha256':source_hashes(),
        'evalmath_pin':pin,
        'roster_sha256':{k:oracle.hash_roster(v) for k,v in rosters.items()},
        'checkpoint_sha256':pin['checkpoint_sha256'],
        'settings':{'degree':oracle.PREFIX,'future_degree':oracle.FUTURE,'budget':BUDGET,
                    'training_updates':0,'model_candidate_previews':0,
                    'model_recurrence_depth_steps':13,
                    'actual_qwen_inference':False,
                    'candidate_feedback_reused_by_model':False,
                    'offline_class_compiler_programs':256,
                    'finite_prefix_pairs':len(public_collisions)},
        'scores':scores,'trace_sha256':traces,
        'scope':'A frozen RRC-19 recurrent checkpoint orders exact graph-class representatives. 13 GRU timesteps process public coefficients. Iterative candidate checks execute an external exact verifier; this is not a looped-layer Qwen graft. Exact equality is finite through degree 12; future outputs are graded separately.',
    }
    if with_timing:
        import platform
        result['physical_profile']={
            'python':platform.python_version(),'machine':platform.machine(),
            'processor':platform.processor(),
            'checkpoint_load_ms':round(ck_load_ns/1e6,6),
            'class_compile_ms':round(compile_ns/1e6,6),
            'per_split':times,
            'timing_scope':'single warm Python process with no independent evaluation grading charged; machine-specific elapsed wall ns, not an exact replay invariant',
        }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results/arg01_reference.json')
    p.add_argument('--check',action='store_true')
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    record=run(with_timing=args.profile)
    payload=json.dumps(record,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('ARG-01 source-bound receipt differs')
        print('ARG-01 frozen GRU vs exact EvalMath replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        print('saved',args.out)
        for k in oracle.ALL_FAMILIES:
            print(k,{arm:record['scores'][k][arm]['future_correct'] for arm in ARMS})

if __name__=='__main__':main()
