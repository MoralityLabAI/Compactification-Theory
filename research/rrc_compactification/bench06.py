"""RRC-06 frozen exact, raw-neural, negative and depth-ablation replay.

Only Python standard library needed. Neural checkpoints are inference-only;
all exact symbolic calculations are independent grading/verifier baselines.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

try:
    from .rrc06 import (PREFIX,HORIZON,SEED,SPLIT_SIZES,roster,roster_digest,
                        score_case,raw_features,teacher,certify,first_nonzero)
    from .raw_student06 import (infer_raw,parameter_count,DEFAULT_LOOPS)
    from .rrc05 import (exact_program,product_coefficients,
                        coefficient_residual,log_derivative_coefficients)
except ImportError:
    from rrc06 import (PREFIX,HORIZON,SEED,SPLIT_SIZES,roster,roster_digest,
                       score_case,raw_features,teacher,certify,first_nonzero)
    from raw_student06 import (infer_raw,parameter_count,DEFAULT_LOOPS)
    from rrc05 import (exact_program,product_coefficients,
                       coefficient_residual,log_derivative_coefficients)

HERE=Path(__file__).resolve().parent
CHECKPOINT_NAMES=('recurrent','feedforward','shuffled_recurrent')
SOURCE_NAMES=('rrc.py','program04.py','rrc05.py','rrc06.py',
              'raw_student06.py','train06.py','bench06.py','RRC06_PROTOCOL.md')
REGULAR_SPLITS=('dev','unseen_period','unseen_cap','local_gap',
                'nonperiodic','out_of_grammar','collision')


def hash_file(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def load_checkpoint(kind:str)->dict:
    fn=HERE/'results'/f'rrc06_{kind}.json'
    ckpt=json.loads(fn.read_text())
    if ckpt['schema']!='rrc06.raw_coefficient_student.v1' or ckpt['kind']!=kind:
        raise AssertionError('checkpoint identity mismatch')
    if ckpt['parameter_count']!=parameter_count(kind):
        raise AssertionError('incorrect model parameters')
    for name,digest in ckpt['source_sha256'].items():
        if hash_file(HERE/name)!=digest:raise AssertionError('changed training source: '+name)
    if ckpt['trained_test_data_access'] or ckpt['input_algebraic_preprocessing']!='none':
        raise AssertionError('non raw model checkpoint')
    if ckpt['optimization']['steps']!=500 or ckpt['optimization']['batch_size']!=64:
        raise AssertionError('unexpected training budget')
    return ckpt


def summary(records:list[dict])->dict:
    count=len(records)
    return {'cases':count,
            'a_coordinates_correct':sum(r['exact_a_coordinates'] for r in records),
            'a_coordinates_total':count*PREFIX,
            'b_coordinates_correct':sum(r['exact_b_coordinates'] for r in records),
            'b_coordinates_total':count*PREFIX,
            'mean_first_a_error_or_65':round(mean((r['first_a_error'] or 65)
                                                for r in records),4),
            'mean_first_b_error_or_65':round(mean((r['first_b_error'] or 65)
                                                for r in records),4),
            'verified_a_traces':sum(r['first_a_residual'] is None for r in records),
            'verified_b_traces':sum(r['first_b_residual'] is None for r in records),
            'verified_joint_traces':sum(r['joint_trace_verified'] for r in records),
            'accepted_programs':sum(r['prefix_verified'] for r in records),
            'correct_futures':sum(r['correct_future'] for r in records),
            'false_accepts':sum(r['false_accept'] for r in records)}


def oracle_result(case)->dict:
    program=exact_program(case.prefix)
    correct=int(program is not None and product_coefficients(HORIZON,program)==list(case.full))
    accepted=int(program is not None)
    return {'accepted_programs':accepted,'correct_futures':correct,
            'false_accepts':accepted-correct,
            'program':program.as_dict() if program else None}


def run()->dict:
    checkpoints={k:load_checkpoint(k) for k in CHECKPOINT_NAMES}
    splits=roster()
    if {k:len(v) for k,v in splits.items()}!=SPLIT_SIZES:
        raise AssertionError('roster size changed')
    if any(roster_digest(splits[k])!=checkpoints['recurrent'][key]
           for k,key in (('train','train_roster_sha256'),('dev','dev_roster_sha256'))):
        raise AssertionError('training roster source changed')
    train_hashes={case.prefix_hash() for case in splits['train']}
    overlap={name:len(train_hashes.intersection(c.prefix_hash() for c in splits[name]))
             for name in REGULAR_SPLITS}
    collisions=splits['collision']
    for i in range(0,len(collisions),2):
        a,b=collisions[i:i+2]
        if a.prefix!=b.prefix or a.full==b.full or a.collision_group!=b.collision_group:
            raise AssertionError('invalid future collision pair')
        k=next(n for n in range(PREFIX+1,HORIZON+1) if a.full[n]!=b.full[n])
        if k!=(67,79,101,113,127)[i//2]:raise AssertionError('mutation degree mismatch')
    metrics={arm:{} for arm in ('exact_symbolic','recurrent_2','feedforward',
                                'shuffled_recurrent_2')}
    cases=[]
    for name in REGULAR_SPLITS:
        rows={arm:[] for arm in metrics}
        for case in splits[name]:
            exact=oracle_result(case)
            data={'id':case.name,'family':case.family,'prefix_hash':case.prefix_hash(),
                  'oracle':exact}
            rows['exact_symbolic'].append(exact)
            for arm,model in (('recurrent_2','recurrent'),('feedforward','feedforward'),
                              ('shuffled_recurrent_2','shuffled_recurrent')):
                logits=infer_raw(case.prefix,checkpoints[model])
                record=score_case(case.prefix,case.full,logits)
                rows[arm].append(record)
                data[arm]={k:v for k,v in record.items() if k!='program'}
            cases.append(data)
        metrics['exact_symbolic'][name]={
            'cases':len(rows['exact_symbolic']),
            'accepted_programs':sum(r['accepted_programs'] for r in rows['exact_symbolic']),
            'correct_futures':sum(r['correct_futures'] for r in rows['exact_symbolic']),
            'false_accepts':sum(r['false_accepts'] for r in rows['exact_symbolic'])}
        for arm in ('recurrent_2','feedforward','shuffled_recurrent_2'):
            metrics[arm][name]=summary(rows[arm])
    depth={}
    for loops in (0,1,2,4,8):
        measurements=[]
        for case in splits['unseen_period']:
            logits=infer_raw(case.prefix,checkpoints['recurrent'],loops=loops)
            measurements.append(score_case(case.prefix,case.full,logits))
        depth[str(loops)]=summary(measurements)
    # Prefix-only policy returns identical outputs for matched collision inputs.
    for i in range(0,len(collisions),2):
        for kind in CHECKPOINT_NAMES:
            a,b=collisions[i:i+2]
            if infer_raw(a.prefix,checkpoints[kind])!=infer_raw(b.prefix,checkpoints[kind]):
                raise AssertionError('model leaked hidden source label into inference')
    # Exact series reference is independently consistent with both residuals.
    rr=[c for c in splits['local_gap'] if c.name in ('gap2-min1','gap2-min2')]
    rr_reference=[]
    for c in rr:
        a,b=teacher(c.prefix)
        cert=certify(c.prefix,a,b)
        if not cert['verified_joint_trace'] or cert['program'] is None:
            raise AssertionError('RR positive control failed')
        rr_reference.append({'id':c.name,'program':cert['program'].as_dict()})
    trace=json.dumps(cases,sort_keys=True,separators=(',',':')).encode()
    return {
        'schema':'rrc06.raw_to_intermediate.cpu.v1',
        'status':'trained_cpu_exact_finite_degree',
        'train_seed':SEED,'degree_prefix':PREFIX,'degree_horizon':HORIZON,
        'source_sha256':{p:hash_file(HERE/p) for p in SOURCE_NAMES},
        'checkpoint_sha256':{k:hash_file(HERE/'results'/f'rrc06_{k}.json')
                             for k in CHECKPOINT_NAMES},
        'roster_sha256':{k:roster_digest(v) for k,v in splits.items()},
        'split_sizes':{k:len(v) for k,v in splits.items()},
        'training_prefix_aliases':overlap,
        'model_resources':{k:{'fp32_parameters':checkpoint['parameter_count'],
                              'fp32_weight_bits':checkpoint['weight_bits_fp32'],
                              'live_hidden_floats':checkpoint['live_state_fp32'],
                              'trained_passes':checkpoint['training_depth']}
                          for k,checkpoint in checkpoints.items()},
        'metrics':metrics,
        'recurrent_unseen_period_depth':depth,
        'exact_rr_controls':rr_reference,
        'late_defect_pair_count':len(collisions)//2,
        'case_trace_sha256':sha256(trace).hexdigest(),
        'case_trace_count':len(cases),
        'selected_cases':[row for row in cases if row['id'] in
                          ('gap2-min1','gap2-min2','out-00','late-67-base','late-67-mutant')],
        'claims_boundary':'finite exact certificates; raw input neural model, trained on exact labels only; no infinite-identity proof',
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc06_pilot.json')
    args=p.parse_args()
    data=run()
    serialized=json.dumps(data,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=serialized:
            raise SystemExit('source/checkpoint-bound RRC-06 receipt drift')
        print('RRC-06 exact source/checkpoint-bound CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(serialized)
        print(json.dumps({'out':str(args.out),'split_sizes':data['split_sizes'],
                          'unseen_period_scores':{k:v['unseen_period'].get('accepted_programs')
                           for k,v in data['metrics'].items()},
                          'depth_scores':{k:v['accepted_programs'] for k,v in data['recurrent_unseen_period_depth'].items()}},indent=2,sort_keys=True))

if __name__=='__main__':main()
