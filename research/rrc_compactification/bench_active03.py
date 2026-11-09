"""Source-bound RRC-03 CPU replay across query budgets and unseen hypothesis counts.

No torch import or model training in the replay path. The query interface
returns only binary comparisons, with exact transcript accounting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .active03 import (HERE, STUDENT_H_SIZES, DEFAULT_HYPOTHESES,
                           MAX_QUERIES, baseline, candidate_defect, run_policy,
                           midpoint_policy, static_policy, adaptive_success_bound,
                           static_success_bound, experiment_without_students)
    from .student03 import QueryStudent, PARAMETERS, HIDDEN, INPUT, PACKED
except ImportError:
    from active03 import (HERE, STUDENT_H_SIZES, DEFAULT_HYPOTHESES,
                          MAX_QUERIES, baseline, candidate_defect, run_policy,
                          midpoint_policy, static_policy, adaptive_success_bound,
                          static_success_bound, experiment_without_students)
    from student03 import QueryStudent, PARAMETERS, HIDDEN, INPUT, PACKED

SOURCE_FILES = ('rrc.py','rrc02.py','active03.py','student03.py',
                'train_active03.py','bench_active03.py')
WEIGHT_FILES = ('results/rrc03_recurrent.json','results/rrc03_feedforward.json')
SIZES = tuple(sorted(set(STUDENT_H_SIZES+(DEFAULT_HYPOTHESES,))))


def to_bytes(data) -> bytes:
    return json.dumps(data,sort_keys=True,separators=(',',':')).encode()


def blinded(policy):
    """Keep the query schedule but censor each previous binary reply from policy.

    The evaluator and final MAP decoder still see true replies. The bound for
    a fixed schedule is B+1 transcript classes under ordered threshold tests.
    """
    def act(history,size,budget):
        return policy(tuple((query,0) for query,response in history),size,budget)
    return act


def run_replay() -> dict:
    models={name:QueryStudent.load(HERE / file)
            for name,file in (('recurrent',WEIGHT_FILES[0]),('feedforward',WEIGHT_FILES[1]))}
    policies={'adaptive_midpoint':midpoint_policy,'fixed_thresholds':static_policy,
              'recurrent':models['recurrent'].action,
              'feedforward':models['feedforward'].action,
              'recurrent_feedback_censored':blinded(models['recurrent'].action),
              'feedforward_feedback_censored':blinded(models['feedforward'].action)}
    exact=experiment_without_students()
    rows=[]
    for size in SIZES:
        for budget in range(MAX_QUERIES+1):
            results={}
            for name,policy in policies.items():
                traces=[run_policy(i,size,budget,policy) for i in range(size)]
                correct=sum(t.exact for t in traces)
                transcript=[{'candidate':t.actual,'predicted':t.predicted,
                             'observations':t.observations,'posterior':t.posterior}
                            for t in traces]
                results[name]={
                    'correct':correct, 'accuracy_numerator':correct,
                    'accuracy_denominator':size,
                    'distinct_transcripts':len({t.observations for t in traces}),
                    'total_queries':sum(t.queries_used for t in traces),
                    'wasted_queries':sum(t.wasted_queries for t in traces),
                    'trace_sha256':hashlib.sha256(to_bytes(transcript)).hexdigest(),
                }
            if results['adaptive_midpoint']['correct'] != adaptive_success_bound(size,budget):
                raise AssertionError('adaptive teacher lost its exact optimum')
            if results['fixed_thresholds']['correct'] != static_success_bound(size,budget):
                raise AssertionError('non-adaptive policy missed its sharp capacity')
            if results['recurrent_feedback_censored']['correct'] > static_success_bound(size,budget):
                raise AssertionError('feedback-blind recurrent violated transcript bound')
            if results['feedforward_feedback_censored']['correct'] > static_success_bound(size,budget):
                raise AssertionError('feedback-blind feedforward violated transcript bound')
            rows.append({'size':size,'budget':budget,
                         'adaptive_upper_bound_correct':adaptive_success_bound(size,budget),
                         'nonadaptive_upper_bound_correct':static_success_bound(size,budget),
                         'policies':results})
    roster={str(s):[candidate_defect(i,s) for i in range(s)] for s in SIZES}
    result={
        'schema':'rrc03.active-student.v1','status':'frozen_cpu_imitation_pilot',
        'train_sizes':models['recurrent'].checkpoint['training']['train_sizes'],
        'heldout_sizes':list(STUDENT_H_SIZES),
        'hypotheses_by_size':roster,
        'query_interface':'one binary bit indicating queried coefficient exceeds known baseline',
        'known_base_product':'parts congruent to 1 or 4 modulo 5; includes part 1',
        'oracle_experiment':exact,
        'model_parameters':{name:mod.checkpoint['parameter_count'] for name,mod in models.items()},
        'training':{name:mod.checkpoint['training'] for name,mod in models.items()},
        'policies':rows,
        'resource_accounting':{
            'weight_bits_fp32_each_model':PARAMETERS*32,
            'hidden_state_bits_fp32_recurrent':HIDDEN*32,
            'replayed_history_bits_for_max_six_query_records':MAX_QUERIES*(6+1),
            'fixed_packed_history_features_fp32':PACKED,
            'query_bit_cost_per_observation':1,
            'recurrent_macs_four_queries_streaming_approx':sum(HIDDEN+HIDDEN*(INPUT+HIDDEN)*t for t in range(4)),
            'feedforward_macs_four_queries_approx':4*(HIDDEN*PACKED+HIDDEN),
            'excluded_resources':['exact posterior decoder', 'baseline coefficient table',
                                  'integer polynomial algorithms','checkpoint loader',
                                  'program code','training optimizer states'],
        },
        'evidence_scope':'finite uniform-prior binary-threshold task, supervised optimal-policy imitation; no new infinite identity, generic theorem discovery or raw-coefficient neural training',
    }
    result['sources_sha256']={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest()
                              for p in SOURCE_FILES+WEIGHT_FILES}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results/rrc03_pilot.json')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    result=run_replay()
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.exists() or args.out.read_text()!=payload:
            raise SystemExit('RRC-03 replay mismatch or missing receipt')
        print('RRC-03 source-and-weight-bound replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        print(f'wrote {args.out}')


if __name__=='__main__':main()
