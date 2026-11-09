"""RRC-05 CPU/stdlib replay of exact algebraic certificates and learned gains.

Existing RRC-04 raw-student checkpoints are read only. The RRC-05 students
receive extra exact b_n/n structure and a wired divisor graph, and their
performance is not a matched-information improvement over raw RRC-04 students.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

try:
    from .rrc05 import (PREFIX,HORIZON,PERIOD_CAP,roster,roster_digest,
        log_derivative_coefficients,learned_iteration,feedforward_estimate,
        jacobi_exact,candidate_from_estimate,exact_program,product_coefficients,
        proper_divisors)
    from .student04 import ProgramStudent
except ImportError:
    from rrc05 import (PREFIX,HORIZON,PERIOD_CAP,roster,roster_digest,
        log_derivative_coefficients,learned_iteration,feedforward_estimate,
        jacobi_exact,candidate_from_estimate,exact_program,product_coefficients,
        proper_divisors)
    from student04 import ProgramStudent

HERE=Path(__file__).resolve().parent
DEPTHS=(0,1,2,4,6,7,8)
GROUPS=('dev','unseen_modulus','local_gap','nonperiodic','out_of_grammar','collision')
NEW_VARIANTS=('learned_recurrent','structural_jacobi','shuffled_labels',
              'divisor_erased','one_pass_feedforward','raw_rrc04_recurrent',
              'raw_rrc04_feedforward','exact_symbolic')
SOURCE_FILES=('rrc.py','program04.py','student04.py','rrc05.py','train05.py','bench05.py',
              'test_rrc05.py','RRC05_THEORY.md','RRC05_PROTOCOL.md')
CHECKPOINTS=('rrc05_recurrent.json','rrc05_feedforward.json',
             'rrc05_shuffled_recurrent.json','rrc04_recurrent.json',
             'rrc04_feedforward.json')


def sha_file(p:Path)->str:
    return sha256(p.read_bytes()).hexdigest()


def one_case(case,program, *, intermediate_exact:bool|None=None,
             first_residual:int|None=None)->dict:
    if program is None:
        return {'accepted':False,'correct_future':False,
                'first_future_mismatch':None,'syntax_bits':None,
                'program':None,'intermediate_exact':intermediate_exact,
                'first_residual':first_residual}
    generated=product_coefficients(HORIZON,program)
    if generated[:PREFIX+1]!=list(case.prefix):
        raise AssertionError('unverified proposed program crossed boundary')
    first=next((n for n in range(PREFIX+1,HORIZON+1)
                if generated[n]!=case.full[n]),None)
    return {'accepted':True,'correct_future':first is None,
            'first_future_mismatch':first,'syntax_bits':program.syntax_bits,
            'program':program.as_dict(),'intermediate_exact':intermediate_exact,
            'first_residual':first_residual}


def score(items:list[dict])->dict:
    n=len(items)
    accepted=sum(x['accepted'] for x in items)
    correct=sum(x['correct_future'] for x in items)
    return {'cases':n,'prefix_verified':accepted,'correct_future':correct,
            'false_accepts':accepted-correct,'abstentions':n-accepted,
            'intermediate_exact':sum(x['intermediate_exact'] is True for x in items),
            'mean_grammar_bits_accepted':round(
                sum(x['syntax_bits'] for x in items if x['accepted'])/accepted,4)
                if accepted else None}


def evaluate(*, include_trace:bool=False)->dict:
    split=roster()
    train_signatures={c.prefix_hash() for c in split['train']}
    alias={g:sum(x.prefix_hash() in train_signatures for x in split[g])
           for g in GROUPS}
    checkpoint_path={k:HERE/'results'/k for k in CHECKPOINTS}
    checkpoints={k:json.loads(p.read_text()) for k,p in checkpoint_path.items()
                 if k.startswith('rrc05')}
    for k,data in checkpoints.items():
        if data.get('schema')!='rrc05.two_scalar.v1' or data['parameter_count']!=2:
            raise AssertionError(f'invalid student checkpoint {k}')
        if data['train_roster_sha256']!=roster_digest(split['train']) or \
                data['dev_roster_sha256']!=roster_digest(split['dev']):
            raise AssertionError(f'training roster drift: {k}')
        if data['source_sha256']!={s:sha_file(HERE/s) for s in ('rrc05.py','train05.py')}:
            raise AssertionError(f'training source drift: {k}')
    old_models={k:ProgramStudent.load(HERE/'results'/f'rrc04_{k}.json')
                for k in ('recurrent','feedforward')}
    evidence=[]
    by_variant={name:defaultdict(list) for name in NEW_VARIANTS}
    by_variant_depth={str(depth):defaultdict(list) for depth in DEPTHS}
    for group in GROUPS:
        for case in split[group]:
            b=log_derivative_coefficients(case.prefix)
            row={'id':case.name,'group':group,'family':case.family,
                 'prefix_sha256':case.prefix_hash(),'collision_group':case.collision_group,
                 'predictions':{}}
            oracle=exact_program(case.prefix)
            entry=one_case(case,oracle)
            by_variant['exact_symbolic'][group].append(entry)
            row['predictions']['exact_symbolic']=entry
            for depth in DEPTHS:
                actual=jacobi_exact(b,depth)
                structural=candidate_from_estimate(case.prefix,b,actual)
                p=one_case(case,structural['program'],intermediate_exact=structural['exact_intermediate'],
                           first_residual=structural['first_series_residual'])
                by_variant_depth[str(depth)]['structural_jacobi'].append((group,p))
                row['predictions'][f'jacobi_{depth}']=p
                ck=checkpoints['rrc05_recurrent.json']
                x=learned_iteration(b,depth,ck['alpha'],ck['beta'])
                pred=candidate_from_estimate(case.prefix,b,x)
                q=one_case(case,pred['program'],intermediate_exact=pred['exact_intermediate'],
                           first_residual=pred['first_series_residual'])
                by_variant_depth[str(depth)]['learned_recurrent'].append((group,q))
                row['predictions'][f'learned_{depth}']=q
            variant_models={
                'shuffled_labels':('rrc05_shuffled_recurrent.json',False),
                'divisor_erased':('rrc05_recurrent.json',True),
                'one_pass_feedforward':('rrc05_feedforward.json',None),
            }
            for v,(filename,erase) in variant_models.items():
                checkpoint=checkpoints[filename]
                x=(feedforward_estimate(b,checkpoint['alpha'],checkpoint['beta'])
                   if erase is None else
                   learned_iteration(b,7,checkpoint['alpha'],checkpoint['beta'],
                                     erase_proper=erase))
                pred=candidate_from_estimate(case.prefix,b,x)
                res=one_case(case,pred['program'],intermediate_exact=pred['exact_intermediate'],
                             first_residual=pred['first_series_residual'])
                by_variant[v][group].append(res)
                row['predictions'][v]=res
            for v,key in (('raw_rrc04_recurrent','recurrent'),
                          ('raw_rrc04_feedforward','feedforward')):
                raw=old_models[key].predict(case.prefix,depth=2)
                # Existing RRC-04 grammar period<=12 only; this is a historical
                # lower-information comparator, not an RRC-05 trained network.
                program=None
                if hasattr(raw,'cap'):
                    # Imported here to keep package/script parity.
                    try:
                        from .rrc05 import Program05
                    except ImportError:
                        from rrc05 import Program05
                    program=Program05(raw.cap,raw.period,raw.residues)
                    if product_coefficients(PREFIX,program)!=list(case.prefix):
                        program=None
                pred=one_case(case,program)
                by_variant[v][group].append(pred)
                row['predictions'][v]=pred
            evidence.append(row)
    for i in range(6):
        offset=len(evidence)-12+2*i
        a,b=evidence[offset:offset+2]
        if a['prefix_sha256']!=b['prefix_sha256'] or \
            a['collision_group']!=b['collision_group']:
            raise AssertionError('broken collision construction')
        for k in a['predictions']:
            x,y=a['predictions'][k],b['predictions'][k]
            if x['program']!=y['program'] or x['accepted']!=y['accepted']:
                raise AssertionError('prefix-only predictor leaked the future')
            if int(x['correct_future'])+int(y['correct_future'])>1:
                raise AssertionError('collision member exceeded information bound')
    for d,entries in by_variant_depth.items():
        for variant in ('structural_jacobi','learned_recurrent'):
            by_variant[variant][d]={g:score([e for grp,e in entries[variant] if grp==g])
                                    for g in GROUPS}
    metrics={}
    for name in NEW_VARIANTS:
        if name in ('structural_jacobi','learned_recurrent'):
            metrics[name]=by_variant[name]
        else:
            metrics[name]={g:score(by_variant[name][g]) for g in GROUPS}
    if not all(metrics['structural_jacobi']['7'][g]['intermediate_exact']==len(split[g])
               for g in GROUPS):
        raise AssertionError('Jacobi exact-seven identity failed')
    if any(x['accepted'] and not x['correct_future']
           for x in by_variant['exact_symbolic']['unseen_modulus']):
        raise AssertionError('exact solver failed an in-class future control')
    source_hashes={s:sha_file(HERE/s) for s in SOURCE_FILES}
    # Audit history including RRC-04 and complete source/checkpoint content.
    checkpoint_hashes={s:sha_file(HERE/'results'/s) for s in CHECKPOINTS}
    trace_bytes=json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()
    selected_ids={'unseen_modulus-0000','gap2-min1','gap2-min2',
                  'gap3-min1','pentagonal-cap0','pair0-base','pair0-mutant',
                  'out-000'}
    output={
        'schema':'rrc05.cpu_replay.v1',
        'scope':'Finite exact prefix certificates; gains trained with exact algebraic b_n/n and wired divisor graph; no new infinite theorem or raw-input neural advantage',
        'split_sizes':{k:len(v) for k,v in split.items()},
        'alias_with_train':alias,
        'roster_digest':{k:roster_digest(v) for k,v in split.items()},
        'prefix_degree':PREFIX,'forecast_degree':HORIZON,
        'learned_parameter_count':2,'weight_bits_fp32':64,
        'algebraic_preprocessor':'O(N^2) exact integer log-derivative b; divisor adjacency wired',
        'proper_divisor_edges':sum(len(row) for row in proper_divisors(PREFIX)),
        'checkpoint_sha256':checkpoint_hashes,
        'source_sha256':source_hashes,
        'gains':{k:{name:checkpoints[k][name] for name in ('alpha','beta')}
                 for k in ('rrc05_recurrent.json','rrc05_feedforward.json',
                           'rrc05_shuffled_recurrent.json')},
        'metrics':metrics,
        'collision_pairs':6,
        'trace_sha256':sha256(trace_bytes).hexdigest(),
        'case_sha256':{x['id']:sha256(json.dumps(x,sort_keys=True,
                          separators=(',',':')).encode()).hexdigest() for x in evidence},
        'selected_cases':[x for x in evidence if x['id'] in selected_ids],
    }
    if include_trace:
        output['trace']=evidence
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc05_pilot.json')
    args=p.parse_args()
    result=evaluate()
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.exists() or args.out.read_text()!=payload:
            raise SystemExit('RRC-05 missing or stale source/checkpoint-bound receipt')
        print('RRC-05 source/checkpoint-bound stdlib replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        totals={name:(m['7'] if name in ('learned_recurrent','structural_jacobi') else m)
                for name,m in result['metrics'].items()}
        print(json.dumps({'output':str(args.out),'sizes':result['split_sizes'],
                          'selected_scores':{name:record.get('unseen_modulus')
                          for name,record in totals.items()}},sort_keys=True,indent=2))

if __name__=='__main__': main()
