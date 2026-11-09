"""RRC-09 finite-program synthesis audit, with source/checkpoint-hashed replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import synth09 as ds, model09
except ImportError:
    import synth09 as ds
    import model09

HERE=Path(__file__).resolve().parent
FILES=('synth09.py','model09.py','train09.py','bench09.py','test_synth09.py',
       'rrc.py','rrc05.py')
CHECKPOINTS=('recurrent','feedforward','shuffled_recurrent','guided_ranker')
METHODS=('exact','recurrent_1','recurrent_2','recurrent_4','feedforward',
         'shuffled','guided_1','guided_4','guided_8','guided_31',
         'sketch_1','sketch_4','sketch_8','sketch_31')


def sources():
    return {p:sha256((HERE/p).read_bytes()).hexdigest() for p in FILES}


def load_checkpoints():
    hashes=sources();observed={}
    for key in CHECKPOINTS:
        path=HERE/'results'/f'rrc09_{key}.json'
        blob=path.read_bytes()
        ck=json.loads(blob)
        if ck['schema'] != ('rrc09.ranker.v1' if key=='guided_ranker' else 'rrc09.direct.v1'):
            raise AssertionError('wrong schema')
        if ck['train09_sha256']!=hashes['train09.py'] or ck['synth09_sha256']!=hashes['synth09.py']:
            raise AssertionError(f'{key} source drift')
        observed_training_ops=(ck['trained_candidate_ops'] if key=='guided_ranker'
                               else ck['training_operators'])
        if observed_training_ops != list(ds.TRAIN_OPS):
            raise AssertionError('missing training split lock')
        if key in ('recurrent','shuffled_recurrent') and ck['parameter_count']!=1692:
            raise AssertionError('recurrent budget changed')
        if key=='feedforward' and ck['parameter_count']!=1695:
            raise AssertionError('feedforward budget changed')
        observed[key]=(ck,sha256(blob).hexdigest())
    train=ds.generate(include_test=False)['train']
    for kind,(c,_) in observed.items():
        if c['training_roster_sha256']!=ds.roster_digest(train,include_full=False):
            raise AssertionError(f'{kind} roster drift')
    return observed


def verify_program(case:ds.Case09, program, *, include_future:bool=True):
    """Independent exact execution of a proposal. Only observed prefix gates it."""
    if program is None:
        return {'prefix_verified':0,'correct_future':0,'first_fault':None,
                'future_first_fault':None}
    prog=tuple(program)
    if len(prog)>2 or any(p not in ds.OPS for p in prog):
        raise ValueError('invalid grammar proposal')
    evaluated=ds.execute(case.source[:ds.PREFIX+1],prog)
    fault=ds.first_fault(evaluated,case.target[:ds.PREFIX+1])
    if fault is not None:
        return {'prefix_verified':0,'correct_future':0,'first_fault':fault,
                'future_first_fault':None}
    future_fault=None
    if include_future:
        full=ds.execute(case.source,prog)
        future_fault=ds.first_fault(full,case.target,start=ds.PREFIX+1)
    return {'prefix_verified':1,'correct_future':int(future_fault is None) if include_future else 0,
            'first_fault':None,'future_first_fault':future_fault}


def _collect(rows):
    if not rows:return {}
    return {'cases':len(rows),
            'prefix_verified':sum(r['prefix_verified'] for r in rows),
            'future_correct':sum(r['correct_future'] for r in rows),
            'false_accept':sum(r['prefix_verified']-r['correct_future'] for r in rows),
            'abstentions':sum(not r['prefix_verified'] for r in rows),
            'mean_candidate_attempts':round(sum(r['candidates'] for r in rows)/len(rows),3),
            'mean_public_primitive_calls':round(sum(r['primitive_calls'] for r in rows)/len(rows),3),
            'mean_estimated_coefficient_work':round(sum(r['estimated_arithmetic_work'] for r in rows)/len(rows),3),
            'mean_full_verifications':round(sum(r['full_verifications'] for r in rows)/len(rows),3)}


def run():
    loaded=load_checkpoints();data=ds.generate()
    training_sigs={case.input_signature for case in data['train']}
    split_alias={k:sum(c.input_signature in training_sigs for c in v)
                 for k,v in data.items() if k!='train'}
    if any(split_alias.values()):raise AssertionError('new evaluation data aliases training')
    assert all('C' not in c.hidden_program and 'D' not in c.hidden_program for c in data['train']+data['dev'])
    metrics=defaultdict(lambda:defaultdict(list))
    collisions=[]
    for split,cases in data.items():
        if split=='train':continue
        for case in cases:
            proposals={}
            exact=ds.catalog_search(case.source,case.target)
            proposals['exact']=(exact['program'],exact['candidates_examined'],exact['primitive_calls'],exact['full_verifications'],exact['estimated_arithmetic_work'])
            for key,ckname,sweeps in (
                ('recurrent_1','recurrent',1),('recurrent_2','recurrent',2),
                ('recurrent_4','recurrent',4),('feedforward','feedforward',1),
                ('shuffled','shuffled_recurrent',1)):
                program=model09.direct_program(case,loaded[ckname][0],sweeps=sweeps)
                proposals[key]=(program,1,len(program),1,ds.arithmetic_work(program,ds.PREFIX))
            for k in (1,4,8,31):
                prediction=ds.guided_search(case.source,case.target,loaded['guided_ranker'][0]['weights'],budget=k)
                proposals[f'guided_{k}']=(prediction['program'],prediction['preview_candidates'],
                                           prediction['primitive_calls'],prediction['full_verifications'],
                                           prediction['estimated_arithmetic_work'])
                sketch=ds.guided_search(case.source,case.target,None,budget=k)
                proposals[f'sketch_{k}']=(sketch['program'],sketch['preview_candidates'],
                                           sketch['primitive_calls'],sketch['full_verifications'],
                                           sketch['estimated_arithmetic_work'])
            for name,(prog,attempted,calls,checks,work) in proposals.items():
                result=verify_program(case,prog)
                result.update(candidates=attempted,primitive_calls=calls,
                              estimated_arithmetic_work=work,full_verifications=checks)
                metrics[name][split].append(result)
    for a,b in zip(data['late_collision'][::2],data['late_collision'][1::2]):
        assert a.input_signature==b.input_signature and a.target!=b.target
        collisions.append({'pair':a.pair,'degree':ds.first_fault(a.target,b.target,ds.PREFIX+1),
                           'same_observed':True})
    checks={}
    for kind,splits in metrics.items():
        totals=[r for rows in splits.values() for r in rows]
        checks[kind]={'total':_collect(totals),'by_split':{k:_collect(rows) for k,rows in splits.items()},
                      'heldout':_collect([r for k,rows in splits.items() if k!='dev' for r in rows])}
    return {'schema':'rrc09.receipt.v1','status':'exact_CPU_bounded_program_synthesis',
            'source_sha256':sources(),'checkpoint_sha256':{k:s for k,(_,s) in loaded.items()},
            'train_roster_sha256':ds.roster_digest(data['train']),
            'roster_sha256':{k:ds.roster_digest(v) for k,v in data.items()},
            'split_alias_with_training':split_alias,
            'parameters':{'train_count':ds.TRAIN_COUNT,'dev_count':ds.DEV_COUNT,
                          'test_counts':ds.TEST_COUNTS,'prefix_degree':ds.PREFIX,
                          'preview_degree':ds.PREVIEW,'future_degree':ds.HORIZON,
                          'public_full_grammar_size':len(ds.all_programs()),
                          'training_grammar_size':len(ds.all_programs(ds.TRAIN_OPS)),
                          'excluded_training_primitives':['C','D'],
                          'direct_student_fp32_params':1692,
                          'feedforward_fp32_params':1695,
                          'direct_student_live_hidden_fp32':20,
                          'guided_ranker_float64_params':6,
                          'program_syntax_bits_excluding_interpreter':6,
                          'all_candidate_previews_charged':True},
            'metrics':checks,
            'collision_pairs':collisions,
            'limitations':'Guided learner sees outputs of the supplied public C/D interpreter at test; exact search is the class-complete control. Direct neural models have no supervised C/D labels. Verified prefixes do not certify infinite identities or late mutations.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc09_pilot.json')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    obj=run();raw=json.dumps(obj,sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.exists() or args.out.read_text()!=raw:
            raise SystemExit('RRC-09 receipt missing/stale; validate source and rerun without --check')
        print('RRC-09 exact source-bound replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(raw)
        print(json.dumps({'path':str(args.out),'heldout':{k:v['heldout'] for k,v in obj['metrics'].items()}},indent=2))

if __name__=='__main__':main()
