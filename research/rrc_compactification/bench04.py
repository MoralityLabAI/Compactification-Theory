"""RRC-04: source-bound finite replay and honest program-forecast scoring.

No training, torch, GPU or network is used by this script. It reads frozen
checkpoint arrays and deterministic raw-count rosters, proposes programs, and
independently verifies prefixes/future exact integer coefficients. The test
families are constructed only here, not by train04.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

try:
    from .program04 import (HERE,PREFIX,HORIZON,MAX_PERIOD,CAPS,Case,Program,
                            roster,roster_digest,coefficients,synthesize,
                            synthesize_all,complete_skeleton,verify_prefix)
    from .student04 import ProgramStudent
except ImportError:
    from program04 import (HERE,PREFIX,HORIZON,MAX_PERIOD,CAPS,Case,Program,
                           roster,roster_digest,coefficients,synthesize,
                           synthesize_all,complete_skeleton,verify_prefix)
    from student04 import ProgramStudent

SOURCE_FILES=('rrc.py','program04.py','student04.py','train04.py','bench04.py')
DIRECTIONS={
    'recurrent':(0,1,2,4,8),
    'feedforward':(2,),
    'shuffled_recurrent':(2,),
}


def sha_file(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def score_case(case:Case,program:Program|None)->tuple[bool,bool,int|None]:
    if program is None: return False,False,None
    candidate=coefficients(HORIZON,program)
    certified=candidate[:PREFIX+1]==list(case.prefix)
    mismatch=next((n for n in range(PREFIX+1,HORIZON+1)
                  if candidate[n]!=case.full[n]),None)
    return certified,certified and mismatch is None,mismatch


def metric(items:list[dict])->dict:
    n=len(items)
    accepted=sum(int(item['prefix_verified']) for item in items)
    future=sum(int(item['correct_horizon']) for item in items)
    false_accept=accepted-future
    return {'cases':n,'prefix_verified':accepted,'correct_horizon':future,
            'false_accept':false_accept,'abstained':n-accepted,
            'mean_syntax_bits_accepted':round(sum(
                item['syntax_bits'] for item in items if item['prefix_verified'])/accepted,4)
                if accepted else None}


def run(*,train_size:int=1500,dev_size:int=180,unseen_size:int=108)->dict:
    splits=roster(train_size=train_size,dev_size=dev_size,unseen_size=unseen_size)
    cases=[case for split in ('dev','unseen_modulus','local_gap','nonperiodic','collision')
           for case in splits[split]]
    checkpoints={name:HERE/'results'/f'rrc04_{name}.json' for name in DIRECTIONS}
    models={name:ProgramStudent.load(path) for name,path in checkpoints.items()}
    by_case=[]
    by_model={name:{depth:[] for depth in depths} for name,depths in DIRECTIONS.items()}
    oracle_decisions=[]
    alias_counts={}
    train_signatures={case.signature() for case in splits['train']}
    for split in ('dev','unseen_modulus','local_gap','nonperiodic','collision'):
        alias_counts[split]=sum(case.signature() in train_signatures for case in splits[split])
    for case in cases:
        oracle=synthesize(case.prefix)
        c_ok,f_ok,mismatch=score_case(case,oracle)
        oracle_decisions.append({'id':case.name,'family':case.family,
                                 'split':case.split if case.split!='test' else
                                 ('collision' if case.collision_group else
                                  ('local_gap' if case.family=='local_gap' else 'nonperiodic')),
                                 'group':next((label for label in splits if case in splits[label]),None),
                                 'rule':None if oracle is None else oracle.as_dict(),
                                 'prefix_verified':c_ok,'correct_horizon':f_ok,
                                 'first_future_mismatch':mismatch,
                                 'syntax_bits':oracle.bits if oracle else None})
        row={'case_id':case.name,'source_family':case.family,
             'prefix_sha256':case.signature(), 'collision_group':case.collision_group,
             'oracle_program':None if oracle is None else oracle.as_dict(),
             'model_proposals':{}}
        for name,model in models.items():
            encoded=model.encode(case.prefix)  # encoding shared across depth ablation
            predictions={}
            for depth in DIRECTIONS[name]:
                output=model.from_embedding(encoded,depth)
                proposal=model.decode(output)
                accepted,correct,first_mismatch=score_case(case,proposal)
                # Separately charge an exact solver that fills *only* the mask;
                # this hybrid is explicitly not a neural full-program proposal.
                repaired=complete_skeleton(case.prefix,proposal.cap,proposal.period)
                repair_accepted,repair_correct,repair_mismatch=score_case(case,repaired)
                pred={'proposal':proposal.as_dict(),
                      'prefix_verified':accepted,'correct_horizon':correct,
                      'syntax_bits':proposal.bits,
                      'first_future_mismatch':first_mismatch,
                      'hybrid_program':None if repaired is None else repaired.as_dict(),
                      'hybrid_prefix_verified':repair_accepted,
                      'hybrid_correct_horizon':repair_correct,
                      'hybrid_first_future_mismatch':repair_mismatch}
                predictions[str(depth)]=pred
                by_model[name][depth].append({'prefix_verified':accepted,
                    'correct_horizon':correct,'syntax_bits':proposal.bits,
                    'hybrid_prefix_verified':repair_accepted,
                    'hybrid_correct_horizon':repair_correct})
            row['model_proposals'][name]=predictions
        by_case.append(row)

    grouped=defaultdict(list)
    for item in oracle_decisions: grouped[item['group']].append(item)
    oracle_metrics={group:metric(rows) for group,rows in grouped.items()}
    models_metrics={}
    for name,depths in DIRECTIONS.items():
        model=models[name]
        models_metrics[name]={}
        for depth,items in by_model[name].items():
            result=metric(items)
            result['hybrid_verified']=sum(int(i['hybrid_prefix_verified']) for i in items)
            result['hybrid_correct_horizon']=sum(int(i['hybrid_correct_horizon']) for i in items)
            result['approx_inference_macs_per_case']=model.estimated_macs(depth)
            result['split_breakdown']={}
            start=0
            for split in ('dev','unseen_modulus','local_gap','nonperiodic','collision'):
                length=len(splits[split]); chunk=items[start:start+length]
                record=metric(chunk)
                record['hybrid_verified']=sum(int(i['hybrid_prefix_verified']) for i in chunk)
                record['hybrid_correct_horizon']=sum(int(i['hybrid_correct_horizon']) for i in chunk)
                result['split_breakdown'][split]=record
                start+=length
            assert start==len(items)
            models_metrics[name][str(depth)]=result
    # A prefix-only agent must produce the same program for both partners.
    for pair in range(8):
        base,mutant=splits['collision'][2*pair:2*pair+2]
        if base.prefix!=mutant.prefix: raise AssertionError('collision prefix changed')
        offset=len(by_case)-16+2*pair
        rb,rm=by_case[offset:offset+2]
        for model_name in DIRECTIONS:
            for depth in DIRECTIONS[model_name]:
                p0=rb['model_proposals'][model_name][str(depth)]
                p1=rm['model_proposals'][model_name][str(depth)]
                for field in ('proposal','prefix_verified','hybrid_program','hybrid_prefix_verified'):
                    if p0[field]!=p1[field]:
                        raise AssertionError('prefix-only predictor saw hidden future')
        if rb['oracle_program']!=rm['oracle_program']:
            raise AssertionError('exact prefix oracle leaked future')
    fingerprints={key:sha_file(HERE/key) for key in SOURCE_FILES}
    model_hashes={name:sha_file(path) for name,path in checkpoints.items()}
    frozen={
        'schema':'rrc04.pilot.v1',
        'scope':'CPU-trained raw-prefix toy policy and finite-degree exact verification; no infinite theorem or recurrent advantage',
        'prefix_degree':PREFIX,'forecast_degree':HORIZON,
        'train_moduli':[2,3,4,6,8,9,10,12],
        'unseen_moduli':[5,7,11],
        'cap_semantics':{'0':'unlimited','1':'distinct','2':'at_most_two'},
        'sizes':{key:len(rows) for key,rows in splits.items()},
        'prefix_aliases_with_train':alias_counts,
        'source_sha256':fingerprints,'checkpoints_sha256':model_hashes,
        'roster_sha256':{key:roster_digest(rows) for key,rows in splits.items()},
        'recurrent_parameter_count':models['recurrent'].parameters,
        'feedforward_parameter_count':models['feedforward'].parameters,
        'oracle_by_split':oracle_metrics,
        'models_by_depth':models_metrics,
        'collision_bound':{'pairs':8,'universal_correct_limit_given_shared_prefix':8,
                           'pairs_with_identical_prefix':8},
        'full_case_trace_sha256':sha256(json.dumps(by_case,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'case_sha256_by_id':{row['case_id']:sha256(json.dumps(
            row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            for row in by_case},
        'selected_case_details':[row for row in by_case if row['case_id'] in (
            'dev-0000','unseen_modulus-0000','gap2-min1','gap2-min2',
            'gap3-min1','squares-cap0','pair0-base','pair0-mutant')],
    }
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc04_pilot.json')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    data=run()
    payload=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if a.check:
        if not a.out.is_file() or a.out.read_text()!=payload:
            raise SystemExit('RRC-04 missing/stale source-bound receipt; regenerate without --check')
        print('RRC-04 source-bound CPU replay verified')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(payload)
        print(json.dumps({'path':str(a.out),'summary':data['models_by_depth']['recurrent']['2']},indent=2))

if __name__=='__main__':main()
