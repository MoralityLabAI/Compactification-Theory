"""Post-freeze future audit; withheld results never return to the curriculum."""
import argparse
import json
from pathlib import Path
from research.dimensional_atlas.gym.tasks import load_evalmath, make_evalmath_case
from research.dimensional_atlas.gym.scm18_rtx3050.core import sha, write_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    args=p.parse_args()
    result=args.root/'results'
    frozen=json.loads((result/'freeze.json').read_text())
    registration=json.loads((result/'registration.json').read_text())
    assert sha(result/'registration.json')==frozen['registration_sha256']
    oracle=load_evalmath(args.oracle)
    rows=[]
    for row in registration['mining']['rows']:
        seed=int(row['id'].split(':')[-1])
        task=make_evalmath_case(oracle,seed,case_id=row['id'],family=row['family'])
        assert task.signature==row['signature']
        graph=tuple(row['maximum']['graph'])
        predicted=oracle.execute(task.oracle_case.query,task.topology,graph)
        target=task.hidden_target
        first=next((i for i,(a,b) in enumerate(zip(predicted,target)) if a!=b),None)
        prefix=row['maximum']['prefix_accepted']
        rows.append({'case_id':row['id'],'prefix_accepted':prefix,
                     'observed_first_faults':row['maximum']['first_faults'],
                     'first_query_mismatch_coefficient':first,
                     'future_correct':first is None,
                     'passes_prefix_fails_future':prefix and first is not None})
    write_new(result/'discovery_future_audit.json',{'freeze_sha256':sha(result/'freeze.json'),
        'oracle_sha256':sha(args.oracle),'status':'post_freeze_not_available_to_training',
        'source_make_case_calls':len(rows),'source_query_execute_calls':len(rows),
        'note':'source generation performs its own exact demo/query executions', 'records':rows})


if __name__=='__main__':main()
