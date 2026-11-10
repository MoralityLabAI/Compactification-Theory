"""ARG-01 offline real-model proposal exchange for Qwen / looped checkpoints.

This module never downloads a model, calls a network service, or claims that a
listed depth corresponds to internal recurrent steps without an activation
capture and independently pinned inference implementation.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import re
from pathlib import Path
from spiral_ln import algebraic_reasoning_oracle as oracle

DEPTHS=(0,1,2,4,8)
TEST_SPLITS=oracle.ALL_FAMILIES+("negative","collision")
ALLOWED_FIELDS={"case_id","public_sha256","depth","program","model_revision",
                "inference_ms","generated_tokens","model_forward_calls"}


def public_records(*,splits=TEST_SPLITS):
    roster=oracle.roster()
    for split in splits:
        if split not in TEST_SPLITS:raise ValueError('unrecognized public split')
        for case in roster[split]:
            payload=case.public
            payload['public_sha256']=case.signature
            yield payload


def write_public(path:Path,*,splits=TEST_SPLITS):
    rows=list(public_records(splits=splits))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in rows))
    return len(rows)


def read_proposals(path:Path):
    if not path.is_file():raise FileNotFoundError(path)
    result={}
    revisions=set()
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:p=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f'line {line_no}: bad JSON') from exc
        if not isinstance(p,dict) or set(p)-ALLOWED_FIELDS:
            raise ValueError(f'line {line_no}: unexpected field (future/labels forbidden)')
        required={"case_id","public_sha256","depth","program","model_revision"}
        if not required.issubset(p):raise ValueError(f'line {line_no}: incomplete proposal')
        if type(p['depth']) is not int or p['depth'] not in DEPTHS:raise ValueError('invalid loop depth')
        if not isinstance(p['model_revision'],str) or not re.fullmatch(r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})',p['model_revision']):
            raise ValueError('unpinned model revision')
        if not isinstance(p['public_sha256'],str) or len(p['public_sha256'])!=64:
            raise ValueError('invalid public input digest')
        if p['program'] is not None and (not isinstance(p['program'],list) or
                tuple(p['program']) not in oracle.PROGRAMS):raise ValueError('invalid program syntax')
        if not isinstance(p['case_id'],str):raise ValueError('invalid case id')
        for k in ('inference_ms','generated_tokens','model_forward_calls'):
            if k in p and (not isinstance(p[k],(int,float)) or isinstance(p[k],bool) or p[k]<0):
                raise ValueError('invalid resource value')
        key=(p['case_id'],p['depth'])
        if key in result:raise ValueError('duplicate case/depth proposal')
        result[key]=p
        revisions.add(p['model_revision'])
    if len(revisions)!=1:raise ValueError('exactly one resolved model revision per receipt')
    return result,next(iter(revisions))


def score(path:Path):
    proposals,model=read_proposals(path)
    data=oracle.roster()
    cases={c.identifier:c for split in TEST_SPLITS for c in data[split]}
    if any(k[0] not in cases for k in proposals):raise ValueError('unknown case id')
    per_depth={}
    for depth in DEPTHS:
        supplied=0;valid=0;correct=0;false_future=0;queries=0
        for case_id,case in cases.items():
            p=proposals.get((case_id,depth))
            if p is None:continue
            if p['public_sha256']!=case.signature:raise ValueError('public input digest mismatch')
            supplied+=1
            if p['program'] is None:continue
            verified=oracle.search(case,[p['program']],budget=1)
            result=oracle.grade(case,verified)
            valid+=result['accepted']
            correct+=result['future_correct']
            false_future+=result['false_future_acceptance']
            queries+=verified['queries']
        per_depth[str(depth)]={'supplied':supplied,'expected':len(cases),'exact_accepted':valid,
                              'future_correct':correct,'false_future':false_future,
                              'paid_graph_checks':queries,'missing':len(cases)-supplied}
    return {'schema':'algernon.arg01.external.v1','model_revision':model,
            'proposal_sha256':sha256(path.read_bytes()).hexdigest(),
            'candidate_depths_are_model_declared':True,
            'claim_boundary':'Model depth labels require separate weight/activation evidence to establish internal loop counts.',
            'scores':per_depth}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--export-public',type=Path)
    options.add_argument('--score-proposals',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.export_public:
        print('wrote',write_public(args.export_public),'public cases to',args.export_public)
    else:
        data=json.dumps(score(args.score_proposals),sort_keys=True,indent=2)+'\n'
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(data)
            print('saved',args.out)
        else:print(data,end='')

if __name__=='__main__':main()
