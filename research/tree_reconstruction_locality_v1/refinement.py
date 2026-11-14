"""KSR-10: exact finite one-step refinement of the native audit labels.

A refinement preserves the original parent label and separates all distinct
child-law vectors inside that label. It restores independence of the ORIGINAL
child labels conditional on the refined parent, for this finite kernel.
Recursive closure of the refined labels is a separate requirement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
try:
    from .dependence import probability_row
except ImportError:
    from dependence import probability_row


def certify(records,q):
    if type(q) is not int or q<2 or not records:
        raise ValueError('nonempty parent support and q>=2 required')
    groups=defaultdict(list)
    ids=set()
    for r in records:
        label=r['label']; row=probability_row(r['one_step_child_law'])
        if type(label) is not int or label not in range(q) or len(row)!=q:
            raise ValueError('invalid parent/child alphabet')
        if r['case_id'] in ids:
            raise ValueError('duplicate parent ID')
        ids.add(r['case_id'])
        groups[label,row].append(r['case_id'])
    ordered=sorted(groups)
    certificates=[]
    for code,(label,row) in enumerate(ordered):
        certificates.append({'code':code,'original_label':label,
            'child_law':[str(x) for x in row],
            'case_ids':groups[label,row],
            'conditional_sibling_diagonal_excess':'0'})
    per_label=[sum(1 for label,_ in groups if label==c) for c in range(q)]
    size=sum(per_label)
    if size!=len(groups) or sum(len(x['case_ids']) for x in certificates)!=len(records):
        raise AssertionError('partition did not cover the original support')
    return {'root_states':len(records),'original_alphabet_size':q,
            'distinct_kernels_per_original_label':per_label,
            'minimal_supported_refinement_states':size,
            'minimal_joint_fixed_length_bits':(size-1).bit_length(),
            'original_label_fixed_length_bits':(q-1).bit_length(),
            'fixed_extra_bits_given_original_label':max((k-1).bit_length() if k else 0 for k in per_label),
            'refined_parent_original_child_independence':True,
            'refined_child_recursive_markov_closure_claimed':False,
            'static_partition_codebook_is_separate':True,
            'groups':certificates}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    raw=args.audit.read_bytes(); audit=json.loads(raw)
    if audit.get('schema')!='ksr01c.rrc25_channel.v1':
        raise ValueError('requires original KSR-01C audit receipt')
    report={'status':'EXACT_FINITE_ONE_STEP_REFINEMENT',
            'audit_sha256':hashlib.sha256(raw).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_sha256':audit['checkpoint_sha256'],
            'cohorts':{}}
    for name in ('calibration','heldout'):
        report['cohorts'][name]=certify(audit[name]['root_witness_records'],audit['q'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cohorts':{
        name:{k:v for k,v in result.items() if k!='groups'}
        for name,result in report['cohorts'].items()}},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
