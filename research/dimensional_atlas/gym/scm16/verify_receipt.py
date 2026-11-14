"""Read-only SCM-16 public receipt auditor; checks source lock and gold separation.

No private gold is read or regenerated. This validates provenance and record
consistency, not the model's mathematical skill or floating point accuracy.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from .campaign import VERSION
from ..tasks import EXPECTED_EVALMATH_SHA256


def verify(receipt_path:Path,public_path:Path)->dict:
    receipt=json.loads(receipt_path.read_text())
    public=json.loads(public_path.read_text())
    if receipt.get('schema')!=VERSION:raise ValueError('receipt schema mismatch')
    if sha256(public_path.read_bytes()).hexdigest()!=receipt['new_public_sha256']:
        raise ValueError('new public byte hash mismatch')
    if len(public)!=receipt['fresh_challenge_count']:
        raise ValueError('new challenge count mismatch')
    if len(set(r['public_sha256'] for r in public))!=len(public):
        raise ValueError('duplicate generated challenge public signature')
    if any(('future' in r or 'source_program' in r or 'source_graph' in r) for r in public):
        raise ValueError('private gold leaked into generated public tasks')
    for row in public:
        basis={k:row[k] for k in ('topology','demos','query','backend')}
        signature=sha256(json.dumps(basis,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if signature!=row['public_sha256']:raise ValueError('invalid generated task signature')
    root=Path(__file__).parent
    for name,digest in receipt['source_sha256'].items():
        if name not in ('campaign.py','fiber.py','landscape.py'):
            raise ValueError('unexpected source lock')
        if sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('source lock mismatch: '+name)
    if receipt['backend']=='evalmath' and receipt['oracle_sha256']!=EXPECTED_EVALMATH_SHA256:
        raise ValueError('untrusted EvalMath oracle source')
    if receipt['backend'] not in ('evalmath','toy'):
        raise ValueError('unsupported backend')
    if receipt['mining']['verifier_queries']!=receipt['mining']['landscape_tasks']*64:
        raise ValueError('uncharged discovery oracle queries')
    if any(p['peak_type'] not in ('first_fault_local_peak','accepted_prefix_alias')
           for p in receipt['mining']['peaks']):
        raise ValueError('unrecognized local maximum')
    if len(receipt['results'])!=receipt['frozen_protocol']['seed_count']:
        raise ValueError('seed count mismatch')
    for row in receipt['results']:
        if row['readout_kernel_residual']>1e-5:
            raise ValueError('readout fiber is not a null of the frozen head')
        if not row['fiber_mediation_diagnostic']['last_visit_fiber_is_readout_null']:
            raise ValueError('failed terminal head null control')
        for arm in ('tied','joint','visible_only','fiber_only','shuffled_visit','zero_codes'):
            if row['official'][arm]['overall']['n']!=receipt['readout_count']:
                raise ValueError('incomplete matched official evaluation')
        if row['official']['zero_codes']['overall']!=row['official']['tied']['overall']:
            raise ValueError('SAE-zero control differs from tied baseline')
    return {'status':'receipt_metadata_and_source_verified',
            'backend':receipt['backend'],'seed_count':len(receipt['results']),
            'readout_count':receipt['readout_count'],
            'fresh_challenge_count':len(public)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--public',required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(verify(args.receipt,args.public),indent=2))

if __name__=='__main__':main()
