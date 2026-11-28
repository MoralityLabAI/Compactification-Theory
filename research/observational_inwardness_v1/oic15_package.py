"""Produce deterministic OIC-15 receipts with independent model weights archival."""
from __future__ import annotations
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULT=HERE/'results'


def sha256(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def write_b64(payload:dict,path:Path)->dict:
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')
    comp=gzip.compress(raw,compresslevel=9,mtime=0)
    path.write_text(base64.b64encode(comp).decode('ascii')+'\n')
    return {'path':path.name,'raw_sha256':sha256(raw),'gzip_sha256':sha256(comp),
            'raw_bytes':len(raw),'gzip_bytes':len(comp),'base64_bytes':path.stat().st_size}


def main():
    full=json.loads((RESULT/'oic15_full.json').read_text())
    analysis=json.loads((RESULT/'oic15_analysis.json').read_text())
    thin=copy.deepcopy(full)
    checkpoints=[]
    for original,record in zip(full['runs'],thin['runs']):
        w=record.pop('weights')
        checkpoints.append({'architecture':original['architecture'],'split':original['split'],
                            'seed':original['seed'],'weights':w})
    thin.pop('wall_seconds',None)
    files={
        'traces':write_b64(thin,RESULT/'oic15_traces.json.gz.b64'),
        'weights':write_b64({'protocol':full['protocol'],'runs':checkpoints},RESULT/'oic15_weights.json.gz.b64')
    }
    source_files=['oic15_pilot.py','oic15_analyze.py','oic15_math.py','test_oic15_math.py','oic15_package.py','oic15_verify.py','test_oic15_verify.py']
    codes={name:sha256((HERE/name).read_bytes()) for name in source_files}
    full_stats={'n_fits':len(full['runs']),'seeds':full['protocol']['seeds'],
                'training_splits':full['protocol']['splits'],
                'architectures':full['protocol']['architectures'],
                'train_steps':full['protocol']['steps'],
                'training_examples_per_fit':full['protocol']['steps']*full['protocol']['batch'],
                'cpu_wall_seconds_recorded':full['wall_seconds']}
    digest={
        'tag':'OIC-15-CPU-01','status':'executed_finite_synthetic_monitor_experiment',
        'scope':'one declared hidden-cycle toy system, not a pretrained model',
        'files':files,'source_sha256':codes,
        'analysis_sha256':sha256((RESULT/'oic15_analysis.json').read_bytes()),
        'full_json_sha256':sha256((RESULT/'oic15_full.json').read_bytes()),
        'experiment':full_stats,
        'ci':'pure-stdlib trace/oracle verifier; optional torch weight-replay verifier'
    }
    (RESULT/'oic15_manifest.json').write_text(json.dumps(digest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(digest,indent=2,sort_keys=True))

if __name__=='__main__':main()
