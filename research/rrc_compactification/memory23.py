"""RRC-23 independent-process peak-RSS profiles for degree-192/384 policies.

One worker per arm/horizon. Model loading, exact class construction, roster
materialization, and warm execution all count toward the process memory peak.
Not part of the exact CPU semantic receipt; values are host-dependent.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
try:
    from . import bench23 as b, scale23 as s, learn18 as l
except ImportError:
    import bench23 as b
    import scale23 as s
    import learn18 as l

HERE=Path(__file__).resolve().parent
ARMS=('symbolic_progressive','selective_progressive')
DEGREES=(192,384)


def rss_kib():
    if not Path('/proc/self/status').exists():return None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])
    return None


def worker(arm,degree):
    if arm not in ARMS or degree not in DEGREES:raise ValueError('invalid measurement arm')
    started=time.perf_counter()
    models=cal=None
    classes=l.class_map()
    if arm.startswith('selective'):
        models,cal,_=b.checkpoints()
    cases=s.dataset() if degree==192 else s.stress_dataset()
    after_setup_rss=rss_kib()
    for original in cases:
        case=s.observed(original,degree)
        b.predict(case,arm,models,cal,classes,degree)
    return {'arm':arm,'degree':degree,'cases':len(cases),
            'setup_resident_kib':after_setup_rss,
            'after_execution_resident_kib':rss_kib(),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'wall_elapsed_ms':round((time.perf_counter()-started)*1000,3)}


def measure():
    records={}
    for degree in DEGREES:
        for arm in ARMS:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                                 '--worker',arm,str(degree)],check=True,text=True,
                                capture_output=True,timeout=90,
                                env={**os.environ,'PYTHONHASHSEED':'0'})
            records[f'{degree}/{arm}']=json.loads(proc.stdout)
    return {'schema':'rrc23.memory.v1','source_sha256':b.source_hashes(),
            'memory_script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'machine':{'system':platform.system(),'python':platform.python_version()},
            'scope':'Fresh process per arm/horizon; Linux ru_maxrss KiB including setup and model loading. Host specific, not exact CI evidence.',
            'records':records}


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=2)
    p.add_argument('--out',type=Path,default=HERE/'results/rrc23_memory_profile.json')
    a=p.parse_args()
    if a.worker:
        arm,degree=a.worker
        print(json.dumps(worker(arm,int(degree))))
    else:
        result=measure();a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps(result['records'],indent=2))

if __name__=='__main__':main()
