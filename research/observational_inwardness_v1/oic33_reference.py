"""OIC-33 proof-carrying reference for mandatory-address pruning.

--check: validate pinned hashes, q20 policy, exact state-count manifest.
--cpp-check: compile original/new C++ and recheck Q14 UNSAT, Q20 SAT.
The Q19 capped exploration is reported separately; it is not a proof.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from oic28_verify import verify_tree

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results'/'oic33_reference.json'
TREE=HERE/'results'/'oic33_q20_tree.json.gz.b64'
NEW=HERE/'oic33_tight.cpp'
OLD=HERE/'oic28_affine.cpp'


def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()


def canonical(data):return json.dumps(data,sort_keys=True,separators=(',',':')).encode('utf8')


def check():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if m['schema']!='OIC33-TIGHT-MARKER-REF-V1':raise ValueError('unknown schema')
    for file,key in ((NEW,'new_solver_sha256'),(OLD,'baseline_solver_sha256'),
                     (HERE/'oic33_tight_reference.py','reference_sha256'),
                     (HERE/'test_oic33_tight.py','tests_sha256')):
        if sha(file.read_bytes())!=m['source_hashes'][key]:
            raise AssertionError('source hash mismatch: '+str(file))
    comp=base64.b64decode(TREE.read_bytes().strip(),validate=True)
    decoded=gzip.decompress(comp)
    if sha(comp)!=m['certificate_gzip_sha256'] or sha(decoded)!=m['certificate_canonical_sha256']:
        raise AssertionError('policy certificate digest mismatch')
    obj=json.loads(decoded)
    if canonical(obj)!=decoded:raise AssertionError('noncanonical policy serialization')
    res=verify_tree(obj,n=7,q=20)
    if res!=m['verified_policy']:
        raise AssertionError(f'policy verification mismatch: {res}')
    if m['comparisons']['14']['new']['status']!='UNSAT':raise AssertionError('missing Q14 negative')
    if m['comparisons']['20']['new']['status']!='SAT':raise AssertionError('missing Q20 positive')
    return m,obj


def cpp_check():
    m,tree=check()
    with tempfile.TemporaryDirectory(prefix='oic33_') as d:
        binaries={}
        for label,source in (('baseline',OLD),('new',NEW)):
            out=Path(d)/(label+'_solver')
            subprocess.run(['g++','-std=c++17','-O3',str(source),'-o',str(out)],check=True,timeout=40)
            binaries[label]=out
        for q in (14,20):
            for label in ('baseline','new'):
                config=m['comparisons'][str(q)][label]
                out=Path(d)/f'{label}_q{q}.json'
                proc=subprocess.run([str(binaries[label]),str(q),'11000000','0','1','1',str(out)],
                                    capture_output=True,text=True,timeout=90,check=False)
                if proc.returncode != 0:raise AssertionError(f'{label} Q{q} exited {proc.returncode}: {proc.stderr}')
                mm=re.search(r'STATUS (SAT|UNSAT).*?states=(\d+)',proc.stdout)
                if mm is None:raise AssertionError('unrecognized solver status: '+proc.stdout)
                observed={'status':mm.group(1),'states':int(mm.group(2))}
                if observed!=config:raise AssertionError(f'{label} Q{q} mismatch {observed} != {config}')
                if q==20:
                    produced=json.loads(out.read_text(encoding='utf8'))
                    if verify_tree(produced,n=7,q=20)!=m['verified_policy']:
                        raise AssertionError('regenerated policy invalid')
                    if label=='new' and canonical(produced)!=canonical(tree):
                        raise AssertionError('new solver policy differs from pinned certificate')
                print('OIC-33 executable reference',label,'Q',q,observed,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--cpp-check',action='store_true')
    args=parser.parse_args()
    if args.cpp_check:cpp_check()
    else:
        m,_=check()
        print('OIC-33 saved Q20 tree verified:',m['verified_policy'],flush=True)
        print('Q19 status:',m['q19_search_status'],flush=True)

if __name__=='__main__':main()
