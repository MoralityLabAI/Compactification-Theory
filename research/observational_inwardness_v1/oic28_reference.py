"""OIC-28 reproducible n7 Q14 lower and Q20 upper bound.

--check: pure Python saved-tree and checksum validation.
--cpp-check: independently rebuild and execute the pinned source;
  UNSAT must be exhaustive and a resource cap is always a failure.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from oic28_verify import verify_tree

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic28_reference.json'
ARCHIVE=HERE/'results/oic28_n7_q20_tree.json.gz.b64'
CPP=HERE/'oic28_affine.cpp'
VERIFY=HERE/'oic28_verify.py'

def sha(data):return hashlib.sha256(data).hexdigest()

def require(test,msg):
    if not test:raise AssertionError(msg)

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':')).encode('utf-8')

def check():
    manifest=json.loads(MANIFEST.read_text('utf-8'))
    compressed=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
    source=gzip.decompress(compressed)
    tree=json.loads(source)
    require(sha(compressed)==manifest['certificate_gzip_sha256'],'gzip digest mismatch')
    require(sha(canonical(tree))==manifest['certificate_canonical_sha256'],'tree digest mismatch')
    require(sha(CPP.read_bytes())==manifest['solver_source_sha256'],'C++ source digest mismatch')
    require(sha(VERIFY.read_bytes())==manifest['verifier_source_sha256'],'Python verifier digest mismatch')
    require(manifest['n']==7 and manifest['max_flips']==1,'contract mismatch')
    require(manifest['negative_horizon']==14 and manifest['positive_horizon']==20,'horizon change')
    stats=verify_tree(tree,n=7,q=20)
    require(stats==manifest['certificate_stats'],'certificate stats mismatch')
    print('OIC-28 receipt verified:',json.dumps(stats,sort_keys=True))
    return manifest,tree

def recompile_check():
    manifest,tree=check()
    with tempfile.TemporaryDirectory(prefix='oic28_') as directory:
        binary=Path(directory)/'solver'
        subprocess.run(['g++','-std=c++17','-O3',str(CPP),'-o',str(binary)],check=True,timeout=120)
        for config in manifest['searches']:
            q=config['Q']; cap=config['state_cap']; heur=config['heuristic']
            output=Path(directory)/f'policy_q{q}.json'
            result=subprocess.run([str(binary),str(q),str(cap),str(heur),'1','1',str(output)],
                                  capture_output=True,text=True,timeout=100,check=False)
            require(result.returncode==0,f'Q{q} solver returned nonzero (cap or failure): {result.stderr}')
            parsed=re.search(r'STATUS (SAT|UNSAT).*?states=(\d+).*?pair_pruned=(\d+)',result.stdout)
            require(parsed is not None,'missing strict solver status')
            actual={'status':parsed.group(1),'expanded_states':int(parsed.group(2)),
                    'pair_pruned':int(parsed.group(3))}
            expected={key:config[key] for key in actual}
            require(actual==expected,f'Q{q} mismatch: got {actual}, expected {expected}')
            if actual['status']=='SAT':
                require(output.is_file(),'missing certificate file')
                regenerated=json.loads(output.read_text('utf-8'))
                require(canonical(regenerated)==canonical(tree),'regenerated tree differs')
                require(verify_tree(regenerated,n=7,q=q)==manifest['certificate_stats'],'regenerated tree replay differs')
            print('Verified exact executable search:',q,actual)

def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--check',action='store_true');arg.add_argument('--cpp-check',action='store_true')
    a=arg.parse_args()
    if a.cpp_check:recompile_check()
    else:check()
if __name__=='__main__':main()
