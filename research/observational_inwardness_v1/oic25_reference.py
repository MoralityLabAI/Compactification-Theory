"""OIC-25 finite five-phase robust-query certificate and exact replay.

Uses the separately written Python dynamic program and OIC-23's independent
trajectory-level certificate validator; optional C++ cross-check available.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios
from oic25_independent import PythonPrunedGame

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic25_reference.json'
ARCHIVE=HERE/'results/oic25_n5_q14_tree.json.gz.b64'
EXPECTED={11:(False,35542),12:(False,125216),13:(False,379887),14:(True,922)}


def canonical_bytes(tree):
    return json.dumps(tree,sort_keys=True,separators=(',',':')).encode('utf-8')


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def build():
    rows=[]
    winning_tree=None
    for q,(accepted,visited) in EXPECTED.items():
        solver=PythonPrunedGame(q,node_cap=1000000)
        found=solver.solve()
        if found != accepted or solver.visited != visited:
            raise AssertionError(f'changed independent Python result for Q={q}: {found}/{solver.visited}')
        if q==14:
            winning_tree=solver.certificate()
        rows.append({'query_horizon':q,'exists_winning_policy':found,'python_visited_canonical_states':solver.visited})
    if winning_tree is None:
        raise AssertionError('missing 14-query tree')
    exact=ExactFaultGame(5,14,1,0)
    nodes=exact.verify_tree(winning_tree)
    paths=replay_true_scenarios(5,14,1,0,winning_tree)
    if (nodes,paths)!=(662,327):
        raise AssertionError(f'exact policy changed: nodes={nodes}, paths={paths}')
    raw=canonical_bytes(winning_tree)
    compressed=gzip.compress(raw,mtime=0)
    hashes={path:sha_bytes((HERE/path).read_bytes()) for path in ('oic25_fast.cpp','oic25_independent.py')}
    manifest={
        'schema':'OIC25-N5-ROBUST-FLIP-V1',
        'model':'z_t=(z0+t*d)%5; query sensor address m; clean report [z_t=m]',
        'n':5,'max_bit_substitutions':1,'marked_erasures':0,
        'minimum_queries':14,
        'python_exact_searches':rows,
        'cpp_no_pair_pruning_q13':{'exists_winning_policy':False,'visited_canonical_states':2597699},
        'cpp_pair_pruning_q13':{'exists_winning_policy':False,'visited_canonical_states':379889},
        'cpp_q14':{'exists_winning_policy':True,'visited_canonical_states':926},
        'policy_tree_nodes':nodes,'independent_world_fault_paths_checked':paths,
        'policy_canonical_json_sha256':sha_bytes(raw),
        'policy_gzip_sha256':sha_bytes(compressed),
        'source_sha256':hashes,
        'claim_status':'exhaustive_finite_computer_assisted; no Lean proof',
    }
    return manifest,winning_tree,compressed


def verify():
    saved=json.loads(MANIFEST.read_text(encoding='utf8'))
    comp=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
    raw=gzip.decompress(comp)
    if sha_bytes(comp)!=saved['policy_gzip_sha256'] or sha_bytes(raw)!=saved['policy_canonical_json_sha256']:
        raise AssertionError('policy hash mismatch')
    stored=json.loads(raw)
    game=ExactFaultGame(5,14,1,0)
    if game.verify_tree(stored)!=saved['policy_tree_nodes']:
        raise AssertionError('policy tree node count mismatch')
    if replay_true_scenarios(5,14,1,0,stored)!=saved['independent_world_fault_paths_checked']:
        raise AssertionError('trajectory-level fault replay mismatch')
    current,regenerated,_=build()
    if current['source_sha256'] != saved['source_sha256']:
        raise AssertionError('solver source hashes changed')
    for name in current:
        if name=='policy_gzip_sha256':continue  # zlib outputs can vary by platform
        if current[name]!=saved[name]:
            raise AssertionError(f'regenerated reference mismatch: {name}')
    if stored!=regenerated:
        raise AssertionError('Python search reconstructed a different decision tree')
    return current


def cpp_verify():
    gxx=shutil.which('g++')
    if not gxx:raise RuntimeError('g++ required for compiled independent replay')
    with tempfile.TemporaryDirectory(prefix='oic25-replay-') as directory:
        binary=Path(directory)/'oic25_fast'
        subprocess.run([gxx,'-std=c++17','-O3',str(HERE/'oic25_fast.cpp'),'-o',str(binary)],check=True)
        for q,prune,wanted,wanted_states in ((13,0,'UNSAT',2597699),(13,1,'UNSAT',379889),(14,1,'SAT',926)):
            tree_path=Path(directory)/'tree.json'
            cmd=[str(binary),str(q),'5000000','1','3000000',str(prune),str(tree_path)]
            proc=subprocess.run(cmd,check=True,capture_output=True,text=True)
            line=proc.stdout.splitlines()[0] if proc.stdout else ''
            if f'STATUS {wanted} ' not in line:
                raise AssertionError(f'unexpected compiled solver status: {line}')
            v=re.search(r'\bnodes=(\d+)',line)
            if v is None or int(v.group(1))!=wanted_states:
                raise AssertionError(f'unexpected compiled solver visit count: {line}')
            if wanted=='SAT':
                tree=json.loads(tree_path.read_text())
                comp=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
                saved=json.loads(gzip.decompress(comp))
                if tree!=saved:
                    raise AssertionError('compiled solver built a different winning tree')
            print('C++ exact search:',line,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--cpp-check',action='store_true')
    args=p.parse_args()
    if args.write:
        data,tree,comp=build()
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        MANIFEST.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')
        ARCHIVE.write_text(base64.b64encode(comp).decode('ascii')+'\n',encoding='ascii')
        print('Wrote OIC-25 manifest and policy certificate')
    if args.check:
        d=verify()
        print(f"OIC-25 verified Q*={d['minimum_queries']}; policy nodes={d['policy_tree_nodes']}; true/fault paths={d['independent_world_fault_paths_checked']}")
    if args.cpp_check:
        cpp_verify()
    if not(args.write or args.check or args.cpp_check):p.print_help()

if __name__=='__main__':main()
