"""OIC-26 n=6 exact one-marked-erasure feasibility reference.

Unquotiented and phase-translation-quotiented solvers independently reject
Q=10. A Q=11 policy is checked by true-state/fault-path enumeration.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios
from oic24_sym import SymFaultGame

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic26_reference.json'
ARCHIVE=HERE/'results/oic26_n6_erasure_tree.json.gz.b64'

def sha(b):return hashlib.sha256(b).hexdigest()

def build():
    no=ExactFaultGame(6,10,0,1,node_cap=2400000)
    if no.solve():raise AssertionError('unexpected unquotiented ten-query policy')
    if no.visited!=2223374:raise AssertionError('unquotiented state count changed')
    reduced=SymFaultGame(6,10,0,1,node_cap=500000)
    if reduced.solve():raise AssertionError('unexpected symmetry-reduced ten-query policy')
    if reduced.visited!=370580:raise AssertionError('reduced state count changed')
    yes=SymFaultGame(6,11,0,1,node_cap=100000)
    if not yes.solve():raise AssertionError('missing eleven-query policy')
    if yes.visited!=27577:raise AssertionError('positive state count changed')
    tree=yes.certificate()
    independent=ExactFaultGame(6,11,0,1)
    nodes=independent.verify_tree(tree)
    paths=replay_true_scenarios(6,11,0,1,tree)
    if (nodes,paths)!=(753,331):raise AssertionError('failed independent replay')
    blob=json.dumps(tree,sort_keys=True,separators=(',',':')).encode('utf8')
    comp=gzip.compress(blob,mtime=0)
    manifest={
        'schema':'OIC26-N6-ERASURE-V1',
        'n':6,'substitutions':0,'marked_erasures':1,
        'minimum_adaptive_queries':11,
        'negative_horizon':10,
        'negative_unquotiented_states':no.visited,
        'negative_translation_quotient_states':reduced.visited,
        'winning_translation_quotient_states':yes.visited,
        'winning_tree_nodes':nodes,
        'true_world_fault_paths_checked':paths,
        'policy_canonical_json_sha256':sha(blob),
        'policy_gzip_sha256':sha(comp),
        'scope':'finite_computer_assisted_exact; no general_n_theorem',
    }
    return manifest,tree,comp

def check(recompute:bool=False):
    m=json.loads(MANIFEST.read_text(encoding='utf8'))
    comp=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
    raw=gzip.decompress(comp)
    if sha(comp)!=m['policy_gzip_sha256'] or sha(raw)!=m['policy_canonical_json_sha256']:
        raise AssertionError('archive digest failure')
    tree=json.loads(raw)
    independent=ExactFaultGame(6,11,0,1)
    if independent.verify_tree(tree)!=m['winning_tree_nodes']:
        raise AssertionError('wrong policy node count')
    if replay_true_scenarios(6,11,0,1,tree)!=m['true_world_fault_paths_checked']:
        raise AssertionError('wrong true-state replay')
    if recompute:
        new,newtree,_=build()
        for k,v in new.items():
            if k!='policy_gzip_sha256' and m.get(k)!=v:
                raise AssertionError('regenerated manifest mismatch: '+k)
        if newtree!=tree:raise AssertionError('regenerated policy mismatch')
    return m

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--write',action='store_true')
    a.add_argument('--check',action='store_true')
    a.add_argument('--recompute',action='store_true')
    args=a.parse_args()
    if args.write:
        m,_,comp=build()
        MANIFEST.parent.mkdir(exist_ok=True,parents=True)
        MANIFEST.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf8')
        ARCHIVE.write_text(base64.b64encode(comp).decode('ascii')+'\n',encoding='ascii')
        print('wrote OIC-26 reference')
    if args.check or args.recompute:
        x=check(recompute=args.recompute)
        print('verified n6 erasure minimum',x['minimum_adaptive_queries'],'with',x['winning_tree_nodes'],'policy nodes')
    if not(args.write or args.check or args.recompute):a.print_help()

if __name__=='__main__':main()
