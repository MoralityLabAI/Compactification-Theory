"""OIC-24 phase-translation quotient and n=5 finite robust-query replay."""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json
from pathlib import Path
from oic24_sym import SymFaultGame
from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios, best_fixed_schedule
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic24_reference.json'
TREE=HERE/'results/oic24_n5_erasure_policy.json.gz.b64'


def experiment()->tuple[dict,dict]:
    # Negative erasure result uses the *unquotiented* independent OIC-23 oracle.
    impossible=ExactFaultGame(5,8,0,1,node_cap=200000)
    if impossible.solve():raise AssertionError('unexpected 8-query robust erasure policy')
    possible=SymFaultGame(5,9,0,1,node_cap=200000)
    if not possible.solve():raise AssertionError('nine-query erasure policy absent')
    policy=possible.certificate()
    nodes=possible.verify_tree(policy)
    scenarios=replay_true_scenarios(5,9,0,1,policy)
    # Candidate translation quotient is complete and found the negative answer
    # after exploring fewer than the declared 600k states.
    flip_impossible=SymFaultGame(5,11,1,0,node_cap=600000)
    if flip_impossible.solve():raise AssertionError('unexpected 11-query one-flip policy')
    fixed_erasure=[]
    for size,horizon in ((2,4),(3,5),(4,7),(5,9)):
        distance,word,count=best_fixed_schedule(size,horizon)
        if distance!=(2 if size==2 else 1):
            raise AssertionError('fixed one-erasure distance reference changed')
        fixed_erasure.append({'n':size,'query_budget':horizon,
                              'best_fixed_cross_shift_distance':distance,
                              'best_fixed_word':list(word),'exhaustive_schedules':count,
                              'fixed_schedule_one_erasure_correctable':distance>=2})
    blob=json.dumps(policy,sort_keys=True,separators=(',',':')).encode('utf8')
    comp=gzip.compress(blob,mtime=0)
    manifest={
        'schema':'OIC24-REFERENCE-V1','n':5,
        'erasure_only':{
            'minimum_queries':9,'impossible_at':8,
            'unquotiented_failure_states':impossible.visited,
            'symmetry_quotient_success_states':possible.visited,
            'policy_tree_nodes':nodes,'independent_true_world_paths':scenarios,
        },
        'fixed_one_erasure_comparison':fixed_erasure,
        'one_substitution':{
            'known_impossible_through':11,
            'symmetry_quotient_failure_states':flip_impossible.visited,
            'Q12_status':'inconclusive_700000_state_cap',
        },
        'policy_canonical_json_sha256':hashlib.sha256(blob).hexdigest(),
        'policy_gzip_sha256':hashlib.sha256(comp).hexdigest(),
        'evidence_level':'exact_n5_erasure_and_one_flip_lower_bound; no upper_bound_for_one_flip',
    }
    return manifest,policy


def verify():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    comp=base64.b64decode(TREE.read_bytes().strip(),validate=True)
    blob=gzip.decompress(comp)
    if hashlib.sha256(comp).hexdigest()!=manifest['policy_gzip_sha256'] or hashlib.sha256(blob).hexdigest()!=manifest['policy_canonical_json_sha256']:
        raise AssertionError('stored policy artifact integrity failure')
    tree=json.loads(blob)
    if replay_true_scenarios(5,9,0,1,tree)!=manifest['erasure_only']['independent_true_world_paths']:
        raise AssertionError('independent replay disagrees with receipt')
    new,regen=experiment()
    if tree!=regen or {k:v for k,v in manifest.items() if k!='policy_gzip_sha256'}!={k:v for k,v in new.items() if k!='policy_gzip_sha256'}:
        raise AssertionError('cross-platform canonical certificate regeneration failed')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if args.check:
        x=verify()
        print('OIC-24 exact n=5 erasure certificate and one-flip lower bound passed:',x['erasure_only'])
        return
    m,tree=experiment()
    blob=json.dumps(tree,sort_keys=True,separators=(',',':')).encode('utf8')
    comp=gzip.compress(blob,mtime=0)
    MANIFEST.parent.mkdir(exist_ok=True,parents=True)
    MANIFEST.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf8')
    TREE.write_text(base64.b64encode(comp).decode('ascii')+'\n',encoding='utf8')
    print('wrote',MANIFEST,TREE)

if __name__=='__main__':main()
