"""OIC-23 finite reproducible certificate generator and independent verifier.

No third-party dependencies. Certificates are full adaptive decision trees;
correctness is independently checked by enumerating every true (d,z0), every
possible *online* fault path, and the sensor's actual output at each query.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path

from oic23_exact import ExactFaultGame

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic23_reference.json'
CERTIFICATE=HERE/'results/oic23_certificates.json.gz.b64'
CASE_Q={2:{(1,0):6,(0,1):4,(1,1):8},
        3:{(1,0):8,(0,1):5,(1,1):9},
        4:{(1,0):10,(0,1):7,(1,1):12}}


def replay_true_scenarios(n:int, Q:int, e:int,r:int,tree:dict)->int:
    """Independent trajectory-level, online adversarial replay of all actions.

    Enumerates EVERY hidden shift and initial phase plus all legal online
    substitutions/erasures. This deliberately does not use solver candidate masks.
    """
    total=0
    def walk(d,z,t,used_e,used_r,node):
        nonlocal total
        if 'shift' in node:
            if node['shift'] != d:raise AssertionError((n,d,z,t,'wrong shift',node['shift']))
            total+=1
            return
        if t >= Q:raise AssertionError((n,d,z,t,'query after horizon'))
        m=node['query']
        if not isinstance(m,int) or not 0<=m<n:raise AssertionError('invalid address')
        signal=int((z+t*d)%n==m)
        possible=[(str(signal),used_e,used_r)]
        if used_e<e:possible.append((str(1-signal),used_e+1,used_r))
        if used_r<r:possible.append(('?',used_e,used_r+1))
        for k,e2,r2 in possible:
            if k not in node['children']:raise AssertionError((n,d,z,t,k,'missing child'))
            walk(d,z,t+1,e2,r2,node['children'][k])
    for d,z in product(range(n),repeat=2):walk(d,z,0,0,0,tree)
    return total


def min_interfamily_distance(n:int,word:tuple[int,...])->int:
    """Cross-shift codebook distance for one fixed sensor-address word."""
    if n<2:raise ValueError('n>=2')
    groups=[]
    for d in range(n):
        groups.append(tuple(sum(1<<t for t,m in enumerate(word) if (z+t*d)%n==m)
                            for z in range(n)))
    return min((a^b).bit_count() for d in range(n) for d2 in range(d+1,n)
               for a in groups[d] for b in groups[d2])


def best_fixed_schedule(n:int,Q:int)->tuple[int,tuple[int,...],int]:
    """Exhaustive fixed policies up to phase translation; first address = 0."""
    if Q<1:raise ValueError('Q>=1')
    best=-1;word_best=None;searched=0
    for suffix in product(range(n),repeat=Q-1):
        word=(0,)+suffix
        d=min_interfamily_distance(n,word)
        searched+=1
        if d>best:
            best=d;word_best=word
    if word_best is None:raise AssertionError('fixed schedule search had zero candidates')
    return best,word_best,searched


def deterministic_gzip(blob:bytes)->bytes:
    return gzip.compress(blob,mtime=0)


def build_reference()->tuple[dict,dict]:
    entries=[]
    trees=[]
    for n in CASE_Q:
        for e,r in ((1,0),(0,1),(1,1)):
            Q=CASE_Q[n][e,r]
            impossible=ExactFaultGame(n,Q-1,e,r)
            if impossible.solve():raise AssertionError('claimed impossible horizon has a policy')
            solver=ExactFaultGame(n,Q,e,r)
            if not solver.solve():raise AssertionError('claimed feasible horizon has no policy')
            tree=solver.certificate()
            num_nodes=solver.verify_tree(tree)
            population_scenarios=replay_true_scenarios(n,Q,e,r,tree)
            entries.append({'n':n,'substitutions':e,'marked_erasures':r,
                            'necessary_general_lower_bound':n*(e+1)+r,
                            'minimum_queries':Q,'failed_horizon':Q-1,
                            'failed_states_explored':impossible.visited,
                            'successful_states_explored':solver.visited,
                            'policy_tree_nodes':num_nodes,
                            'true_world_fault_paths_checked':population_scenarios})
            trees.append({'n':n,'substitutions':e,'marked_erasures':r,'Q':Q,'policy':tree})
    fixed=[]
    for n in (2,3,4):
        Q=CASE_Q[n][1,0]
        best,word,num=best_fixed_schedule(n,Q)
        fixed.append({'n':n,'queries':Q,'best_fixed_cross_shift_hamming_distance':best,
                      'optimal_word':list(word),'exhaustive_fixed_schedules':num,
                      'single_substitution_correctable_by_fixed':best>=3})
    certificate={'schema':'OIC23-ADAPTIVE-CERT-V1','cases':trees}
    bytes_=json.dumps(certificate,sort_keys=True,separators=(',',':')).encode('utf-8')
    compressed=deterministic_gzip(bytes_)
    manifest={'schema':'OIC23-EXACT-REFERENCE-V1','status':'exhaustive_finite_certificates',
              'threshold_cases':entries,'fixed_schedule_comparison':fixed,
              'certificate_sha256':hashlib.sha256(bytes_).hexdigest(),
              'certificate_gzip_sha256':hashlib.sha256(compressed).hexdigest(),
              'certificate_cases':len(trees),
              'n5_status':'search_aborted_at_300000_nodes_in_initial_branch; inconclusive'}
    return manifest,certificate


def replay_stored():
    saved=json.loads(MANIFEST.read_text(encoding='utf-8'))
    comp=base64.b64decode(CERTIFICATE.read_bytes().strip(),validate=True)
    unpack=gzip.decompress(comp)
    if hashlib.sha256(comp).hexdigest()!=saved['certificate_gzip_sha256'] or hashlib.sha256(unpack).hexdigest()!=saved['certificate_sha256']:
        raise AssertionError('stored certificate hash mismatch')
    cert=json.loads(unpack)
    if cert['schema']!='OIC23-ADAPTIVE-CERT-V1':raise AssertionError('bad schema')
    if len(cert['cases'])!=9:raise AssertionError('incomplete certificate case coverage')
    for case in cert['cases']:
        n,Q,e,r=case['n'],case['Q'],case['substitutions'],case['marked_erasures']
        g=ExactFaultGame(n,Q,e,r)
        independent=replay_true_scenarios(n,Q,e,r,case['policy'])
        expected=next(x for x in saved['threshold_cases'] if x['n']==n and x['substitutions']==e and x['marked_erasures']==r)
        if independent!=expected['true_world_fault_paths_checked']:raise AssertionError('scenario mismatch')
        if g.verify_tree(case['policy'])!=expected['policy_tree_nodes']:raise AssertionError('tree mismatch')
    generated,regenerated_tree=build_reference()
    # Python versions may produce byte-distinct gzip streams even with mtime=0.
    # The stored archive's *own* gzip SHA was already verified above. For
    # independent regeneration, verify the decoded canonical certificate and
    # every platform-independent field of the manifest instead of demanding
    # identical compressed bytes across Python/zlib versions.
    portable_saved={k:v for k,v in saved.items() if k!='certificate_gzip_sha256'}
    portable_generated={k:v for k,v in generated.items() if k!='certificate_gzip_sha256'}
    if portable_saved!=portable_generated or cert!=regenerated_tree:
        raise AssertionError('semantic reference mismatch: solver failed to reproduce the certificate')
    return saved


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    if args.check:
        x=replay_stored()
        print('OIC-23 exact adaptive and fixed-schedule certificate replay passed:',len(x['threshold_cases']),'cases')
        return
    manifest,cert=build_reference()
    blob=json.dumps(cert,sort_keys=True,separators=(',',':')).encode('utf-8')
    compressed=deterministic_gzip(blob)
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    CERTIFICATE.write_text(base64.b64encode(compressed).decode('ascii')+'\n',encoding='utf-8')
    print('Wrote',MANIFEST,CERTIFICATE)

if __name__=='__main__':main()
