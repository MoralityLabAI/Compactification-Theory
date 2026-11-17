"""OIC-27 pinned source, proof-tree, and fixed-schedule experiment verifier.

Only Python standard library is required for --check. --write consumes the
independently compiled solver certificates. The expensive/optional negative
searches and SciPy/HiGHS MILP are separate executable sources.
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
from oic27_verify import verify

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
ARCHIVE=RESULTS/'oic27_n6_flip_policies.json.gz.b64'
MANIFEST=RESULTS/'oic27_reference.json'
SRC_FILES=['oic27_fast.cpp','oic27_affine.cpp','oic27_verify.py','oic27_fixed_milp.py']
WITNESS_21=(0,2,2,3,3,4,2,5,3,4,4,3,3,1,4,5,5,5,1,0,5)


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def minimum_cross(word:tuple[int,...],n:int=6)->int:
    """Independently compute minimum Hamming distance across unequal shifts."""
    code={}
    for d in range(n):
        code[d]=[tuple(int((z+t*d)%n==m) for t,m in enumerate(word)) for z in range(n)]
    return min(sum(u!=v for u,v in zip(a,b))
               for d in range(n) for d2 in range(d+1,n)
               for a in code[d] for b in code[d2])


def fixed_schedule_parity_lower_bound(n:int=6,e:int=1):
    """Special n=6 pair-separation argument: 3 disjoint antipodal pairs.

    Across each time parity, every pair of opposite markers needs >=2e+1
    addressed observations; hence each parity needs >=3*(2e+1) slots.
    """
    if n!=6 or e<0:raise ValueError('specific six-phase construction, e>=0')
    return n*(2*e+1)


def build():
    policies={}
    reviews={}
    for tag,file in [('translation','winner_n6_q15.json'),('affine','winner_affine_n6_q15.json')]:
        tree=json.loads((HERE/file).read_text(encoding='utf8'))
        result=verify(tree,15)
        policies[tag]=tree
        reviews[tag]=result
    payload=json.dumps(policies,separators=(',',':'),sort_keys=True).encode('utf8')
    zipped=gzip.compress(payload,mtime=0)
    manifest={
        'schema':'OIC27-N6-FLIP-V1',
        'n':6,'max_substitutions':1,'marked_erasures':0,
        'exact_minimum_adaptive_queries':15,
        'exhaustive_negative_horizon':14,
        'negative_translation_states':1639144,
        'negative_affine_states':813921,
        'positive_translation_states':214548,
        'positive_affine_states':210059,
        'positive_certificate_checks':reviews,
        'certificates_canonical_json_sha256':digest(payload),
        'certificates_gzip_sha256':digest(zipped),
        'source_sha256':{name:digest((HERE/name).read_bytes()) for name in SRC_FILES},
        'fixed_schedule':{
            'analytically_proved_necessary_queries':fixed_schedule_parity_lower_bound(),
            'milp_scipy_highs_q20_status':'INFEASIBLE',
            'milp_q21_status':'FEASIBLE',
            'milp_q21_witness':list(WITNESS_21),
            'verified_q21_min_cross_shift_distance':minimum_cross(WITNESS_21),
            'claim_grade':'fixed 21-query minimality requires SciPy HiGHS 0-1 feasibility solver; 18-query lower is analytic',
        },
        'online_resource':{'adaptive_queries':15,'returned_bits':15,'address_bits_direct':45,
                           'total_bits_direct':60,'candidate_layer_logical_bits':72},
        'evidence_scope':'finite computer-assisted exact; full policy trees independently validated',
    }
    return manifest,zipped


def write():
    data,zipped=build()
    RESULTS.mkdir(exist_ok=True,parents=True)
    MANIFEST.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')
    ARCHIVE.write_text(base64.b64encode(zipped).decode('ascii')+'\n',encoding='ascii')
    return data


def check():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    compressed=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
    raw=gzip.decompress(compressed)
    if digest(raw)!=manifest['certificates_canonical_json_sha256']:
        raise ValueError('policy JSON SHA-256 mismatch')
    if digest(compressed)!=manifest['certificates_gzip_sha256']:
        raise ValueError('policy gzip SHA-256 mismatch')
    payload=json.loads(raw)
    if set(payload)!=set(('affine','translation')):raise ValueError('certificate family mismatch')
    for name,policy in payload.items():
        observed=verify(policy,15)
        if observed!=manifest['positive_certificate_checks'][name]:
            raise ValueError(f'policy verification changed: {name}')
    for file, expected in manifest['source_sha256'].items():
        if digest((HERE/file).read_bytes())!=expected:
            raise ValueError('source SHA-256 mismatch: '+file)
    fixed=manifest['fixed_schedule']
    if minimum_cross(tuple(fixed['milp_q21_witness']))!=3:
        raise ValueError('fixed MILP witness does not correct one substitution')
    if fixed['analytically_proved_necessary_queries']!=18:
        raise ValueError('analytical fixed lower bound changed')
    if manifest['exact_minimum_adaptive_queries']!=15:
        raise ValueError('adaptive horizon mismatch')
    if manifest['online_resource']['total_bits_direct']!=60:
        raise ValueError('link ledger mismatch')
    return manifest


def cpp_check():
    """Independently recompile both exact solvers and check Q14/Q15 from source.

    A resource-limit error or timeout is an inconclusive search and fails here.
    No external packages beyond a C++17 compiler and Python are required.
    """
    saved=json.loads(gzip.decompress(base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)))
    expected=json.loads(MANIFEST.read_text(encoding='utf8'))
    variants=[
        ('translation','oic27_fast.cpp',2500000,['2','1']),
        ('affine','oic27_affine.cpp',2000000,[]),
    ]
    with tempfile.TemporaryDirectory(prefix='oic27-') as folder:
        for variant,source,limit,extra in variants:
            executable=Path(folder)/variant
            subprocess.run(['g++','-O3','-std=c++17',str(HERE/source),'-o',str(executable)],
                           capture_output=True,text=True,check=True,timeout=50)
            for Q in (14,15):
                policy_path=Path(folder)/f'{variant}-q{Q}.json'
                args=[str(executable),str(Q),str(limit),*extra]
                if Q==15:args.append(str(policy_path))
                result=subprocess.run(args,capture_output=True,text=True,timeout=50)
                if result.returncode!=0:
                    raise RuntimeError(f'{variant} Q={Q} returned non-success (possibly CAP): '
                                       f'{result.stdout} {result.stderr}')
                if f'STATUS {"UNSAT" if Q==14 else "SAT"} Q={Q}' not in result.stdout:
                    raise ValueError(f'{variant} unexpected exact solver output Q={Q}: {result.stdout}')
                field='negative' if Q==14 else 'positive'
                pattern=r'(?:visited|states)=(\d+)'
                match=re.search(pattern,result.stdout)
                if match is None:raise ValueError('solver state count missing')
                if int(match.group(1))!=expected[f'{field}_{variant}_states']:
                    raise ValueError(f'{variant} Q={Q} state count mismatch')
                if Q==15:
                    produced=json.loads(policy_path.read_text(encoding='utf8'))
                    if produced!=saved[variant]:
                        raise ValueError(f'{variant} winning tree differs from pinned certificate')
                    if verify(produced,15)!=expected['positive_certificate_checks'][variant]:
                        raise ValueError(f'{variant} independent replay mismatch')
                print(f'{variant} Q={Q}: PASS ({match.group(1)} exact states)')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--cpp-check',action='store_true')
    args=parser.parse_args()
    if args.write:
        receipt=write()
        print('wrote pinned OIC27 n6 one-flip certificates')
    if args.check:
        receipt=check()
        print('PASS n6 adaptive minimum',receipt['exact_minimum_adaptive_queries'],
              'trees',[(x,v['belief']['nodes'],v['real_world']['paths'])
                       for x,v in receipt['positive_certificate_checks'].items()])
    if args.cpp_check:
        cpp_check()
    if not(args.write or args.check or args.cpp_check):parser.print_help()

if __name__=='__main__':main()
