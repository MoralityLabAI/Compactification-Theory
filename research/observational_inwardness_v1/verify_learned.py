"""OIC-12 frozen GRU receipt verifier; Python standard library only.

Checks the archived experimental trajectories against the exact, independently
reimplemented finite-state oracle and against their reported metrics. Training
itself requires PyTorch and is outside lightweight CI.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / 'results' / 'learned_pilot_summary.json'
DEFAULT_ARCHIVE = HERE / 'results' / 'learned_pilot_full.json.gz.b64'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def oracle_trace(n: int, family: str, initial_phase: int, action_word: str) -> list[int]:
    require(family in {'fixed', 'rotating'}, 'unknown family')
    z = initial_phase
    x = 0
    states = [x]
    for ch in action_word:
        require(ch in ('0', '1'), 'nonbinary action')
        active = int(ch)
        x ^= 1 ^ (active & int(z == 0))
        z = (z + int(family == 'rotating')) % n
        states.append(x)
    return states


def hamming_distance(a: list[int], b: list[int]) -> int:
    require(len(a) == len(b), 'trace length mismatch')
    return sum(x != y for x, y in zip(a, b))


def cross_distance(a: list[list[int]], b: list[list[int]]) -> int:
    return min(hamming_distance(x, y) for x in a for y in b)


def decode_archive(encoded: bytes) -> tuple[dict, bytes, bytes]:
    compressed = base64.b64decode(encoded.strip(), validate=True)
    payload = gzip.decompress(compressed)
    return json.loads(payload), payload, compressed


def validate_payload(payload: dict) -> list[dict]:
    require(payload.get('protocol', {}).get('tag') == 'OIC-12-CPU-GRU-01', 'protocol tag mismatch')
    runs = payload['runs']
    require(len(runs) == 30, 'expected 30 matched model-family pairs')
    require(len({(r['n'],r['training_split'],r['seed']) for r in runs}) == 30, 'duplicate run')
    require({r['n'] for r in runs} == {4,5,6}, 'n coverage mismatch')
    require({r['seed'] for r in runs} == set(range(5)), 'seed coverage mismatch')
    require({r['training_split'] for r in runs} == {'full','heldout'}, 'phase split coverage mismatch')
    expected_schedules = {'latch','latch_long','always_probe','never_probe','alternating','random_budget'}
    for r in runs:
        n = r['n']; split = r['training_split']; seed = r['seed']
        require(len(r['evaluations']) == len(expected_schedules), 'schedule count')
        require(set(r['evaluations']) == expected_schedules, 'schedule names')
        for fam in ('fixed','rotating'):
            train=r['training'][fam]
            expected_phase_set = list(range(n)) if split == 'full' else [p for p in range(n) if p not in {1,n-2}]
            require(list(train['train_phase_indices']) == expected_phase_set, 'train phase mismatch')
            require(list(train['heldout_phase_indices']) == [1,n-2], 'heldout indices mismatch')
            require(train['train_steps']==350 and train['train_batch_size']==96, 'training budget drift')
            require(train['parameter_count']==2185, 'architecture drift')
            require(train['final_teacher_loss'] >= 0, 'negative loss')
        for name, entry in r['evaluations'].items():
            word=entry['word']
            require(set(word)<=set('01') and len(word)>=1, 'invalid schedule')
            if name=='latch':require(word=='1'*n+'0'*n, 'latch changed')
            if name=='latch_long':require(word=='1'*n+'0'*(2*n), 'long latch changed')
            if name=='always_probe':require(word=='1'*(2*n), 'all active changed')
            if name=='never_probe':require(word=='0'*(2*n), 'all passive changed')
            if name=='alternating':require(word==''.join('1' if i%2==0 else '0' for i in range(2*n)), 'alternating changed')
            if name=='random_budget':require(len(word)==2*n and word.count('1')==n, 'budget random changed')
            for fam in ('fixed','rotating'):
                part = entry['families'][fam]
                observed = part['predicted_state_traces']
                reference = part['exact_state_traces']
                require(len(observed)==n and len(reference)==n, 'phase count mismatch')
                calculated = [oracle_trace(n,fam,z,word) for z in range(n)]
                require(reference==calculated, 'stored oracle trace mismatch')
                require(all(len(trace)==len(word)+1 and trace[0]==0 and set(trace)<= {0,1} for trace in observed), 'invalid emitted trace')
                exact_flags = [int(x==y) for x,y in zip(observed,calculated)]
                require(exact_flags==part['whole_trace_exact_by_phase'], 'incorrect exact-match record')
                wrong=sum(hamming_distance(x,y) for x,y in zip(observed,calculated))
                require(wrong==part['total_wrong_raw_state_samples'], 'wrong bit-error count')
            a=entry['families']['fixed']; b=entry['families']['rotating']
            true_d=cross_distance(a['exact_state_traces'],b['exact_state_traces'])
            model_d=cross_distance(a['predicted_state_traces'],b['predicted_state_traces'])
            require(entry['oracle_code_distance']==true_d, 'oracle distance incorrect')
            require(entry['learned_code_distance']==model_d, 'learned distance incorrect')
            for label,key in (('oracle','exact_state_traces'),('learned','predicted_state_traces')):
                overlap=bool({t[-1] for t in a[key]} & {t[-1] for t in b[key]})
                require(entry[f'{label}_terminal_overlap']==overlap, 'terminal overlap incorrect')
            if name=='latch' and n%2==0:
                require(true_d==n+1,'exact even-phase latch theorem violated')
            if name=='latch' and n%2==1:
                require(entry['oracle_terminal_overlap'],'odd phase terminal obstruction violated')
    summaries=[]
    for n in (4,5,6):
        for split in ('full','heldout'):
            subset=sorted((r for r in runs if r['n']==n and r['training_split']==split),key=lambda r:r['seed'])
            require([r['seed'] for r in subset]==list(range(5)), 'uneven seed group')
            lat=[r['evaluations']['latch'] for r in subset]
            distances=[v['learned_code_distance'] for v in lat]
            reference=[v['oracle_code_distance'] for v in lat]
            summaries.append({
                'n':n, 'training_split':split,'seeds':list(range(5)),
                'latch_full_trace_accuracy_fixed':sum(sum(v['families']['fixed']['whole_trace_exact_by_phase']) for v in lat)/(n*5),
                'latch_full_trace_accuracy_rotating':sum(sum(v['families']['rotating']['whole_trace_exact_by_phase']) for v in lat)/(n*5),
                **{f'latch_{fam}_{group}_initial_phase_accuracy':(
                    sum(sum(value for z,value in enumerate(v['families'][fam]['whole_trace_exact_by_phase']) if (z in {1,n-2}) == (group == 'withheld')) for v in lat)
                    /(5*(2 if group == 'withheld' else n-2))
                    ) for fam in ('fixed','rotating') for group in ('trained','withheld')},
                'latch_empirical_min_code_distances':distances,
                'latch_oracle_min_code_distances':reference,
                'runs_identifiable_without_noise':sum(d>=1 for d in distances),
                'runs_identifiable_with_one_raw_substitution':sum(d>=3 for d in distances),
                'runs_with_exact_oracle_code_distance':sum(d==o for d,o in zip(distances,reference)),
                'total_runs':len(subset),
            })
    return summaries


def verify(summary_path: Path = DEFAULT_SUMMARY, archive_path: Path = DEFAULT_ARCHIVE, check_code: bool = True) -> dict:
    manifest=json.loads(summary_path.read_text(encoding='utf8'))
    archive, decoded, compressed = decode_archive(archive_path.read_bytes())
    require(manifest['full_receipt_decoded_sha256']==hashlib.sha256(decoded).hexdigest(), 'payload hash mismatch')
    require(manifest['full_receipt_gzip_sha256']==hashlib.sha256(compressed).hexdigest(), 'gzip hash mismatch')
    require(archive['code_sha256']==manifest['training_code_sha256'], 'source hash metadata mismatch')
    if check_code:
        source_sha=hashlib.sha256((HERE/'learned_pilot.py').read_bytes()).hexdigest()
        require(archive['code_sha256']==source_sha, 'training implementation differs from frozen run')
    calculated=validate_payload(archive)
    require(calculated==manifest['summary'], 'summary derived from payload mismatches')
    require(manifest['model_fits']==60 and manifest['run_count']==30, 'run count incorrect')
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    manifest=verify()
    print('OIC-12 verified: 60 trained GRUs, 30 run-pairs, 6 schedule families, five seeds')
    for row in manifest['summary']:
        print(f"n={row['n']} split={row['training_split']} "
              f"trace={row['latch_full_trace_accuracy_fixed']:.1%}/{row['latch_full_trace_accuracy_rotating']:.1%} "
              f"dist={row['latch_empirical_min_code_distances']}")


if __name__=='__main__':
    main()
