"""SCM-21: retrospective exact-count audit of SCM-20 rank-four interchange.

Inputs: public source-locked SCM-20 summary (optional) or its explicit distilled
visit-two snapshot. This module never loads trained weights or hidden gold.
Only method-level counts are available; pairwise overlap and norm-matched
causal effects remain unidentified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / 'results' / 'scm20_visit2_snapshot.json'
METHODS = ('full_cls', 'head_readable', 'head_fiber', 'pca4',
           'semantic_diff4', 'sae_group4', 'native_write4',
           'native_vpd_write', 'random_fiber4', 'sham')
SEEDS = (191, 1191, 2191)
STRATA = ('id', 'ood')


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _source_pack(payload: dict) -> list[dict]:
    if payload.get('schema') != 'scm20.summary.v1':
        raise ValueError('unexpected SCM-20 summary schema')
    if not payload.get('audit', {}).get('passed'):
        raise ValueError('SCM-20 source audit did not pass')
    if len(payload.get('secondary', [])) != 720:
        raise ValueError('incomplete SCM-20 method/visit/model matrix')
    bykey = {}
    for x in payload['secondary']:
        key = (x['seed'], x['stratum'], x['model'], x['visit'], x['method'])
        if key in bykey:
            raise ValueError(f'duplicate cell {key}')
        if x['n'] < 0 or not 0 <= x['correct'] <= x['n']:
            raise ValueError('invalid counts')
        bykey[key] = x
    rows = []
    for seed in SEEDS:
        for stratum in STRATA:
            cells = [bykey[(seed, stratum, 'causal', 1, method)] for method in METHODS]
            n = cells[0]['n']
            if n == 0 or any(x['n'] != n for x in cells):
                raise ValueError('SCM-20 comparison uses inconsistent common pair denominator')
            rows.append({'seed': seed, 'stratum': stratum, 'n': n,
                         'counts': [x['correct'] for x in cells],
                         'mean_norms': [round(x['mean_patch_norm'], 3) for x in cells]})
            prim = next(x for x in payload['primary'] if
                        x['seed'] == seed and x['stratum'] == stratum and x['model'] == 'causal')
            if prim['n'] != n or prim['correct'] != cells[METHODS.index('native_vpd_write')]['correct']:
                raise ValueError('primary native mediation inconsistent with secondary receipt')
    clean = []
    for seed in SEEDS:
        for stratum in STRATA:
            def grab(model):
                return next(x['correct'] for x in payload['clean'] if
                            x['seed'] == seed and x['stratum'] == stratum and x['model'] == model)
            clean.append([seed, stratum, grab('causal'), grab('warm')])
    return rows, clean


def load_snapshot(path: Path = SNAPSHOT, *, summary_path: Path | None = None) -> dict:
    snap = json.loads(path.read_text(encoding='utf-8'))
    if snap.get('schema') != 'scm21.scm20_public_snapshot.v1' or snap.get('method_order') != list(METHODS):
        raise ValueError('snapshot schema or method order changed')
    if len(snap.get('rows', [])) != 6 or set((r['seed'], r['stratum']) for r in snap['rows']) != set(
            (s, t) for s in SEEDS for t in STRATA):
        raise ValueError('missing or duplicated seed/stratum')
    for row in snap['rows']:
        if row['n'] <= 0 or len(row['counts']) != len(METHODS) or len(row['mean_norms']) != len(METHODS):
            raise ValueError('malformed cell')
        if any(type(k) is not int or not 0 <= k <= row['n'] for k in row['counts']):
            raise ValueError('invalid success count')
        if any(not isinstance(v, (float, int)) or v < 0 for v in row['mean_norms']):
            raise ValueError('invalid norm')
        if row['counts'][METHODS.index('sham')] != 0:
            raise ValueError('negative control unexpectedly successful')
    if summary_path is not None:
        raw = summary_path.read_bytes()
        if git_blob_sha1(raw) != snap['source_git_blob_sha1']:
            raise ValueError('source Git blob hash differs from frozen public SCM-20 receipt')
        source = json.loads(raw)
        rows, clean = _source_pack(source)
        if rows != snap['rows'] or clean != snap['clean']:
            raise ValueError('derived snapshot differs from SCM-20 source')
    return snap


def pairwise_success_bounds(n: int, a: int, b: int) -> dict:
    """Sharp Fréchet bounds on (A succeeds, B fails) from marginal counts."""
    if not (0 <= a <= n and 0 <= b <= n):
        raise ValueError('invalid marginals')
    return {'minimum_a_only': max(0, a-b),
            'maximum_a_only': min(a, n-b),
            'minimum_b_only': max(0, b-a),
            'maximum_b_only': min(b, n-a),
            'success_difference': a-b}


def audit(snap: dict) -> dict:
    methods = snap['method_order']
    counts = defaultdict(int)
    num = 0
    per_cell = []
    for row in snap['rows']:
        i = lambda method: methods.index(method)
        n = row['n']; num += n
        for method, count in zip(methods, row['counts']):
            counts[method] += count
        alt = row['counts'][i('semantic_diff4')]
        native = row['counts'][i('native_write4')]
        learned = row['counts'][i('native_vpd_write')]
        compare = pairwise_success_bounds(n, alt, native)
        per_cell.append({
            'seed': row['seed'], 'stratum': row['stratum'], 'n': n,
            'semantic_diff4': alt, 'pca4': row['counts'][i('pca4')],
            'native_write4': native, 'native_vpd_write': learned,
            'full_cls': row['counts'][i('full_cls')],
            'guaranteed_semantic_only_vs_write_span': compare['minimum_a_only'],
            'mean_norm_ratio_semantic_to_write_span': round(row['mean_norms'][i('semantic_diff4')] / row['mean_norms'][i('native_write4')], 6)
        })
    total_alt = counts['semantic_diff4']
    total_native = counts['native_write4']
    per_stratum = {}
    for stratum in STRATA:
        part = [r for r in per_cell if r['stratum'] == stratum]
        n = sum(r['n'] for r in part)
        per_stratum[stratum] = {method: [sum(r['counts'][methods.index(method)] for r in snap['rows'] if r['stratum'] == stratum),n] for method in METHODS}
    report = {
        'schema': 'scm21.retrospective_capacity_audit.v1',
        'scientific_scope': 'retrospective SCM-20 counts, exact deterministic arithmetic, not new model execution',
        'source_git_blob_sha1': snap['source_git_blob_sha1'],
        'total_seed_task_comparisons': num,
        'method_successes': dict(counts),
        'method_success_fraction': {m: [counts[m], num] for m in METHODS},
        'minimum_semantic_diff_only_vs_native_write_span': sum(r['guaranteed_semantic_only_vs_write_span'] for r in per_cell),
        'semantic_vs_native_write_difference': total_alt-total_native,
        'semantic_vs_native_actual_difference': total_alt-counts['native_vpd_write'],
        'all_six_cells_semantic_exceeds_native_write': all(r['semantic_diff4'] > r['native_write4'] for r in per_cell),
        'semantic_norm_ratio_range': [min(r['mean_norm_ratio_semantic_to_write_span'] for r in per_cell),max(r['mean_norm_ratio_semantic_to_write_span'] for r in per_cell)],
        'per_cell': per_cell,
        'per_stratum': per_stratum,
        'clean_accuracy_consequence': [{'seed':s,'stratum':t,'causal':c,'warm':w,'delta':c-w} for s,t,c,w in snap['clean']],
        'limits': [
            'Method-level totals contain no pairwise cross-method outcome identities; only sharp overlap bounds can be claimed.',
            'Success counts across seeds reuse test cases; aggregate counts are not independent statistical trials.',
            'Different PCA and native-write patches are not per-case norm-matched; mean-norm ratios cannot remove this confound.',
            'The readout-visible method has higher effective rank and is not a rank-four matched comparator.',
            'This is existing trained-model behavior from SCM-20 receipts; SCM-21 ran no trained checkpoint or new evaluation.'
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--source', type=Path, help='Full SCM-20 public summary.json from checkout; validate Git blob and snapshot')
    parser.add_argument('--out', type=Path, default=HERE / 'results' / 'scm21_cpu_audit.json')
    parser.add_argument('--check', action='store_true', help='Verify committed report without modification')
    args = parser.parse_args()
    data = audit(load_snapshot(args.snapshot, summary_path=args.source))
    rendered = json.dumps(data, sort_keys=True, indent=2) + '\n'
    if args.check:
        if args.out.read_text(encoding='utf-8') != rendered:
            raise SystemExit('SCM-21 CPU audit receipt differs from frozen output')
        print('SCM-21 public receipt integrity PASS')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding='utf-8')
        print('SCM-21 CPU audit written', args.out)
    print('rank-four semantic difference-PCA', data['method_successes']['semantic_diff4'], '/', data['total_seed_task_comparisons'])
    print('rank-four native write-span', data['method_successes']['native_write4'], '/', data['total_seed_task_comparisons'])
    print('native learned VPD write', data['method_successes']['native_vpd_write'], '/', data['total_seed_task_comparisons'])
    print('minimum semantic-only victories over native write-span', data['minimum_semantic_diff_only_vs_native_write_span'])


if __name__ == '__main__':
    main()
