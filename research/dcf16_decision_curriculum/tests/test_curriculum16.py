"""Independent checks on the committed DCF-16 data files."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import fg  # noqa: E402
import curriculum16 as C  # noqa: E402


def test_manifest_hashes_match_files():
    man = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, meta in man['files'].items():
        raw = gzip.open(ROOT / 'data' / name, 'rb').read()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256_uncompressed']
        assert raw.count(b'\n') == meta['examples']


def test_splits_disjoint_and_heldout_families_unseen():
    ids, fams = {}, {}
    for split in ('train', 'val', 'heldout'):
        exs = C.load(split)
        ids[split] = {e['id'] for e in exs}
        fams[split] = {(e['family'], e['n']) for e in exs}
        assert all(e['split'] == split for e in exs)
    assert not (ids['train'] & ids['heldout']) and not (ids['val'] & ids['heldout']) and not (ids['train'] & ids['val'])
    assert not (fams['heldout'] & (fams['train'] | fams['val']))


def test_posteriors_regrets_and_targets_rederive():
    for split in ('val', 'heldout'):
        for e in C.load(split)[:40]:
            prob = C.to_problem(e)
            p = fg.variable_elimination(prob)['p']
            if not e['conditions']['model_misspecified']:
                assert abs(p - e['verification']['reference_p']) < 1e-8
            pen = -e['decision_rule']['reward']['wrong']
            ref = e['verification']['reference_p']
            assert abs(e['regret'] - (fg.oracle_utility(ref, pen) - fg.expected_utility(e['final']['action'], ref, pen))) < 1e-9
            comp, variables = fg.query_component(prob)
            assert e['targets']['relevant_variables'] == variables
            assert e['targets']['abstain'] == (e['final']['action'] < 0)
            within = [o for o in e['operators'] if o['within_budget']]
            if e['selected']['option'] != 'abstain':
                assert e['selected']['ops'] <= 2 ** e['budget_log2_ops']
            assert all(o['ops'] <= 2 ** e['budget_log2_ops'] for o in within)
