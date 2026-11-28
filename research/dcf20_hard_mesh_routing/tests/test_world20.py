"""World generator and escalation policy checks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import world20 as W  # noqa: E402

M = W.M


def test_world_axes_change_the_generator_as_described():
    e = W.episode(5, 1.0, 'tight', 1.0, True, True)
    cross = [f for f in e['facs'] if len({b // 4 for b in range(16) if (f[0] >> b) & 1}) == 2]
    assert cross and all(f[1] == 0 for f in cross)          # xor_frac 1: every cross sensor is XOR
    assert all(f[3] <= 0.03 for f in cross)                  # tight noise band
    aligned = sum(len({b // 4 for b in range(16) if (q >> b) & 1}) == 2 for q in e['queries'])
    assert aligned >= M.T // 3                               # most queries straddle a cross sensor


def test_escalation_matches_direct_execution():
    tr = W.trace([95000 + i for i in range(4)], 0.6, dict(noise='tight', xor_frac=1.0, triangle=True, aligned=True), 4)
    rows = W.escalate_rows(tr, 0.15)
    i = 0
    for b in tr:
        for s in b['steps']:
            for q in s['queries']:
                k = W.R19B.near(q['bp']['p'], 0.15)
                assert rows[i]['op'] == ('exact' if k else 'bp')
                assert rows[i]['eu'] == (q['exact']['eu'] if k else q['bp']['eu'])
                i += 1
    qo = W.R19.summary(W.escalate_rows(tr, oracle=True))['eu']
    assert qo >= W.R19.summary(W.evaluate(tr, 'fixed_exact'))['eu'] - 1e-12


def test_trace_operators_are_independent_of_policy():
    w = dict(noise='low', xor_frac=1.0, triangle=True, aligned=True)
    a = W.trace([96000, 96001], 0.5, w, 2)
    b = W.trace([96000, 96001], 0.5, w, 2)
    assert [q['exact']['p'] for s in a[0]['steps'] for q in s['queries']] == [q['exact']['p'] for s in b[0]['steps'] for q in s['queries']]
