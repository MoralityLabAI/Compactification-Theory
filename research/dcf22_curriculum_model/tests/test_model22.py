"""Encoding checks: shapes, and no operator posterior reaches the inputs."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model22 as M  # noqa: E402


def test_encoding_shapes_and_order():
    e = M.load_split('val')[0]
    it = M.encode(e)
    assert len(it['glob']) == M.GLOB_DIM and all(len(t) == M.TOK_DIM for t in it['tokens'])
    assert len(it['tokens']) <= M.P['max_tokens'] and it['mask'][-1] is True
    d = [t[13] for t in it['tokens']]
    assert d == sorted(d, reverse=True)          # far-to-near


def test_inputs_do_not_depend_on_operator_posteriors():
    for e in M.load_split('val')[:20]:
        a = M.encode(e)
        f = copy.deepcopy(e)
        for o in f['operators']:
            if o['p'] is not None:
                o['p'] = 1.0 - o['p']
            o['action'] = -o['action'] if o['action'] in (0, 1) else o['action']
        b = M.encode(f)
        assert a['tokens'] == b['tokens'] and a['glob'] == b['glob'] and a['mask'] == b['mask']


def test_macs_grow_with_tokens():
    assert M.macs('gru', 24) > M.macs('gru', 1) > 0
