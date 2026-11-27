"""Checks for mp23: BP beliefs, batching, and the pure variant's isolation from BP."""
import copy
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mp23 as M  # noqa: E402


def test_bp_beliefs_exact_on_a_tree():
    from gen import make_problem
    prob = make_problem('tree', 9, 3, query_size=1)
    comp, variables = M.fg.query_component(prob)
    bel = M.bp_beliefs(comp, variables, 200, 0.0)
    for v in variables:
        num = den = 0.0
        for x in itertools.product((0, 1), repeat=prob.n):
            w = np.prod([t[tuple(x[u] for u in s)] for s, t in zip(prob.scopes, prob.tables)])
            den += w
            num += w * x[v]
        assert abs(bel[v] - num / den) < 1e-7


def test_collate_offsets_and_forward_shapes():
    items = [M.featurize(e) for e in M.load_split('val')[:5]]
    g = M.collate(items)
    assert g['edge_v'].max() < g['vfeat'].shape[0] and g['edge_f'].max() < g['ffeat'].shape[0]
    assert int(g['q_graph'].max()) == len(items) - 1
    for variant in ('mp_bp', 'mp_pure'):
        out = M.build(variant, 0)(g)
        assert out.shape == (len(items),)


def test_pure_variant_ignores_bp_inputs():
    import torch
    items = [M.featurize(e) for e in M.load_split('val')[:4]]
    alt = copy.deepcopy(items)
    for it in alt:
        it['p_bp'] = 1 - it['p_bp']
        for v in it['vfeat']:
            v[1] = -v[1]
    net = M.build('mp_pure', 0)
    with torch.no_grad():
        assert torch.allclose(net(M.collate(items)), net(M.collate(alt)))
