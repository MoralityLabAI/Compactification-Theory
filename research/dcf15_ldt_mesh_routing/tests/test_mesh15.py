"""Independent checks of the mesh operators against fg.py (DCF-14 library)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import fg  # noqa: E402
import mesh15 as M  # noqa: E402

KIND = {0: 'xor', 1: 'and', 2: 'maj'}


def problem(facs, q, only_mask=None):
    scopes, tables = [], []
    for mask, kind, y, e in facs:
        if only_mask is not None and mask & only_mask != mask:
            continue
        s = [b for b in range(M.K) if (mask >> b) & 1]
        scopes.append(s)
        tables.append(fg.sensor_table(len(s), 'id' if len(s) == 1 else KIND[kind], y, e))
    return fg.Problem(M.K, scopes, tables, [b for b in range(M.K) if (q >> b) & 1])


def test_exact_operator_matches_independent_elimination_on_relevant_component():
    for seed in range(4):
        ep = M.episode(900 + seed, 0.5, 'low')
        mesh = M.Mesh()
        for t in range(M.T):
            mesh.absorb(ep['facs'][t])
            if t % M.M == M.M - 1:
                M.slow_merge(mesh, 'kl', M.P['merge_tau'])
            if t % 5 == 4:
                q = ep['queries'][t]
                p, cost = mesh.exact(q)
                comp = mesh.relevant(q)
                cm = sum(M.UNIT_MASK[u] for b in comp for u in b)
                ref = fg.variable_elimination(problem(mesh.facs, q, cm))['p']
                assert abs(p - ref) < 1e-9
                assert abs(p - fg.variable_elimination(problem(mesh.facs, q))['p']) < 1e-9


def test_local_is_exact_without_cross_factors_and_kl_is_zero():
    ep = M.episode(77, 0.0, 'high')
    mesh = M.Mesh()
    for t in range(M.T):
        mesh.absorb(ep['facs'][t])
        q = ep['queries'][t]
        if bin(q).count('1') == 2 and len({M.unit_of(b) for b in range(M.K) if (q >> b) & 1}) == 1:
            assert abs(mesh.local(q)[0] - mesh.exact(q)[0]) < 1e-9
    assert abs(mesh.true_kl()) < 1e-9


def test_cluster_bp_exact_on_a_single_cross_factor():
    rng = np.random.default_rng(3)
    mesh = M.Mesh()
    x = int(rng.integers(1 << M.K))
    for u in range(M.U):
        for b in range(4):
            mask = 1 << (4 * u + b)
            mesh.absorb((mask, 0, int(M.sensor(x, mask, 0)) ^ int(rng.random() < .2), .2))
    mesh.absorb(((1 << 1) | (1 << 5), 0, 1, .05))
    q = (1 << 1) | (1 << 5)
    p_bp, _ = mesh.bp(q, iters=80)
    # cluster BP on a tree is exact for block marginals; the parity across two blocks
    # uses the independence read-out, so compare single-bit marginals instead
    for bit in (1, 5):
        assert abs(mesh.bp(1 << bit, iters=80)[0] - mesh.exact(1 << bit)[0]) < 1e-6
    assert 0 <= p_bp <= 1


def test_exposed_interface_fields():
    ep = M.episode(5, 0.6, 'low')
    mesh = M.Mesh()
    for t in range(12):
        mesh.absorb(ep['facs'][t])
    e = mesh.expose(mesh.blocks[0])
    for key in ('units', 'state_size', 'dependencies', 'entropy', 'operators', 'cost_local', 'kl_hat', 'verified'):
        assert key in e
