"""Trace evaluation of the 19B signals equals direct online execution."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import regime19b as B  # noqa: E402

M = B.M


def online(seeds, rho, noise, bs, kind, delta, phi):
    out = []
    for start in range(0, len(seeds), bs):
        batch = seeds[start:start + bs]
        eps = [M.episode(s, rho, noise) for s in batch]
        meshes = [M.Mesh() for _ in batch]
        op, prev = 'bp', []
        for t in range(M.T):
            for e, mesh in zip(eps, meshes):
                mesh.absorb(e['facs'][t])
                if t % M.M == M.M - 1:
                    M.slow_merge(mesh, 'kl', M.P['merge_tau'])
            if t % M.M == 0:
                if kind == 'probe':
                    ps = [mesh.bp(e['queries'][t])[0] for e, mesh in zip(eps, meshes)]
                    op = 'exact' if np.mean([B.near(p, delta) for p in ps]) >= phi else 'bp'
                else:
                    op = 'bp' if t == 0 else ('exact' if np.mean([B.near(p, delta) for p in prev]) >= phi else 'bp')
                    prev = []
            for e, mesh in zip(eps, meshes):
                q = e['queries'][t]
                p_ref = M.parity_p(mesh.full, M.STATES, q)
                p, w = mesh.bp(q) if op == 'bp' else mesh.exact(q)
                prev.append(p)
                a = B.R19.decide(p)
                out.append((op, 0.0 if a < 0 else (5 * p_ref - 4 if a == 1 else 1 - 5 * p_ref), int(w)))
    return out


def test_probe_and_lagged_match_online():
    seeds = [93000 + i for i in range(6)]
    tr = B.trace(seeds, 0.6, 'low', 3)
    for kind in ('probe', 'lagged'):
        for delta, phi in ((0.1, 0.2), (0.3, 0.05), (0.05, 0.5)):
            rows = B.evaluate(tr, kind, dict(delta=delta, phi=phi))
            ref = online(seeds, 0.6, 'low', 3, kind, delta, phi)
            assert len(rows) == len(ref)
            for r, (op, eu, w) in zip(rows, ref):
                assert r['op'] == op and abs(r['eu'] - eu) < 1e-12 and r['work'] == w


def test_probe_charges_extra_bp_kernel_only_when_switching():
    tr = B.trace([94000 + i for i in range(3)], 0.6, 'low', 3)
    rows = B.evaluate(tr, 'probe', dict(delta=1.0, phi=0.0))   # always exact
    first = [r for r in rows[:3]]
    assert all(r['op'] == 'exact' and r['launches'] == 2 for r in first)
    rows = B.evaluate(tr, 'probe', dict(delta=0.0, phi=1.0))   # never exact
    assert all(r['op'] == 'bp' and r['launches'] == 1 for r in rows)
