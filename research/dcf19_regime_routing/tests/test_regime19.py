"""Trace evaluation equals direct online execution of the regime policy."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import regime19 as R  # noqa: E402

M = R.M


def online(seeds, rho, noise, B, tau):
    """Run the regime policy directly, executing only the chosen operator."""
    out = []
    for start in range(0, len(seeds), B):
        batch = seeds[start:start + B]
        eps = [M.episode(s, rho, noise) for s in batch]
        meshes = [M.Mesh() for _ in batch]
        op = 'bp'
        for t in range(M.T):
            for e, mesh in zip(eps, meshes):
                mesh.absorb(e['facs'][t])
                if t % M.M == M.M - 1:
                    M.slow_merge(mesh, 'kl', M.P['merge_tau'])
            if t % M.M == 0 and t > 0:
                op = 'exact' if np.mean([R.pending_kl(m) for m in meshes]) >= tau else 'bp'
            for e, mesh in zip(eps, meshes):
                q = e['queries'][t]
                p_ref = M.parity_p(mesh.full, M.STATES, q)
                p, w = mesh.bp(q) if op == 'bp' else mesh.exact(q)
                a = R.decide(p)
                out.append((op, 0.0 if a < 0 else (5 * p_ref - 4 if a == 1 else 1 - 5 * p_ref), int(w)))
    return out


def test_trace_matches_online_execution():
    seeds = [91000 + i for i in range(6)]
    for rho, noise, tau in [(0.6, 'low', 1.0), (0.2, 'high', 0.5), (0.4, 'low', 1e9), (0.4, 'low', 0.0)]:
        tr = R.trace(seeds, rho, noise, 3)
        rows = R.evaluate(tr, 'regime', tau)
        ref = online(seeds, rho, noise, 3, tau)
        assert len(rows) == len(ref)
        for r, (op, eu, w) in zip(rows, ref):
            assert r['op'] == op and abs(r['eu'] - eu) < 1e-12 and r['work'] == w


def test_epoch_oracle_dominates_fixed_on_kernel_objective():
    tr = R.trace([92000 + i for i in range(4)], 0.6, 'low', 4)
    o = R.summary(R.evaluate(tr, 'epoch_oracle'))['objective_kernel']
    for p in ('fixed_bp', 'fixed_exact'):
        assert o >= R.summary(R.evaluate(tr, p))['objective_kernel'] - 1e-12
