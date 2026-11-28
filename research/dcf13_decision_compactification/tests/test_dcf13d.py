"""Independent checks for audit13d.py on development seeds (never the fresh seeds)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit13d as A  # noqa: E402


def test_symbolic_work_and_cost_hybrid_exact():
    for k, sc in [(8, 'bridge'), (12, 'bridge'), (12, 'rewired'), (16, 'ambiguous')]:
        q, mm, kk, yy, ee, truth = A.fixture(k, 55 + k, 24, sc)
        ref = A.predict(k, q, mm, kk, yy, ee, 0, k, .5)[0]
        for i in range(len(q)):
            assert A.ve_symbolic(k, q[i], mm[i], k)[0] == A.eliminate(k, q[i], mm[i], kk[i], yy[i], ee[i], k, .5, 0)[2]
        for alpha in (0.01, 1.0, 100.0):
            p, route, work = A.hybrid_cost(k, q, mm, kk, yy, ee, alpha)
            assert np.max(np.abs(p - ref)) < 1e-10


def test_polynomial_masses_match_brute_force():
    q, mm, kk, yy, ee, truth, w = A.fixture_w(12, 77, 6)
    for i in range(6):
        (n0, n1), _, _ = A.masses_by_component(12, q[i], mm[i], kk[i], yy[i], ee[i])
        for mix in (0., .2, .9, 1.):
            z0, z1 = A.brute_mass(12, q[i], mm[i], kk[i], yy[i], ee[i], mix)
            assert abs(A.bern_eval(n1, mix) / (A.bern_eval(n0, mix) + A.bern_eval(n1, mix)) - z1 / (z0 + z1)) < 1e-10


def test_certificate_is_sound_and_detects_negatives():
    rng = np.random.default_rng(0)
    for _ in range(300):
        c = rng.normal(size=int(rng.integers(1, 9))) + rng.uniform(0, 2)
        r = A.certify_nonneg(c, 6)
        vals = [A.bern_eval(c, w) for w in np.linspace(0, 1, 401)]
        if r is True:
            assert min(vals) >= -1e-12
        if r is False:
            assert min(vals) < 0


def test_bernstein_integral():
    c = np.array([0.3, 1.2, 0.0, 2.5])
    xs = np.linspace(0, 1, 200001)
    num = np.trapezoid([A.bern_eval(c, x) for x in xs[::100]], xs[::100])
    assert abs(num - A.bern_integral(c)) < 1e-6
