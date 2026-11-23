"""Deterministic synthetic diagnostic; prints a JSON receipt to stdout."""
import json
import platform
import numpy as np
from surgery import plan_surgery, audit, bf16_round


def run():
    rng = np.random.default_rng(20261104)
    w = bf16_round(rng.normal(scale=0.05, size=(96, 160)))
    s = rng.normal(size=(96, 4))
    d = rng.normal(size=(96, 8)) + s @ rng.normal(size=(4, 8))
    h = rng.normal(size=(160, 8))
    rows = []
    for rank in (1, 2, 4, 8):
        for strength in (0.25, 0.5, 1.0):
            p = plan_surgery(w, d, s, h, rank=rank, strength=strength)
            exact = p.merge(w)
            rows.append({"requested_rank": rank, "strength": strength,
                         "float64": audit(w, exact, p),
                         "bf16_storage": audit(w, bf16_round(exact), p)})
    identity = np.eye(3)
    sink = identity[:, :1]
    direction = np.array([[-0.77], [np.sqrt(1-0.77**2)], [0]])
    naive = plan_surgery(identity, direction)
    protected = plan_surgery(identity, direction, sink)
    return {"experiment": "PSS-01", "evidence": "synthetic CPU only",
            "seed": 20261104, "python": platform.python_version(), "numpy": np.__version__,
            "matrix_shape": list(w.shape), "baseline_storage": "BF16",
            "sink_example": {"naive_protected_output_change": float(np.linalg.norm(sink.T @ naive.delta())),
                             "protected_output_change": float(np.linalg.norm(sink.T @ protected.delta())),
                             "retained_failure_energy": protected.retained_failure_energy},
            "rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
