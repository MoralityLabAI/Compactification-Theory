# DCF-15: LDT mesh routing

An experimental inference mesh in the Lattice Deduction Transformer sense: explicit, checkable unit state plus a router that trades predicted decision regret, computation, discarded correlation and batch fragmentation, `J(a) = E[R̂(a)] + λC(a) + μ KL̂(a) + γF(a)`.

The world has 16 latent bits in four units and streams one noisy sensor and one parity query per step. Cross-unit dependencies change every slow epoch. Routing runs on three timescales:
- fast: absorb evidence and answer by local / cluster-BP / exact / abstain;
- slow: merge or split units by estimated discarded correlation;
- global: exact inference over the relevant component on request.

The routers compared are fixed (local, BP, exact), random, dev-tuned symbolic thresholds, symbolic J, a learned MLP regret model, and a privileged oracle.

- [`mesh15.py`](mesh15.py): world, `Mesh` with `expose()`, operators, routers, cost model, stages `calibrate | dev | fit | test | report`.
- [`PROTOCOL.json`](PROTOCOL.json): conditions, seeds, λ/μ/γ and gates.
- Receipts:
  - `cost_model15.json`: timed batched kernels and fit.
  - `receipt15_dev.json`: development labels.
  - `receipt15_test.json.gz`: every held-out query, gzip (`report` reads it directly).
  - `results15.json`.
- Frozen models: `model15.pkl`, `model15_summary.json`.
- Tests: `tests/test_mesh15.py`, with independent checks against `../dcf14_width_routing/fg.py`.
- Architecture port: [HybridTRMLDT `research_gym/ldt_mesh`](https://github.com/MoralityLabAI/HybridTRMLDT/tree/master/research_gym/ldt_mesh), a dependency-free typed `MeshUnit` with the seven exposed fields, the operators, the J router and 11 tests.

Results are in [RESULTS.md](RESULTS.md) and theory in [THEORY.md](THEORY.md). Headline: the learned router fails promotion 0/6. Fixed BP has the best cost-charged objective; slow-timescale unit merging cuts discarded correlation 36%; batching amortises the exact kernel 3.7× but adaptive routers fragment batches.

## Reproduce

Python 3.12, NumPy 2.3.5, scikit-learn 1.7.2.

    python -m pytest tests
    python mesh15.py calibrate   # writes cost_model15.json (host-specific timings)
    python mesh15.py dev
    python mesh15.py fit
    sha256sum -c PRE_TEST_SHA256
    python mesh15.py test        # about 75 min on one core
    python mesh15.py report
