# DCF-19: regime-level routing

Tests the DCF handoff's second recommendation on the DCF-15 mesh: route once per batch per slow epoch instead of once per query. One batch statistic chooses between cluster BP and exact-over-relevant-component inference for the next 8 steps. The statistic is the mean pending discarded-correlation estimate S = Σ(ln 2 − H(e)) over unresolved cross-unit sensors. Each step therefore launches a single kernel, and routing costs one pass over pending factors per episode per epoch.

- [`regime19.py`](regime19.py): traces (both operators executed on every query), policy evaluation, stages `dev | test | sweep | report`.
- [`PROTOCOL.json`](PROTOCOL.json): τ grid, seeds and gates.
- `selection19.json`: τ = 2 with the dev grid; committed before held-out data.
- Traces: `trace19_dev.json.gz`, `trace19_test.json.gz`, `trace19_sweep.json.gz`. Results: `results19.json`.
- [`tests/test_regime19.py`](tests/test_regime19.py): trace evaluation equals direct online execution.

Results are in [RESULTS.md](RESULTS.md) and theory in [THEORY.md](THEORY.md):
- R1 fails (1 win, 5 losses vs fixed BP); R2 passes (5/6 within .001 of the better fixed operator).
- Routing cost drops to 0.005 ms per query.
- Regime routing has the best mean objective of any non-oracle policy on this mesh, but its gain is concentrated in one condition.
- It also corrects DCF-15's tuning objective, which charged batched launch overhead per query.

## DCF-19B

`regime19b.py`, `PROTOCOL19B.json`, `selection19b.json`, traces `trace19b_{dev,test}.json.gz`, `results19b.json`, `exploratory19b.json` (post-gate analysis) and `tests/test_regime19b.py`. Decision-margin signals did not beat the DCF-19 rule on development data. On fresh seeds the rule replicates: best on the mean, R1 fails 1/5. Exact changes only 1.6% of BP decisions on this mesh, which bounds what any router can win here. See the DCF-19B section of [RESULTS.md](RESULTS.md).

    sha256sum -c PRE_RUN_19B_SHA256
    python regime19b.py dev
    sha256sum -c PRE_TEST_19B_SHA256
    python regime19b.py test
    python regime19b.py report

## Reproduce

Python 3.12, NumPy 2.3.5; needs `../dcf15_ldt_mesh_routing` unchanged (hash-locked in `PRE_RUN_SHA256`).

    python -m pytest tests
    sha256sum -c PRE_RUN_SHA256
    python regime19.py dev       # about 1 min; writes selection19.json
    sha256sum -c PRE_TEST_SHA256
    python regime19.py test      # about 3 min
    python regime19.py sweep     # about 8 min
    python regime19.py report
