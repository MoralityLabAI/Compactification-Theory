# DCF-20: routing in a harder mesh world

DCF-19B found that on the DCF-15 mesh exact inference changes BP's decision on only 1.6% of queries, so routing had almost nothing to win. DCF-20 keeps the mesh, operators and calibrated cost model, and changes the world along three axes that stress BP:
- XOR-heavy cross-unit sensors;
- unit-level loops (triangles);
- queries aligned with recent cross-unit sensors.

Steps:
1. A world is selected on dedicated seeds.
2. Routing policies are tuned on development seeds and tested on fresh seeds.
3. An exploratory compute-price sweep in the hardest world.
4. A gated fresh-seed test of the hypothesis that sweep produced (DCF-20B).

- [`world20.py`](world20.py): world generator, traces, policies (fixed, regime, probe, escalate, oracles); stages `world | dev | test | report`. [`PROTOCOL.json`](PROTOCOL.json).
- `world20.json`: the 16-world scan and selection. `selection20.json`: development choice. `trace20_{dev,test}.json.gz`, `results20.json`.
- [`explore20.py`](explore20.py), `explore20.json`, `trace20_explore.json.gz`: exploratory, not gated.
- [`gated20b.py`](gated20b.py), [`PROTOCOL20B.json`](PROTOCOL20B.json), `PRE_RUN_20B_SHA256`, `trace20b.json.gz`, `results20b.json`: DCF-20B.
- [`tests/test_world20.py`](tests/test_world20.py).

Headline:
- Gated DCF-20 fails W1, R1 and R2: selecting by epoch headroom picked an easy world.
- In the hardest world 95% of decision changes occur where BP abstains.
- **DCF-20B escalate-on-abstention passes its gate 6/6 at both declared compute prices (λ = 0.1, 0.3)** and loses at the DCF-15 price (λ = 0.01).

Details are in [RESULTS.md](RESULTS.md) and [THEORY.md](THEORY.md).

## Reproduce

Python 3.12, NumPy 2.3.5. Needs `../dcf15_ldt_mesh_routing` and `../dcf19_regime_routing` unchanged.

    python -m pytest tests
    python world20.py world          # about 13 min
    python world20.py dev && python world20.py test && python world20.py report
    python explore20.py              # exploratory
    sha256sum -c PRE_RUN_20B_SHA256
    python gated20b.py run && python gated20b.py report

`report` stages re-measure routing wall time, so `route_ms` and `objective_total` move in the fourth decimal between runs. Policy choices and all other fields are deterministic.
