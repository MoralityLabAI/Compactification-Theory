# DCF-14: computational width as a routing signal

Tests whether estimated induced width predicts the cost of exact reasoning better than hidden-state dimension or connected-component size, and whether a learned algorithm selector beats strong symbolic routing at an explicit compute budget. CPU only, no paid compute.

- [`fg.py`](fg.py): binary factor graphs, elimination orders (min-degree, min-fill, weighted min-fill), symbolic cost, and exact inference along four independent paths: einsum VE, broadcast VE, log-space enumeration, and batched cutset conditioning. GF(2) elimination for noise-free parity systems. Approximations: loopy BP, mini-bucket, likelihood weighting. Shared by DCF-13D, 15 and 16.
- [`gen.py`](gen.py): ten families (chain, tree, star, grid, sparse, dense, hypergraph, planted k-tree, noise-free XOR, weakly coupled dense). Noise, factor kinds, query scope, model misspecification and required confidence vary independently of structure.
- [`width14.py`](width14.py): stages `dev`, `fit`, `test`, `report`. [`PROTOCOL.json`](PROTOCOL.json) holds the seeds, budgets and gates.
- Receipts: `receipt14_dev.json`, `receipt14_test.json` (per instance: features, every operator's posterior, ops and seconds, reference and verification paths). Frozen selectors: `selector14.pkl`, `selector14_summary.json`. Summary: `results14.json`.

Results are in [RESULTS.md](RESULTS.md) and theory in [THEORY.md](THEORY.md). Headline: H1 passes, but width misprices GF(2)-solvable, weakly coupled and decision-trivial problems. The learned selectors fail the promotion gate against exact-else-mini-bucket.

## Reproduce

Python 3.12, NumPy 2.3.5, scikit-learn 1.7.2, pytest. From this directory:

    python -m pytest tests            # 129 verification tests
    sha256sum -c PRE_RUN_SHA256       # code as of the dev run; fg.py and width14.py changed afterwards (see below)
    python width14.py dev
    python width14.py fit             # or use the committed selector14.pkl
    sha256sum -c PRE_TEST_SHA256
    python width14.py test
    python width14.py report

Changes after the hash locks, each committed with its reason:
- `fg.py` gained einsum operand chunking after the first held-out launch crashed on star hubs with more than 30 factors in a bucket. The crash happened before any receipt was written; the log is kept as `test14_crash.log`. The new path never ran on development instances.
- The `report` stage was re-run after fixing a key collision that hid the H1 row for feature `n`, and after adding a descriptive λ-objective.

Neither change touches receipts or frozen selectors.
