# LSV-05 frozen test protocol

The frozen checkpoint and numerical bounds were selected before running the external pretrained weights in CI.

- Treat `source_lock.json` as immutable during each evaluation. Record both SHA1 Git blob and SHA256 fingerprints. A mismatch prevents further work.
- Evaluate 64 puzzles from seeds `202611200000..202611200063`, with 30 holes per puzzle and the upstream generator. Do not select a best seed from the test roster.
- Freeze source, architecture, generator, precision and damage controls. Report raw full-grid, all-cell, missing-cell and valid-Sudoku metrics, including zero successes.
- Compare four-stage clean run, one-step first-stage ablation, rounded BF16 storage, one stale full-state read, and hidden-damage amplitudes `0.02`, `0.10`, `0.50` across two independent seeded noise tensors.
- The perturbation is applied after the first recursion. Models have three remaining passes for recovery; record the full hidden-state difference trajectory. Damage seeds are independent of the target puzzles.
- Exact mathematical claims must depend on globally valid derivatives/bounds and frozen weights, and be clearly separated from observation-derived metrics. An oversized bound is a failed practical certification gate.
- CPU runtime is recorded only as a contextual measurement. It depends heavily on runner hardware and is not compared as a controlled latency benchmark.
- Report any source-hash validation failures, numerical errors, invalid Sudoku outputs, run failure, or missing external weights. Checkpoint source remains upstream and is not copied into this repository.
