# MONO-01

- `monoshot.py`: closed-form edit, conditional leverage, greedy/beam/exhaustive support selection, bf16 rounding, eigenvalue shift, minimax mixture.
- `tests/test_monoshot.py`: run with `python -m unittest discover -s research/algernon/mono01/tests` (11 tests, NumPy and optional torch).
- `policy_surgery_audit.py`, `phase_c.py`, `phase_d.py`: run inside a checkout of latentpaths/policy-surgery at 46f95e6 with `uv run --frozen --extra weights python <script> <out>` (see `receipts/POLICY_SURGERY_REPRODUCTION.md` for the Windows fixes). `phase_d.py` also takes the path of `phase_c_results.json`.
- `NOTE.md`: results, limits and the next experiment. `receipts/`: raw result JSON and reproduction notes.

## MONO-HODGE-01 (2026-11-04)

- [Mathematics, archive provenance and execution gates](MONO_HODGE01.md).
- `hodge01.py`: strict observational evidence-order diagnostics; Woodbury generalized order-sensitive modes; family-grouped predictive probe; synthetic rank-one Gauss-Newton commutator editor.
- `hodge01_analyze.py`: analyze explicitly normalized HF export files locally. Public HODGE/M-VOI datasets were not accessible in this execution runtime; **no real 27B results are claimed**.
- `tests/test_hodge01.py`: 15 deterministic CPU unit tests with affine-operator and numerical controls.
- `receipts/mono_hodge01_cpu_toy_20261104.json`: synthetic demonstration, kept separate from 27B evidence.

Run: `python -m unittest discover -s research/algernon/mono01/tests -p test_hodge01.py -v`.
