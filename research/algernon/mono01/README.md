# MONO-01

- `monoshot.py`: closed-form edit, conditional leverage, greedy/beam/exhaustive support selection, bf16 rounding, eigenvalue shift, minimax mixture.
- `tests/test_monoshot.py`: run with `python -m unittest discover -s research/algernon/mono01/tests` (11 tests, NumPy and optional torch).
- `policy_surgery_audit.py`, `phase_c.py`, `phase_d.py`: run inside a checkout of latentpaths/policy-surgery at 46f95e6 with `uv run --frozen --extra weights python <script> <out>` (see `receipts/POLICY_SURGERY_REPRODUCTION.md` for the Windows fixes). `phase_d.py` also takes the path of `phase_c_results.json`.
- `NOTE.md`: results, limits and the next experiment. `receipts/`: raw result JSON and reproduction notes.
