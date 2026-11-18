# CGC-02 engineering amendment 02: preserve CGC-01 NumPy-only source checks

**November 18, 2026.** CGC-01's source-only [original 18-test job](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/.github/workflows/cgc01-source-math.yml) installed only NumPy and used broad recursive `test_*.py` discovery. Adding CGC-02's distinct PyTorch package under `research/causal_gauge_compactification/cgc02/` caused that job to import `cgc02/test_experiment.py` and fail with `ModuleNotFoundError: torch` (example [run #44239637130](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44239637130)). The prior exact rational CGC-01 source/math test definitions and verified receipt were unchanged.

This CI-only amendment changes the CGC-01 discovery glob from `test_*.py` to `test_finite_audit.py`, retaining **precisely the 18 original CGC-01 tests** in the lightweight NumPy job. All CGC-02 PyTorch source, geometric and native BPD tests execute in the separately source-locked `cgc02-engineering.yml` job. Both workflows remain required for a clean PR.

This amendment does not change either mathematical theorem, synthetic teacher physics, test-root roster, rank selection, science criteria or earlier CGC-01 evidence. It was made **before training CGC-02's four fresh roots**.
