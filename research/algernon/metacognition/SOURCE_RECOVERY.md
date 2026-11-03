# H-M experiment source recovery

The historical ChatGPT sandbox archive has 11 Python files (70,774 source bytes) under `experiments/`, including the H-M0 baseline/pruner, H-M0 terminal-utility model, cost-aware stopping comparisons, H-M1 revision environment, toy contracts, and five test modules.

The corresponding full archive remains in ChatGPT's sandbox. This repository currently holds the handoff and upload instructions, not the full source or all results.

**Reproduction caveat:** H-M2 was reported in conversation without a retained runnable source or evaluation record. Recreate and register it separately.

The GitHub connector supports text file writes. A reliable transfer requires importing the files from a host with access to the original archive. The SHA-256 of the full archive is `88d8e23581f0445aca9ac7b7b9a9c7f8d7df34f00bbc5126fced84511053f7b4`.

## Inventory

- `benchmark_terminal_runtime.py`
- `h_m0_fair_baselines.py`
- `h_m0_pruner.py`
- `h_m0_terminal.py`
- `h_m1_revision.py`
- `toy_contracts.py`
- `tests/test_h_m0_fair_baselines.py`
- `tests/test_h_m0_pruner.py`
- `tests/test_h_m0_terminal.py`
- `tests/test_h_m1_revision.py`
- `tests/test_toy_contracts.py`
