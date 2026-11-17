# SMC-27E: Correct existing audit module import before test data

**18 November 2026, source-only engineering note.** The first SMC-27 CI run #44136178234 halted at Python unit-test import discovery, before downloading teacher checkpoints or running any SMC-27 old calibration/native effects or fresh test data.

The predictor and independent auditor imported the historical prior-Sudoku cohort manifest from module smc26_audit. Its actual source repository filename and importable Python module are audit_smc26. Both files now use the exact source module name. This is a source-identifier repair and does not change the 640 prior registered source cohorts, their seeds, the SMC-27 fresh Sudoku seeds, original SMC-26 teacher/SAE hashes, calibration-selected threshold grid, action utility, coverage or familywise statistical gates.

The original failed CI remains visible; the first scientific run with model effects is a separate rerun on the same fixed registered protocol. No previously sealed test outcome was accessed during repair.
