# MONO-01 Phase E1: rank-one eigenvalue-shift edit on a trained RNN (CPU, 2026-11-04)

Code: `phase_e1_recurrent.py` (written by a delegated subagent; 7 unit tests pass, rerun by the lead). Setup: 48-unit tanh RNN trained on noisy delayed recall at horizon T = 40 and read out at t = 40 and t = 160; model seeds 0, 1, 2; calibration 1,000 sequences and test 2,000 sequences with disjoint seed ranges (roster and prereg JSON in each seed folder); one held-out evaluation per seed (`test_ledger_*.json`). Numbers marked (seed 0) were checked against `noise0.5_seed0/test_results.json` by the lead; the rest are as reported by the subagent and are in the other seeds' result files.

## Observations
- Baseline test MSE (seed 0): 0.00199 at t = 40 and 0.01976 at t = 160 (seeds 1 and 2 within 0.001).
- There is no unstable mode in the mean Jacobian: leading real eigenvalue = spectral radius = 0.996, eigenvector condition 1.20, transient growth 1.25, negative Lyapunov exponents (seed 0). The recurrent weight itself has spectral radius 1.0108. So the experiment tested a slow, near-unit mode, not a true instability.
- Readout drift depends on amplitude (gain at t = 160 of 1.77, 1.27, 0.89 for small, mid, large |s|; subagent-reported).
- Bare eigenvalue edits break t = 40: calibration MSE at t = 40 rises 4x to 150x, because the trained readout gain is tuned to the decayed T = 40 signal. After this failure and before the test roster was touched, a readout-gain refit variant, a gradient line search and a zero-edit control were added (deviations from the original plan).
- Rule-selected edit `Jc:x0.95` (seed 0): dMSE@40 -0.00173 [-0.00181, -0.00165], dMSE@160 +0.309 [+0.296, +0.323] (chance-level). The subagent reports it was feasible only because a 6.5x gain refit cancels the contraction at t = 40.
- Principled edit `Jc:unit` (move the eigenvalue to 1.0), seed 0: dMSE@40 +0.00053 [+0.00051, +0.00055], dMSE@160 +0.0457 [+0.0437, +0.0475]. Achieved eigenvalue shift on T = 40 trajectories +0.0026 of +0.0037 targeted, and -0.0006 on T = 160 trajectories (subagent-reported).
- Controls (seed 0): equal-norm random and same-support random edits are inert (dMSE@160 about +0.0003 to +0.018 depending on arm); a line-search gradient edit that uses t = 160 labels gives dMSE@160 -0.0054 [-0.0060, -0.0048], about -27%.
- Prediction test (46 points, seed 0, test roster): Spearman between the local eigenvalue shift and dMSE@160 is +0.49 for `Jc:x0.95` and +0.75 for `Jc:unit`. The subagent reports the sign is negative (-0.06 to -0.18) on calibration, so the local shift did not predict long-horizon behavior consistently.

## Reading
Negative result: a rank-one eigenvalue shift derived from the mean Jacobian did not give a useful repair, and the local shift is not a reliable predictor of long-horizon error. Hypotheses, not isolated: the long-horizon error is amplitude-dependent tanh drift that a scalar eigenvalue shift cannot flatten, and the t = 40 sensitivity is the horizon-tuned readout meeting mu^40. No stability claim should be drawn from local eigenvalue shifts alone, as the workorder warned.
