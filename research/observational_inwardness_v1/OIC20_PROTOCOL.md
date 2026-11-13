# OIC-20: Predictive state learning under adversarial raw-sensor faults

**Prospective, unexecuted (2026-11-10).** This protocol follows the completed OIC-19 study. It targets the remaining n=8 single-substitution errors in otherwise accurate predictive cyclic filters. No OIC-20 trained-model result is claimed.

## Registered hypothesis

Combining an observable predictive next-state likelihood with a fault-aware raw-sample training distribution can retain the OIC-19 rotating-kernel and gauge-relative phase-recovery gains while correcting every registered single raw substitution in the exact finite evaluation panel.

The baseline treatment is the *frozen OIC-19 predictive* model, with 13 trainable parameters, `n=4,5,6` training, seeds 0–4, splits full/heldout, 500 Adam steps × 128 episodes, learning rate .035. All training examples must be generated from the same family-blind action schedule and exact finite simulator, with n and public histories as the only inference-time inputs. True phase labels must be excluded from the nonprivileged treatments.

## Frozen controls and prospective variations

1. Frozen OIC-19 clean predictive loss and original raw-likelihood model (all 10 archived weights).
2. Predictive loss with 4% independently substituted raw sensor bits and 4% independently marked erasures during the 500-step training run, matching OIC-16/OIC-18 augmentation RNG and reporting any extra training-work cost.
3. A raw-sensor corruption likelihood with a two-sample temporal dependency explicitly modeling \(\Delta b_t=f_t\oplus f_{t+1}\). Train with the same data, charged live state and additional parameters. Avoid using the exact hidden phase or the correct constant shift at inference.
4. Exact deterministic cyclic-codebook decoder as the privileged structural ceiling, together with an input-matched generic recurrent observer as a learned control.

The observation schedule remains `1^n 0^2` for even n. The primary robustness set is n=8, 24, 64 with all initial phases and every individual substituted position; secondary tests include single marked erasures, double substitutions at the appropriate longer horizon, and passive-family leakage. Report n=7,8,12,16,24,32,64 clean OOD behavior separately.

## Locked metrics and acceptance

Freeze seed-averaged and worst-case phase accuracy, full-position substitution and erasure rates, Brier calibration, rotating-kernel mass on the correct gcd class, and rotating-phase MAP modulo the observational gauge. For the restricted five-shift model report the \(\pm1\) mass separately. Require all five seeds in both phase splits to retain at least .90 mass on the correct rotating-step representative, .90 gauge-relative phase MAP and at least .95 clean family accuracy at n24/64, with a passive posterior deviation below 1e-4.

The predeclared *robust* success gate is **zero errors** across all one-substitution raw-sample positions, both family labels, all initial phases and all five seeds at n=8,24,64. Any positive error count constitutes a failed worst-case gate, regardless of very high mean accuracy. Report exact error counts and positions. Training use of synthetic sensor-fault labels and any additional assumption about fault position must be charged to the information budget.

## Controls and evidence labels

Hold out two initial phases z0=1,n−2 from training. Include clean/passive and corrupted/no-probe controls, non-dihedral conjugate latent-state relabeling of *both* transition and marker, reflection, and an unmatched noise-channel negative control. Archive each optimizer seed, complete weights, per-position fault counts, evaluator source hashes and a standard-library independent scalar replay. Report a loss of phase-recovery accuracy when optimizing robust likelihood even if the family classification improves.

OIC-T26 proves that constant shifts are identifiable only up to gcd class under a uniform unknown phase and one marked state. Each reported generator score must respect that observational equivalence. The experiment is a synthetic finite-state monitoring exercise; it provides no general neural or philosophical inference about inwardness.
