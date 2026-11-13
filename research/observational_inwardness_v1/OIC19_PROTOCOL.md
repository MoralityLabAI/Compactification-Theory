# OIC-19 | Exact transition recovery with a continuous localized emission

**Prospective, unexecuted.** OIC-18 learned an n-dependent phase-localized emission that identifies model *families* at unseen n, while von Mises models frequently retain a diffuse rotating transition kernel. This protocol tests whether phase-family identification and exact phase propagation can be jointly learned from public histories.

## Registered question

For the same Z₂ × Zₙ finite hidden-cycle dynamics and public sensor, can a continuous stochastic circulant filter recover a concentrated ±1 rotating generator **up to the observational reflection gauge**, while retaining OIC-18 large-n classification, posterior calibration, and sensor fault tolerance?

## Models and matched controls

Retain the OIC-18 n²-concentrated von Mises emission and its 13 trainable baseline scalars. Compare under equal 500-step/128-example seed-matched training episodes:

1. Existing family BCE objective with the OIC-18 kernel regularizer, the frozen reference.
2. A stronger trained-kernel entropy penalty, coefficient selected using only training-n validation (n=4–6), with penalty arithmetic and training searches charged.
3. Multi-horizon **observable predictive log likelihood** of future raw-state readouts from intervention histories; do not disclose the initial phase or exact generator in the loss.
4. As an explicitly privileged ceiling, an auxiliary training loss supervised by the simulator's true phase, with the reflection/translation gauge minimized. Phase labels are unavailable at inference. Count those labels as privileged training information.

Preserve held-out initial phases 1 and n-2 and phase-count OOD n=7,8,12,16,24,32,64. Include five seeds per condition. Test a consistent non-dihedral latent-state relabeling by conjugating the underlying dynamics and emission together, and test ordinary reflection separately.

## Acceptance and falsification

Register both metrics and denominators before fitting: (a) learned rotating-kernel mass on the correct ±1 step; (b) conditional posterior phase-recovery mass after n consecutive probes, up to reflection/translation; (c) family classification accuracy; (d) Brier score; (e) all positions of one substituted/erased raw sample at the n+2 even-phase correction horizon; (f) training compute and live belief-state bits. Fixed-family nonzero phases are evaluated as their common observational equivalence class, not individually.

A strong positive result requires at least 0.90 correct-step kernel mass and at least 0.90 gauge-relative rotating-phase posterior MAP accuracy at unseen n=24 and 64, with family accuracy >=0.95 and prior-level passive posterior. Also report the fraction of runs that satisfy *all* criteria, not only means. The substitution panel must remain close to the exact codebook ceiling; classify a solution with any corrupted-case failures as a distributional success rather than a worst-case certificate.

Any stronger latent supervision or exact cyclic projection must be disclosed as a separate architectural or informational advantage. The finite model supplies no evidence about subjective experience or pretrained frontier-model internals.
