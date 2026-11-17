# OIC-18: Phase-resolution scaling without a hard-coded one-hot emission

**Prospective, unexecuted.** OIC-16's post-hoc n=9–32 scaling sweep found that a shared low-frequency Fourier emission broadens across increasing phase counts. The exact OIC-T25 bound explains why fixed coefficients cannot preserve a constant adjacent-phase logit margin indefinitely.

## Hypothesis

An n-dependent phase-localizing emission with controlled bandwidth improves learned continuous-belief identification across previously unseen n, retaining comparable sensor robustness without projecting its dynamics onto an exact group generator.

## Preregistered comparison

Train n=4,5,6 with five seeds and separate heldout phase splits, as in OIC-16. Freeze training interventions and 500-step/128-episode budget. Compare:

1. The original fixed-coefficient Fourier emission with learned circulant kernels (OIC-16 baseline).
2. A trainable n-dependent temperature whose sharpening is constrained to a declared polynomial family, with its parameter count and n supplied to all controls.
3. A periodic von Mises-shaped soft pulse whose concentration is learned while its n scaling is explicitly billed as an architectural prior.
4. An equal-capacity higher-harmonic emission, charging all Fourier coefficients and their runtime arithmetic.

Keep an **exact deterministic projection** as a reference ceiling and the generic OIC-15 GRU as a baseline with matched inputs and documented unmatched compute. Preserve a matched noncyclic label permutation control, phase-reflection/gauge control, and unchanged passive prior gate.

## Evaluation

Registered OOD phase-count tests: n=7,8,12,16,24,32,64. Use n=7,8 only for comparison with the original OIC-16 preregistered gate; the remaining counts become prospective OOD for the new OIC-18 training run. Freeze all architecture choices and hyperparameters before inspecting OIC-18 results.

Report clean balanced family accuracy, Brier score, phase-posterior quality modulo reflection and translation, exact generator-distance spectra, raw state robustness to every single substitution or erasure at T=n+2 (even n), and resource ledger including n-dependent memory, parameter bits and per-step operations. Report held-out worst-case phase performance and errors separately from sample means.

## Acceptance and falsification

A transfer improvement requires at least a predeclared 10-point gain over the old OIC-16 learned continuous filter at n=24, retaining clean n=7,8 accuracy and a prior-level passive posterior. Success on corrected state-codebook classification that requires a new hard-coded one-hot emission should be reported as an architecture-assisted reference, separate from learned continuous inference. Include full model weights, saved predictions, source hashes, a torch-free receipt validator, and independent seeds.
