# SCM-13 frozen extension registration: affine disks and campaign integrity

**Registered:** 2026-11-14, prior to trained RMP activation capture for SCM-13. Prior SCM-01..12 experiments remain unchanged. This is a new, exploratory analysis track stacked on SCM-12.

**Primary question:** Does a two-dimensional affine Taylor surrogate explain a model's fixed-chart intervention braids, and where does a certified nonlinear remainder become necessary? The first run is a read-only analysis of SCM-12's original calibration-selection receipts, with no adjustment to existing interventions or hyperparameters.

**Model and source:** original RMP `LOOP-T s1`, `FF-U s3`, eight visits; C3 checkpoints, selected `D-shared` dictionaries, feature-provenance file, and immutable SHA-256 source/weight manifest must pass SCM-12 preflight. The external artifacts are absent from the present execution host. No model training, test-split access or resource-cap changes are authorized by this registration.

**Data:** 64 group-distinct tasks in the selection half of `monitor_calibration`, exactly 16 each from pointer chase, gated pointer chase, reachability and sudoku4, selected with seed 20261113. Use the existing `inject_after_visit=0`, injection token 4, observed token identities 16,32,48, radius 0.05 in source-token RMS, 32 and 64 angular samples, frozen SAE chart and random orthogonal *readout only* chart with identical intervention axes. For sampling audits, use the 32-angle frames; do not silently promote a 64-angle agreement to a continuous certificate.

**Frozen algorithms:** rational affine-disk threshold and clearance in `scm13_affine_disk.py`; uniform per-strand Hessian remainder function; read-only first harmonic screen in `scm13_diagnostic.py`; frozen full-campaign provenance check in `scm13_receipt_integrity.py`. Fix source and Python dependency revisions and run the immutable synthetic fixture before any trained capture.

**Decision policy:** Complete capture and provenance checks are mandatory before reporting four-arm comparisons. An exact-source continuous certificate requires independently validated source Jacobian, Hessian, chart-coordinate and floating-point enclosures at every relevant visit and token. Unavailable Hessian or rank-deficient relative Jacobian gives `inconclusive`. Report artifact absence as `blocked_external_artifacts`, and label all sample-only results as empirical diagnostics.

**Negative controls / falsifiers:** (a) a nontrivial Artin action with zero pair windings should invalidate any incorrectly asserted full-rank affine null certificate; (b) a true collision on the critical circle must refuse; (c) a nonlinear perturbation above its margin must refuse; (d) four-turn alias from sparse sampling stays uncertified; (e) mixed synthetic/real rows, task-level checkpoint changes, chart-hash drift, feature-pair drift and mismatched cross-model inputs must refuse; (f) randomized readout must leave task logits exactly unchanged within the fixed execution contract.

**Future real-model extension:** A radius sweep and counterfactual capability analysis require a separate registration with chosen radii, held-out task families, source-verified nonlinear bounds, matched-compute budgets and multiple model seeds. Four-dimensional Seifert-surface isotopy remains outside this instrument's mathematical scope.
