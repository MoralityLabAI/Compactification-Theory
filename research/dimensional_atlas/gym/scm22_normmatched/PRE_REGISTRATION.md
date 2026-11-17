# SCM-22 pre-data registration (2026-11-17)
Scope: CPU-only fresh synthetic neural control, stacked conceptually after merged SCM-21 PR #205. No access to SCM-20 private trained weights; do not imply trained-model replication or EvalMath ARG-01 results.
Frozen extended protocol SHA-256: `5daa2344fabb543331e3a430fa88cb689984e4f4ccf1965b856ca927d8a0a1d4`.

**Math**: Exact F11 sequential affine maps. First-operator pairs change only first of four possible maps. Second map, operand and nuisance token stay fixed. Exclude pairs with equal outcomes and audit clean correct recipient and donor predictions with independent Python integer arithmetic.

**Model**: Three PyTorch CPU FP32 training seeds 191,1191,2191. A weight-tied four-visit, four-attention-head, hidden-48 pre-LayerNorm model with CLS readout; AdamW LR 0.002, 400 training steps, batch 128, exact-math cross-entropy. This is a fresh synthetic model, not SCM-20's 231K model.

**Frozen feature calibration**: shared TopK-8 SAE, 96 hidden features, fit 250 updates on 2048 independent unpaired examples across visits; 192 calibrated donor-recipient pairs per seed, with seed +90001. Fit rank-4 difference PCA on visit-1 donor minus recipient states; a rank-4 raw-state PCA; rank-4 SAE-decoder directions projected into the frozen *linear* output head kernel; and a seeded random rank-4 head-kernel frame. The last normalization LayerNorm makes the head kernel a local linear control, not an exact nonlinear null.

**Tests**: After model/SAE/frames are frozen, create exactly 256 distinct heldout matched pairs per seed using seed 20261117+model_seed. Primary patch after visit index 1 (second visit). Only clean-correct, distinct-answer pairs qualify, with >=64 qualifying pairs required. Six arms: sham, full-CLS donor difference, difference-PCA4, raw-PCA4, SAE-fiber4 and random-fiber4. For every qualifying pair, each non-sham patch has norm equal to the norm of the full donor CLS difference; secondary half-norm patches multiply by 0.5. Fail if relative/absolute norm discrepancy exceeds 1e-5. Terminal-visit results are descriptive. Report numerator, denominator, pairwise discordance, exact paired sign test (descriptive), projection energy, second-operator subgroups, source/checkpoint hashes and resource use.

**Registered gate**: difference-PCA4 donor-transfer fidelity >=50% and at least +10 percentage points over SAE-fiber4 and random-fiber4 in EACH of three seeds. Retain all null results and stop on bad rank, nonfinite loss, or fewer than 64 eligible cases.

**Limitations**: Finite math task domain can share underlying operator combinations between training and heldout even with fresh seeds; no extrapolative general-math claim. This instrument does not certify continuous activation topology, identify SCM-20's learned VPD span, or test any external GPU checkpoint. Preserve original SCM-20 and SCM-21 results.
