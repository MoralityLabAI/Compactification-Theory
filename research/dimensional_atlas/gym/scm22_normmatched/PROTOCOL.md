# SCM-22: Frozen CPU norm-matched causal subspace experiment

**Pre-data protocol.** Author before the held-out model intervention campaign. Run locally in a separate CPU-only Python process. Source lineage: SCM-20 -> SCM-21. This is a *fresh trained synthetic looped-transformer assay*, not a replay of SCM-20's private checkpoints and not a blinded EvalMath ARG-01 experiment.

## Question
Can a calibration-derived four-dimensional state-difference subspace transfer a named mathematical operation more effectively than equally sized, **per-pair norm-matched** SAE/head-fiber and random head-fiber subspaces, without altering clean mathematical predictions?

## Exact environment
- Prime field F_11. First operator in {(1,1),(1,3),(2,1),(3,2)} and second in {(1,2),(2,4),(3,1),(4,3)}, representing `t -> a*t+b (mod 11)`.
- Input: one-hot first operation (4), second operation (4), operand (11), irrelevant nuisance token (8). Exact target `g2(g1(x)) mod 11`, independently checked using Python integer arithmetic.
- Counterfactual pairs: identical second op, operand and nuisance token; change only first op. Exclude pairs with identical mathematical outcomes; require both unpatched predictions correct for the *primary clean-eligible* population.
- Model: 48-hidden, four-head, four-visit, *weight-tied* full-attention pre-norm block; a learned linear classifier reads final CLS after LayerNorm. Three training seeds (191,1191,2191), 400 steps, AdamW LR 0.002, batch128, FP32. No GPU use.
- Select fixed visit index 1 (the second recurrent visit) for primary intervention; report other visits exploratory. Matched evaluation done with frozen weights.

## Calibration and controls
- After supervised training, freeze each model and record source/checkpoint hashes. Generate 192 calibration paired cases from distinct generator seed (case generator labels hidden from PCA). At visit 1 take recipient/donor CLS differences; fit centered top-4 SVD/PCA.
- Train a shared TopK SAE (width 96, k=8) on 2048 unpaired baseline CLS states collected over four visits, using 250 CPU updates, then freeze. Derive a four-dimensional candidate fiber frame by projecting the four most commonly active decoder directions into the *linear* readout-kernel and QR orthogonalizing. A separate random rank-four frame is also drawn within that kernel. Conditioning the frame selection on donor labels is forbidden.
- For each eligible counterfactual pair, compare: (1) sham, (2) full donor CLS difference, (3) rank-4 difference-PCA, (4) SAE head-fiber rank-4, (5) random head-fiber rank-4, and (6) a rank-4 raw state PCA comparator. Every non-sham rank-4 patch is scaled to the same **per-case norm** as the full donor CLS difference; report separately the unscaled projection-energy fraction of that difference. A secondary half-norm test is registered before seeing evaluation results.
- The readout includes LayerNorm, therefore its linear kernel is **not** an exact nonlinear final-readout null. Explicitly test the terminal visit as a diagnostic.

## Outcome and interpretation
- Generate a fixed 256 held-out pairs per seed after calibration and freezing using seeds `20261117 + model_seed`. Do not select hyperparameters or intervention norms on these pairs. Report clean eligibility, all arm successes and exact pairwise discordances between difference-PCA and SAE-fiber. Use a paired exact binomial sign test descriptive only, because data share model seeds and finite semantic combinations.
- First report all three seeds, full clean accuracy, calibration basis ranks, explained variation, actual per-pair norm errors, task-family (second op) breakdown, and budget/CPU time.
- Positive mechanistic gate: difference-PCA beats both matched-rank/matched-norm fibers in every seed by at least 10 percentage points **and** achieves >=50% donor transfer on eligible pairs. If gate fails retain null.
- This measures a synthetic operation intervention, not source-certified topology, pretrained reasoning, a goodfire VPD decomposition, or causal benefit on trained SCM-20.
- Fail closed if a training run has fewer than 64 eligible pairs, if QR rank degenerates, or if per-pair norm matching error exceeds 1e-5.
