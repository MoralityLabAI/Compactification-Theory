# SCM-22: CPU matched-norm causal geometry results

**Evidence:** new sandbox CPU neural training and interventions on a finite-field synthetic mathematics task, November 17, 2026. Preregistered GitHub commit `56c14ace` precedes the held-out run. Existing SCM-20 model weights and gold are external. Registered primary gate: **PASS on this synthetic experiment**, without claiming transfer to SCM-20.

Each model has hidden size 48, 4 attention heads, 4 weight-tied visits, and a shared TopK-8 SAE (96 features). Three seeds received 400 base-model updates and 250 SAE updates each. Separate 192-pair calibration groups built four-dimensional intervention frames. A fresh 256-pair counterfactual sample was generated per seed, with both clean predictions correct on all 768 recipient/donor cases. First-operator swaps change an exact F11 answer. The source-layer intervention is after zero-based visit 1 (second visit).

## Registered primary test, rank and patch norm matched

| Arm | Seed 191 | Seed 1191 | Seed 2191 | Total /768 |
| --- | ---: | ---: | ---: | ---: |
| Full donor CLS difference (48d) | 255 | 256 | 256 | 767 |
| Difference-PCA rank 4 | **194** | **197** | **181** | **572 (74.5%)** |
| Raw-state PCA rank 4 | 183 | 195 | 173 | 551 |
| TopK SAE head-fiber rank 4 | 5 | 4 | 2 | 11 |
| Random head-fiber rank 4 | 1 | 2 | 0 | 3 |
| Sham | 0 | 0 | 0 | 0 |

All non-sham patches were normalized to the **same exact per-case target norm**, `||h_donor - h_recipient||_2`. The largest observed absolute norm discrepancy was **4.7684e-7**. The difference-PCA basis accounts for **56.2%–62.9%** of calibration difference variance; its mean held-out *unscaled* energy capture was **54.2%–56.4%**, versus **2.9%–4.3%** for SAE fiber. The patch rescaling removes the raw norm confound from the outcome comparison; natural projection energy and direct readout alignment still differ.

The registered criterion required difference-PCA fidelity at least 50% and at least 10 percentage points over both rank-four head fibers in **every seed**. All three pass. From exact paired outcomes, difference-PCA succeeded when SAE fiber failed on **190**, **194**, and **179** cases, with SAE-only successes **1**, **1**, and **0**; these paired sign-test calculations are descriptive because the same finite semantic combinations recur and the models share design assumptions.

At half the intervention norm, difference-PCA was **49/51/57**, compared with full-difference **99/102/112**. Magnitude matters even after orientation is controlled.

## Exploratory second stage: restrict the task-aligned PCA to each fiber

These additional analyses were written **after** inspecting the registered outcome. They are explicitly post-hoc.

| Rank-four calibration-derived frame at visit 1 | Seed 191 | 1191 | 2191 | Total |
| --- | ---: | ---: | ---: | ---: |
| Unrestricted difference-PCA | 194 | 197 | 181 | 572 |
| Difference-PCA restricted to linear head **rowspace** | 186 | 199 | 188 | **573** |
| Difference-PCA restricted to linear head **kernel** | 13 | 8 | 11 | **32** |
| SAE head-fiber | 5 | 4 | 2 | 11 |

The rowspace-restricted basis preserves most of the functional transfer. Within the head fiber, even a task-aligned calibration frame transfers relatively little. The result points to **readout alignment** as a major factor beyond raw dimension and patch norm.

Post-hoc visit sweep, within-fiber task-PCA successes across the three seeds: visit 0 **111/768**, visit 1 **32/768**, visit 2 **0/768**, visit 3 **0/768**. A rank-four head-readable task-PCA carries **360**, **573**, **536**, **498** respective successes. The output has LayerNorm, so the linear-head kernel is an approximate gauge for the nonlinear readout, not a globally exact null.

## Reproducibility and limits

- Local Python 3.13, PyTorch 2.10.0 CPU, NumPy 2.3.5, two CPU worker threads; primary full deterministic replay **PASS** and independent post-hoc replay **PASS**. Eleven source-unit tests pass.
- One full registered replay measured **18.29 seconds** elapsed, peak RSS **349,664 KB** on the sandbox CPU. No GPU resources were used.
- Frozen protocol SHA-256 `5daa2344fabb543331e3a430fa88cb689984e4f4ccf1965b856ca927d8a0a1d4`. Public counts and source/checkpoint hashes in `results/scm22_public_counts.json`. The exact full local registered and post-hoc JSON receipts are included in the downloadable research package.
- This is a finite operation-composition toy. Fresh nuisance instances can share the same underlying finite operation combination as training, so there is **no new mathematical family generalization claim**. The SAE/linear-head fiber controls share dimension and norm with difference-PCA, while the readout visibility differs; the post-hoc rowspace/kernel controls expose that difference.
- The **original SCM-20** write span was learned under a different objective with 128 hidden dimensions; it remains external to this CPU sandbox. These synthetic effects neither reproduce that checkpoint nor establish a pretrained reasoning mechanism. They also make no continuum braid, Seifert-surface isotopy or guaranteed interpretation claim.

**Follow-up:** On the actual SCM-20 checkpoints, compare native write-span vs task-calibrated difference PCA **within the same head fiber**, norm-match per pair, and perform a visit sweep. Keep the unchanged held-out pairs and clean-accuracy guard while disclosing the retrospective status of those source cases.
