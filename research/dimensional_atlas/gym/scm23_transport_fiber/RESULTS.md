# SCM-23: frozen CPU transport-fiber results

**Date:** 2026-11-17. **Scientific gate:** FAIL because seed 1191 did not reach the registered 10% transport-gradient-fiber fidelity threshold. The complete local source package passed 25/25 tests and deterministic three-seed replay. This repository contains its short replay harness, fixed preregistration, and public summary.

## Primary held-out donor transfers, visit two

| Rank-four frame or control | Seed 191 | Seed 1191 | Seed 2191 | Total / 768 |
| --- | ---: | ---: | ---: | ---: |
| Full CLS replacement | 254 | 256 | 256 | 766 |
| Unrestricted difference PCA4 | 168 | 187 | 187 | 542 |
| Readout-visible difference PCA4 | 167 | 185 | 186 | 538 |
| **Gradient transport fiber4** | **36** | **20** | **46** | **102** |
| Difference PCA4 within head fiber | 20 | 5 | 10 | 35 |
| SAE head-fiber4 | 1 | 2 | 4 | 7 |
| Random head-fiber4 | 3 | 1 | 0 | 4 |
| Sham | 0 | 0 | 0 | 0 |

All 768 eligible pairs were clean-correct in both recipient and donor roles. Every non-sham rank-four intervention was rescaled to the full donor-state difference's norm for that *individual* pair, with maximum absolute norm error below 4.8e-7.

## What changes under head-fiber alignment

The frozen gradient-derived basis explains **2.96%** of raw squared donor-state difference energy on average, while the within-fiber difference PCA basis explains **7.32%**. Nevertheless, calibrated gradient orientation reaches 102/768 causal transfers while the variance-directed fiber frame reaches 35/768.

Paired gradient-versus-SAE discordance: 96 gradient-only successes, one SAE-only success. The three seeds share a finite mathematical generator, so pooled comparisons are descriptive. Gradient-fiber fidelity by seed is 14.1%, **7.8%**, and 18.0%. The preregistered positive gate requires at least 10% fidelity and a 5 percentage-point advantage over both SAE and random fiber arms in each seed.

At half patch norm, gradient fiber produces 9/768 transfers. No rank-four fiber arm achieves donor transfer at the final recurrent visit in this synthetic assay. The head-kernel property itself is strictly linear and final LayerNorm remains nonlinear.

The full standalone research package contains the original 768-pair Boolean outcomes, frozen hashes and independent audit. This GitHub source includes a deterministic replay that verifies every seed's numeric endpoints.

**Limits:** fresh synthetic F11 looped neural execution, supervised gradient calibration, no original SCM-20 checkpoint evidence, no EvalMath ARG-01 OOD claim, no certified Seifert topology or new trained-model reasoning advantage. Historical SCM-20 results are unchanged.