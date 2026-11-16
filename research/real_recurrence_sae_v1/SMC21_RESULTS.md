# SMC-21: Calibration-only SAE erasure residual transfer against direct clean-gradient prediction

**Executed November 16, 2026.** The original [SMC-21 prospective protocol commit \`0f0af2d\`](https://github.com/MoralityLabAI/Compactification-Theory/commit/0f0af2dc4b34c6a85dc4e0ba5d5d1e08597bbf59) preceded any new SMC-21 calibration gradients or fresh heldout Sudoku. The immutable original SMC-19 independent SAE refit checkpoints and 64-puzzle calibration effects were retrieved source-locked, and the new calibration fit was uploaded and independently audited **before** any fresh SMC-21 test states were constructed.

The first source-locked, complete [GitHub Actions run #43822895405](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405) passed all three distinct jobs: (1) original 64-puzzle calibration-only gradient/ridge fit and independent audit, (2) separate four-puzzle test engineering pilot and independent audit, and (3) full four-refit × two fresh 64-puzzle evaluation and independent standard-library replay. **Scientific verdict: 1 of 4 receiving SAE refit pairs passed the original strict both-family source-residual prediction gate.** All four prior SMC-19/20 checkpoint histories and their results remain unchanged.

## Prediction and original frozen comparison

On the *same unchanged frozen recurrent Sudoku teacher*, a reference SAE71 native feature erasure of physical decoder contribution \(a\), its matched new SAE131/167 refit native erasure \(b\), and a same-dose other-active refit native decoder edit \(o\) use the original post-stage-one FP32 hidden state. The target-digit-minus-clean-rival task margin \(m\) is continued for three FP64 recurrent steps. The clean FP64 physical hidden-state task gradient \(g=\nabla m(h)\) is computed after original visible-only feature eligibility is fixed.

Define finite native erasure \(E(v)=m(h)-m(h-v)\) and finite residual \(R(v)=E(v)-g^\top v\). The prespecified **primary baseline** is the strong SMC-20 direct clean-gradient predictor \(g^\top b\). The new source correction is \(\widehat E_B^S=g^\top b+\beta_S R(a)\), where \(\beta_S\) is fitted from **only the 64 original SMC-13 \`feature_calibration\` puzzles**. The equal-parameter, equal-one-extra-intervention comparison is \(\widehat E_B^O=g^\top b+\beta_O R(o)\). Both \(\beta\)'s use exactly the same fixed 5% ridge shrinkage and \([-2,2]\) clip, each fitted to the old *calibration* matched-refit residual \(R(b)\).

All source/refit feature mapping SHA-256s, calibration eligibility, original calibration target effects, FP64 gradients, physical edits and both \(\beta\)'s are frozen before new heldout testing. The **new** native refit target effect \(E(b)\) is evaluated only after the direct, source-corrected and other-corrected predictions have been computed.

The registered *fresh-cohort* success criterion per pair on **both** 64-puzzle groups requires ≥24 eligible goals in ≥20 puzzles, identified original calibration coefficients, all 2,000 puzzle-cluster bootstrap draws valid, positive 95% lower paired mean absolute-error improvement against **both** the direct-gradient and other-active calibrated predictors, \(\sum |E_B-\hat E^S| / \sum |E_B-g^\top b|\le0.95\), and source-corrected sign accuracy at least as high as direct-gradient sign accuracy. This is **different** from SMC-19's native SAE label stability gate (2/4) and SMC-20's anchored-vs-source-only gate (4/4).

## Immutable calibration result

Four original trained pairs: B501 SAE71→B501 SAE131, B501 SAE71→B501 SAE167, B701 SAE71→B701 SAE131, B701 SAE71→B701 SAE167. Each uses original SMC-19's *64 feature_calibration puzzles*, never its test outcomes or SMC-20 heldout data.

| Refitted SAE | Eligible original calibration goals / puzzles | \(\beta_S\) (old source residual) | \(\beta_O\) (other-active residual) |
|---|---:|---:|---:|
| B501 SAE131 | 198 / 63 | +0.502533 | −0.061381 |
| B501 SAE167 | 222 / 62 | +0.423139 | +0.166843 |
| B701 SAE131 | 165 / 61 | +0.396573 | +0.134844 |
| B701 SAE167 | 142 / 59 | +0.572915 | +0.157517 |

All eight fits meet their original nondegeneracy/coverage criteria. The coefficients have **one** calibration target/residual budget each. There is no new SAE retraining, heldout sign fitting or extra feature-matching optimization.

## Registered full test results

Novel source-disjoint exact-unique noncanonical Sudoku cohorts, each 64 full puzzles: **202621030000..202621030063** (30 holes) and **202621040000..202621040063** (34 holes). Eight evenly spaced visible-blank goal tokens per puzzle; full solution/input disjointness from every prior SMC-13/15/16/17/18/19/20 source family. New-cohort fingerprint SHA-256s: novel30 \`4ee05c509ec27d2c644f5a22a35e2b55edf78c6873742beaf6206304fc5055c3\`, novel34 \`8c1fd118450694614c5873f238c6aaf2613b5b9eb47e41d192bf47c055b32b43\`.

For paired gains, a larger positive value means **lower absolute new-refit erasure prediction error** under the original source-calibrated residual correction. Intervals are descriptive 95% cluster bootstraps over the 64 puzzle IDs, 2,000 seed-locked draws per pair/family. Each comparison uses the *same original eligible goals* for both predictors.

| Receiving SAE | Fresh cohort | Eligible goals / puzzles | Gain vs direct gradient [95% puzzle CI] | Gain vs other-active calibrated correction [95% CI] | Source-calibrated / direct MAE | Registered family |
|---|---|---:|---:|---:|---:|---|
| B501 SAE131 | novel30 | 166 / 63 | **+0.07534 [0.04028, 0.11608]** | **+0.08976 [0.05184, 0.13325]** | 0.7462 | **PASS** |
| B501 SAE131 | novel34 | 174 / 62 | +0.02966 **[−0.01460, 0.07810]** | +0.04419 **[−0.00128, 0.09327]** | 0.8891 | **FAIL** |
| B501 SAE167 | novel30 | 222 / 64 | **+0.07059 [0.04224, 0.09905]** | **+0.06089 [0.03539, 0.08904]** | 0.7608 | **PASS** |
| B501 SAE167 | novel34 | 209 / 64 | **+0.08661 [0.05758, 0.11906]** | **+0.08148 [0.04888, 0.11694]** | 0.7182 | **PASS** |
| B701 SAE131 | novel30 | 142 / 59 | +0.02160 [0.00142, 0.04431] | +0.00997 **[−0.01080, 0.03461]** | 0.9137 | **FAIL** |
| B701 SAE131 | novel34 | 166 / 62 | +0.02372 **[−0.00062, 0.04879]** | +0.02682 **[−0.00353, 0.06224]** | 0.9020 | **FAIL** |
| B701 SAE167 | novel30 | 132 / 61 | +0.02989 [0.00111, 0.05826] | +0.02254 **[−0.01101, 0.05875]** | 0.8704 | **FAIL** |
| B701 SAE167 | novel34 | 115 / 53 | +0.04167 **[−0.00041, 0.08473]** | +0.02801 **[−0.02537, 0.07848]** | 0.8454 | **FAIL** |

**Exactly B501 SAE167 passes both fresh families**, 1/4 pairs. The B501 SAE131 second cohort misses the paired lower-bound criterion on *both* required baselines. The B701 pairs have weaker gains, several nonpositive lower bounds against the matched-budget other-active control, and B701 SAE131 additionally loses target-erasure sign accuracy against the direct gradient on both cohorts. All original pass/fail criteria remain fixed, including failures.

## Pooled descriptive differences and interpretation

Across **1,326 eligible target goals** (out of 4,096 selected goal records in 512 pair/family/puzzle cases), the separate archived-number review recomputed the original source result:

| Predictive method | Pooled native refit target-effect MAE |
|---|---:|
| **Direct clean gradient \(g^\top b\)** | 0.27357 |
| **Calibration-only old SAE residual correction** | **0.22291** |
| Equal-budget calibrated other-active refit residual correction | 0.27208 |

On the original eligible weighted pool, the source-calibrated correction reduces MAE **18.52%** relative to direct clean gradient. Target finite-erasure sign agreement is **1,231/1,326** for source-calibrated against **1,224/1,326** for direct, a modest 7-goal advantage. These pooled figures are **descriptive** across correlated teacher/SAE comparison families and do not replace the original strict preregistered 1/4 pair-level outcome.

This supports a limited operational relationship between original SAE native-erasure **finite nonlinearity residuals** and new matched-refit erasure responses, after already using a strong answer-conditioned local task gradient. The effect is weaker and less consistently established under cluster uncertainty than SMC-20's original comparisons to source-only and other-active *prediction of the whole erasure*. SMC-21 tests the residual specifically, and its matched-budget control is permitted to use its **own other-active erasure observation**.

All heldout task gradients are defined using the **true Sudoku digit and highest clean competitor** as part of evaluation. This is **an answer-conditioned mechanistic causal diagnostic**, not an inference-time Sudoku solution method or a verified general reasoner. The same frozen B501/B701 teacher weights and old source generator underlie the pair comparisons; individual goal effects are not independent teacher replications. Full-rank inactive SAE encoder duals from SMC-18 and exact reconstruction-equivalent opposite-erasure examples in OLM-19 remain causal feature identity obstructions.

## Source of record, audit and reproducibility

- Original successfully completed [three-stage hosted run #43822895405](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405), source code commit \`6e947b411c8d234b86be25b65f6cc9dcc3b91728\`. Pilot, calibration-only fit and full new test jobs all passed their respective independent audit steps.
- Frozen **original-only** calibration coefficients, per-goal 48D gradients, physical decoder vectors, SMC-19 original causal erasure values, and independent calibration audit: [artifact #14204423839](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405/artifacts/14204423839); ZIP SHA-256 \`eb60ee3bc9be88b3b40c3d43a6064c08c89fc5e35fc0dc84c3e845a7a1fc0020\`; \`residual_calibration.json\` SHA-256 \`e7aa12a3da5b8eaa0132b5de45083ca1b2af5c38e556fef01e1f26d2b94f90af\`; \`calibration-audit.json\` SHA-256 \`c7ac249a4f5547300ab462840094179cd5c5f453578d9ba24b8f54dead103b4d\`.
- Full fresh-cohort \`full.json\` and source-separated \`audit.json\`: [artifact #14205506294](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405/artifacts/14205506294); ZIP SHA-256 \`d926eedb11c4d9541dcf392a8bde785350b5697dee68c228f84c095deabbf6ec\`; original \`full.json\` SHA-256 \`893503626376938009c4527d6cbe45b481c519c09c962f56799e5be3040f08fd\` (25,887,710 bytes) and \`audit.json\` SHA-256 \`cfa217aa28311acbbef0ab409b76b505338e93fd6ba203569b7264b5ff8e5040\`.
- Frozen original SMC-19 four SAE fit/checkpoint/calibration source: [artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904), original \`frozen.json\` SHA-256 \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`. No model or SAE was retrained in SMC-21.
- Independent stdlib replay checks original source + all four trained refit SAE file hashes, original SMC-13C/SMC-19 feature mappings, 64 original calibration subset and exact two univariate ridge fits, every new Sudoku roster/visible-goal selection, physical 48D edits and clean gradients, all three predictor formulas, confidence intervals and **16,000** registered puzzle-bootstrap resamples.
- **Independent separately written archived-JSON replay** reconstructed every original calibration coefficient and all eight means/bootstrap intervals on the same eligible records, and independently SHA-checked the original four refit checkpoint file bytes. The original model forward values and gradients remain source observations.
- The first hosted source-locked run passed. A duplicate simultaneous pull-request CI failed its independent audit on small FP32 last-bit differences relative to the original archived SMC-19 calibration coefficient values. **After original source-receipt-only tolerance adjustment** in commit \`de625200\`, a new full pilot/calibration/new-128-puzzle rerun passed. The narrow relaxation concerns comparing two separately run FP32 historical source evaluations; the new physical-gradient, native decoder reconstruction, residual and bootstrap checks retain their originally tighter numerical tolerances. The original preregistered controls, fits, datasets and source-of-record results are unmodified. Subsequent reruns' file-byte hashes may differ due to FP32 rounding even when scientific gate decisions are preserved.

## Mathematical companion and next discriminating study

[MathCan OLM-21 PR #109](https://github.com/patrickdugan/MathCan/pull/109) supplies written proofs of the unique fixed-shrinkage calibration ridge optimum, the exact risk shift in the three source/target second moments, and exact rational witnesses where *even identical calibration and heldout residual distributions* allow a squared-risk-optimal correction to worsen heldout **absolute** error. A second witness preserves source residual marginals while reversing target residual correlation. Its ten exact-Fraction tests and MathCan mathematical checks passed on the repaired branch; the mathematics is not yet Lean-kernel certified.

The next experiment should determine whether **cluster-robust calibration on absolute error**, or a source edit distance / clean gradient-conditioned residual model, improves on this fixed squared-loss scalar with an equally parameterized direct and other-active baseline. Any new choice belongs to a **new preregistration and source-only validation split**, using new unobserved test families. The original SMC-21 result remains 1/4.
