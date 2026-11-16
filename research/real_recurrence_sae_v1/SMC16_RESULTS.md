# SMC-16: Selective suppression of naturally active matched SAE features

**Executed November 16, 2026.** The original [preregistration commit 777ff633](https://github.com/MoralityLabAI/Compactification-Theory/commit/777ff633010898103d876b54ac5d84264c12a628) precedes evaluation of any SMC-16 novel test states. This study follows [SMC-15 PR #176](https://github.com/MoralityLabAI/Compactification-Theory/pull/176) and remains a separate, stacked [SMC-16 PR #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177). Pure mathematics companion: [MathCan OLM-16 PR #97](https://github.com/patrickdugan/MathCan/pull/97).

[Hosted Actions run #43798110706](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43798110706) passed the source-locked four-puzzle pilot, full four-pair × two-cohort evaluation and independent standard-library audit. **Green CI means the experiment and replay executed; its primary scientific gate passed in 3/4 receiving SAE pairs.** The B501 SAE89 comparison failed its registered 30-hole coverage threshold and is retained as a negative/inconclusive-over-coverage result.

## Intervention and frozen evidence

Reuse the exact original pretrained source teacher A and SAE71, independently trained recurrent teachers B501 and B701 (each with SAE71/89), and all seven original SMC-13C position/visible-digit-conditioned Hungarian assignments. All original trainer/model/SAE receipt hashes and all seven feature-mapping SHA-256s were checked before generating fresh hidden test states. Unlike SMC-15's subtraction of a full physical *decoder contribution*, SMC-16 projects the receiving matched feature's *encoder covector* into the nullspace of the other three clean active features. It applies the unique minimum-norm edit that moves that feature's preactivation below zero by a fixed overshoot 0.02, preserving the other active coefficients to the predeclared full-SAE reencoding tolerance.

The matched treatment is paired with an equal-physical-norm sham projected tangent to **all four** clean active physical encoder covectors. Both must preserve their preregistered selected-activation constraints on unmodified SAE reencoding before the outcome label/margin is scored. The physical kill norm is capped at 6× the corresponding SMC-15 native contribution norm. Ineligible goals, projection failures and reencoding failures are explicitly retained and excluded without imputing zero effects. Each intervention changes exactly one of the 81 stage-one Sudoku token states and continues three FP64 recurrent steps.

Two fresh solution-disjoint noncanonical Sudoku cohorts, each with 64 independently seeded unique puzzles, use seeds **202616030000..202616030063** (30 holes) and **202616040000..202616040063** (34 holes). Every full solution is outside the original canonical Sudoku generator orbit by the alternating-rectangle invariant. Cohorts are distinct from all previous SMC-13 and SMC-15 trained/calibrated/tested grids. Eight missing goal tokens per puzzle are selected solely from the visible blank mask.

## Registered primary result

The primary endpoint is the **correct-digit margin loss after matched selective kill minus its equal-dose active-preserving tangent sham**. A larger positive difference means selective target-feature erasure damages the correct-digit margin more. Per-puzzle cluster bootstrap (2,000 fixed draws) gives descriptive 95% intervals; the 512 sampled blank goals in a given family are not treated as independent.

| Receiving SAE | Fresh 30-hole paired margin difference [95% interval] | Eligible goals / puzzles | Fresh 34-hole difference [95% interval] | Eligible goals / puzzles | Both-family gate |
|---|---|---:|---|---:|---|
| B501 SAE71 | +1.2388 [0.4757, 2.0105] | 37 / 28 | +0.9814 [0.2132, 1.7213] | 36 / 29 | **PASS** |
| B501 SAE89 | +1.6399 [0.7384, 2.7007] | **18 / 16** | +1.3238 [0.3756, 2.3014] | 26 / 22 | **FAIL (coverage)** |
| B701 SAE71 | +0.9622 [0.2305, 1.6846] | 47 / 35 | +0.7833 [0.0956, 1.4904] | 42 / 32 | **PASS** |
| B701 SAE89 | +1.1347 [0.4144, 1.8597] | 49 / 35 | +1.5699 [0.8083, 2.2373] | 48 / 38 | **PASS** |

The frozen gate requires **at least 24 eligible goals in at least 20 distinct puzzles**, 2,000 valid clustered bootstrap resamples, positive lower bootstrap bound for the mean margin-difference, and more correct-to-wrong target prediction flips after kill than after sham. Every tested condition has a positive descriptive lower interval. **B501 SAE89 on 30-hole inputs has 18 eligible goals in 16 distinct puzzles, making the registered overall pair decision FAIL.** Its numerical margin advantage and descriptive interval are preserved as observations.

## Accuracy flips, coverage and geometry

On eligible goals, the exact clean-correct-to-wrong flip counts after selective kill compared with tangent sham are:

| Receiving SAE | 30-hole kill / sham | 34-hole kill / sham |
|---|---:|---:|
| B501 SAE71 | 11 / 0 | 11 / 2 |
| B501 SAE89 | 5 / 1 | 8 / 1 |
| B701 SAE71 | 9 / 4 | 5 / 2 |
| B701 SAE89 | 12 / 2 | 13 / 1 |

**Missing target activation drives coverage.** For each 512 selected goal tokens, between 429 and 466 had an inactive mapped target feature and never entered the selective edit. Between 9 and 23 extra goals per condition failed full-SAE kill reencoding, and 6 to 27 failed tangent sham reencoding. Every eligibility reason is included in the source receipt. Among retained selected goals, the largest relative drift observed in the other active coefficients or tangent-sham preserved coefficients is approximately **4×10⁻⁷**, much smaller than the 1e−4 relative acceptance threshold. Mean physical selective dose is about 1.22× to 1.41× the prior native decoder erasure norm across conditions.

These findings give evidence for a task-functional role of mapped naturally encoded SAE features under a targeted geometric intervention. The conditional subset is small, and the screen on post-edit SAE support is an intervention-dependent selection criterion fixed without task labels. Changes to **inactive or previously unselected SAE preactivations** and nonlinear hidden-state pathways remain possible. The receiving teachers have much lower clean Sudoku accuracy than the original pretrained teacher, and all four model/SAE comparisons share task architecture and training distributions.

## Audited source receipts

- Original complete [SMC-16 hosted run #43798110706](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43798110706), code head \`aea163d740a5ee9f61bbbd5e9bae04f4a1ddda6c\`, with [full intervention and audit artifact #14194824412](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43798110706/artifacts/14194824412).
- Artifact ZIP SHA-256: \`6f34925766ce50ae2cbd2c219b877bb0a0c702a42fd12dbcae1663f9c5f436f7\`.
- Full raw \`full.json\` (3,634,157 bytes) SHA-256: \`534685723438e5b18d6145e8b61aac87d84bf0602e6f518d168c41720b58b728\`.
- Independent \`audit.json\` SHA-256: \`c138ef92d024123cd3dc01945a8d3a849fb79543aa9ff22687beb606616ddf5f\`.
- New-cohort SHA-256 manifests: 30-hole \`320f4521729abb177450004d31bf4acb64e50d2ff79b93a9997fc331a32029d4\`, 34-hole \`9c10ad8be3343101c0f45ad46ac7eadb30823e4515a6d95737ab952990dd3945\`.
- [SMC16_SUMMARY.json](SMC16_SUMMARY.json) records eight condition-level outcomes, 3/4 primary pairwise gates, full ineligibility reasons, flip counts, source hashes and independent audit provenance.

The hosted independent standard-library auditor replays the frozen mapping hashes, new Sudoku solution/goal-mask checks, all geometric eligibility certificates and physical dose values, observed per-goal margin pairs, clustered bootstrap intervals and registered gates. It validates saved numerical receipts with a different implementation; it does **not** independently re-execute PyTorch model forward passes.

## Next discriminating experiment

SMC-16 materially narrows the hidden-state interpretation: under a full-SAE checked edit, three other active features can remain unchanged while removing the matched one reduces downstream correct-digit margins. To distinguish mapped SAE identity from displacement of formerly inactive features, a next test should **explicitly measure and control the entire 96-dimensional preactivation displacement**, with a basis-matched null that aligns its inactive-feature disturbance to the selective kill. Separately, training independent teachers to similar Sudoku capability would remove a large behavioral mismatch. Both would require new preregistration and previously untouched test puzzles. The current SMC-16 gate and its coverage failure must remain intact.
