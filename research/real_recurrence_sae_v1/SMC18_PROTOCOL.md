# SMC-18: Task-gradient attribution under full-rank inactive SAE leakage

**Prospectively registered November 16, 2026.** This file must be a standalone Git commit before generating any SMC-18 fresh test states or neural-gradient outcomes. Stacked on the existing [SMC-17 PR #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179), retaining SMC-17's **1/4** pair gate and all negative/coverage results. SMC-17's original full run #43800600355 is fixed evidence. This is a new experiment and the prior SMC-13 through SMC-17 cohorts are disjoint from its test cohorts.

## Question and hypotheses

SMC-17 found that improving the tangent sham's match of all 92 initially inactive SAE preactivation changes leaves only **1/4** source→new-teacher SAE comparisons passing its registered gate. Every previously tested inactive 92×48 encoder block was full column rank, making exact matching of the inactive displacement to a nonzero selective active-feature kill impossible. We now measure the *actual task-output gradient in 48-dimensional physical hidden state*, separating the receiving model's first-order causal response from finite-displacement curvature.

A successful SAE feature-specific causal attribution requires a task gradient that reacts to the selected kill beyond the best inactive-matched tangent sham. Because the 92 inactive rows span all 48 physical dimensions, the gradient also admits an entirely inactive-feature dual representation. That fact constrains how any reported gradient can be named or interpreted.

## Immutable sources and eligible intervention geometry

Reuse the same original SMC-11C pretrained recurrent Sudoku A and SAE71, SMC-13A independently trained B501/B701 recurrent teachers and each SAE71/89. Verify SHA-256 of all original SMC-11C, SMC-13A and SMC-13C JSON source receipts, binary checkpoint/SAE hashes and the **seven** original position+visible-digit conditioned Hungarian assignments **before constructing any new test hidden state**. Do not fit a new checkpoint, SAE, assignment, optimizer step, intervention radius or outcome orientation.

Primary four pairs in frozen order: A SAE71→B501 SAE71, A71→B501 SAE89, A71→B701 SAE71, A71→B701 SAE89. Reuse unmodified **SMC-16 selective physical kill** and the **SMC-17 fixed 131-candidate inactive-matched tangent sham**. Eligibility, physical doses, re-encoding screening, sham choice, and all active/inactive SAE feature rows are frozen using only stage-one states and visible puzzle inputs before any ground-truth digit or model continuation outcome is scored. Require the original SMC-16 full-SAE reencoding checks and a full-column-rank inactive encoder block; retain exact causes of abstentions. FP32 clean stage-one anchors, FP64 three further recurrent steps, one modified goal token, other 80 states unchanged.

Two fresh exact-unique noncanonical cohorts with 64 independently sampled puzzles each:
- \`smc18_novel_30\`: completed grid seeds **202618030000..202618030063**, 30 holes;
- \`smc18_novel_34\`: seeds **202618040000..202618040063**, 34 holes.

Use source SMC-13 randomized MRV full-grid generator, dig seed grid seed +1000000, positive 2×2 alternating-rectangle invariant and bounded exact-unique Sudoku solver. Verify the new complete solutions and visible puzzles are disjoint from every original SMC-13, SMC-15, SMC-16 and SMC-17 cohort and each other. Publish complete rosters and SHA-256 manifest. A four-puzzle prefix is a smoke test and cannot satisfy the scientific gate. Select exactly eight blank goal positions per puzzle by the frozen visible-mask indices floor(k×n_blanks/8), k=0..7. No target-dependent choice.

## Primary task gradient: clean FP64 autograd

For every eligible receiving goal, let \`h\` be the model's original FP32 stage-one hidden states converted to FP64. Fix the true target digit \`y\` and highest **clean** non-target competitor \`c\` for the same original frozen teacher. Let \`m(h)\` be its final y−c logit margin after three FP64 recurrent blocks. Obtain the 48-dimensional \`g=∂m/∂h_goal\` from torch.autograd on the original **clean** anchor, freezing all model parameters. Use one independent replicated clean anchor per eligible goal, each with a differentiable input tensor; vectorize gradients by summing the independent per-replicate goal margins. Require finite g and check the clean numerical margin matches the original frozen clean FP64 continuation within 2e−8. No new training, gradients of checkpoint parameters, or target conditioning before geometry selection.

Let \`k\` be the original SMC-16 physical selective kill and \`s\` the SMC-17 outcome-blind best inactive-matched tangent sham, both with equal norm r. Preserve the original finite intervention margin loss:
\[
D=m(h-s)-m(h-k).
\]
The clean first-order attribution is \`L=g·(k−s)\`, with components \`L_k=g·k\`, \`L_s=g·s\`. Record the signed finite remainder \`R=D−L\` and absolute ratio \`|R|/(|D|+1e−8)\` per eligible goal. This is a local linear prediction of an already-defined finite task intervention; it does not assume small finite remainders.

To measure curvature, evaluate *four additional full FP64 continuations*, at \`h±0.05 k\` and \`h±0.05 s\`, holding the clean rival fixed. Define central second directional responses \`q_k=[m(h+.05k)−2m(h)+m(h−.05k)]/(.05²)\` and similarly \`q_s\`. The local second-order finite displacement prediction is \`Q=L−(q_k−q_s)/2\`. Record \`D−Q\` and the improvement in absolute error over L as **secondary**, because the Taylor remainder requires unverified third-order derivative bounds. A model's piecewise nonlinearities may cross activation cells; this is measured rather than assumed absent.

## Full-inactive gradient representation and nonidentifiability

Let \`B\` be the 92×48 *physical* inactive SAE encoder covectors, with row indices in ascending frozen feature order. Compute its singular-value spectrum, verify rank 48 at tolerance \`σ_min>1e−10\`, and set its minimum-L2 dual coefficient
\[
\lambda=B(B^\top B)^{-1}g,\quad B^\top\lambda=g.
\]
Require a residual \`||Bᵀλ−g||≤1e−7(1+||g||)\`, otherwise abstain with a dedicated reason. Record \`||λ||\`, \`||B(k−s)||\`, spectral minimum singular value and the rigorous first-order Cauchy–Schwarz ceiling \`||λ||·||B(k−s)||\` on \`|L|\`. This dual decomposition *always* expresses the clean physical gradient in inactive-feature coordinates when B is full column rank, so a positive L **alone cannot identify an active SAE semantic mechanism**. Preserve each sample's row-rank and residual checks.

## Frozen primary scores and gates

For each pair and fresh cohort, retain all puzzle/goal entries (eligible, ineligible and failure reasons). The primary paired finite response \`D\`, clean gradient predictor \`L\`, and signed agreement are measured on the **same** frozen eligible goals.

Report: number of eligible goals and distinct puzzles; puzzle-cluster-weighted means of \`D\`, \`L\`, \`D−L\` and \`|D−L|\`; sign-agreement fraction among goals with \`|D|>1e−8\` and \`|L|>1e−8\`; positive clean-correct→wrong kill-minus-sham flip counts; gradient/inactive dual residuals and all curvature diagnostics. For primary means and paired differences, resample **64 independent puzzle IDs** with replacement exactly **2,000** times, preserving all goal slots in a sampled puzzle. Use \`random.Random(202618130 + pair_index×100 + len(family))\`, report descriptive 95% percentile bounds at sorted indexes 50 and 1949. Mark invalid if a bootstrap draw contains zero eligible goals.

A **frozen prospective gradient-support gate for a pair across BOTH new families** requires:
1. At least **24** eligible goals from at least **20** independent puzzles, 2,000 valid puzzle-cluster draws, and a 95% lower bootstrap endpoint for mean finite \`D\` **strictly above zero**.
2. On nonzero-response goals, at least **24** sign-comparable effects, and the **95% lower puzzle-bootstrap endpoint for signed agreement minus 0.5** strictly above zero; valid resamples must contain at least one comparable goal.
3. Pooled \`mean(|D−L|)/mean(|D|)\` **at most 0.50** among eligible goals, with \`sum|D|>1e−8\`.

These stringent thresholds test whether the *clean local task gradient* adequately describes the observed finite effect in addition to reproducing SMC-17's margin direction. Report all four results even if none passes, never tune parameters or controls after seeing test data. The sign cutoff and zero-response handling are fixed at 1e−8. No statistical population claim is made from four shared-model comparisons or repeated puzzle-generator families. Grade the gradient-support gate and all individual criteria separately from green CI execution.

Secondary metrics include the second-order \`Q\` predictor, one-sided kill and sham gradient components, normalized inactive leakage norm and the ratio of inactive dual ceiling to \`|L|\`. No posthoc filter can alter the registered sample or gates.

## Mathematics and audit

OLM-18 in MathCan will prove the sharp equal-dose finite Taylor gradient discrepancy; uniqueness/minimal norm of the inactive-row dual representation; a gradient-basis nonidentifiability obstruction; and an exact rational third-order finite response counterexample. Kernel Lean verification is a separate, explicitly scoped future task.

An independent **standard-library receipt auditor** (different code from PyTorch runner) must replay roster hashes and Sudoku validity, original mapping hashes, eligibility, gradient dot products, finite remainders, spectrum/dual residual diagnostics, retained task-margin cases, puzzle-level bootstrap and frozen pass/fail. CI uses pinned Python 3.11 and source-compatible CPU torch==2.7.1, original artifacts and no retraining. The raw full JSON with all goal observations and audit must be archived with SHA-256 digests. Nothing from the old SMC-17 heldout output can be repurposed as a new SMC-18 result.
