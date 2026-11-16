# SMC-17: Inactive-preactivation-matched tangent-sham control

**Prospective protocol committed November 16, 2026, before any SMC-17 test state is generated.** Inherits only frozen SMC-13 source checkpoints/conditioned Hungarian assignments and already published SMC-14/15/16 results. SMC-16's 3/4 primary gate, B501 SAE89 coverage failure and all previously scored puzzles remain unchanged. Pure mathematical motivation is [MathCan OLM-17](https://github.com/patrickdugan/MathCan/pull/100).

## Scientific question and primary contrast

SMC-16 showed that suppressing the receiving matched feature while maintaining the other three active affine encoder coefficients causes larger downstream Sudoku margin loss than an equal-dose random tangent sham on a small preselected subset. It does not control the 92 originally inactive SAE encoder preactivation changes. SMC-17 tests whether the effect persists against a *label-blind, inactive-preactivation-matched tangent sham selected from a frozen finite search budget*.

The matched kill delta and baseline tangent sham follow the original SMC-16 source-locked physical encoder constraints without modification: source feature strict top-one, matched feature among four strictly positive target features, minimum-norm projected kill with overshoot tau=0.02, norm cap 6× native SAE decoder contribution, exact preservation of other three clean active encoder rows, full SAE reencoding with maximum relative active coefficient drift 1e-4, tangent sham active preservation and identical physical norm. Retain all SMC-16 rejection reasons. Never condition eligibility on the true Sudoku digit or subsequent logits.

For the receiving SAE, let A be its four clean active 48-dimensional physical encoder rows, B the other 92 rows, and delta the original minimum-norm kill. Compute the exact FP64 matrix-geometric residual floor

\[
 \rho=\min_{s\in\ker A}\|B(\delta-s)\|.
\]

Implement rho with an orthonormal basis U of ker A obtained by a 48×48 full SVD, and least-squares projection of B delta onto the column space of B U. Record the 44-dimensional tangent rank, the full B rank, its minimum nonzero singular value, and numerical residual orthogonality. A full-column-rank B makes exact full inactive matching of a nonzero kill impossible, as proved in OLM-17.

## Frozen finite-budget inactive-matching sham

Preserve the first valid SMC-16 baseline sham (four original source-hash seeded attempts). Additional candidate directions are selected using **only the clean SAE fit, frozen model-independent hidden state and visible input metadata**:

1. The normalized minimum-least-squares tangent vector s0=U(BU)^+B delta and its negative, when ||s0||>1e−8.
2. Exactly **128** independent deterministic CPU Gaussian directions, each projected into ker A using the original FP64 SVD scheme. Use seed = unsigned first 63 SHA-256 bits of the literal string \`SMC17|pair_index|family|puzzle_seed|goal_slot|matched_feature|attempt\`, attempts 0..127.
3. The existing SMC-16 baseline sham is an additional candidate. Every candidate is rescaled to ||delta|| and screened for tangent residual ≤1e−8 and full-SAE reencoding of all four clean active coefficients as strictly positive with drift ≤1e−4*(1+z_l). Keep all successful candidates' inactive residual norms. Select the valid candidate with smallest ε=||B(delta−s)||, breaking ties in the fixed order baseline, positive s0, negative s0, then ascending Gaussian attempt.

The baseline is always in the candidate set, so any SMC-16-eligible goal remains eligible in SMC-17. Record both ε_baseline and ε_best, their ratio, the exact lower floor rho, the chosen candidate family, candidate count and all 96 physical preactivation change entries for each successful goal. If the selected matched feature is inactive, retain its rejection record and perform no outcome inference. No gradient, task label, continuation output, or Sudoku answer may influence candidate search.

## New source-disjoint roster and readout

Rebuild and SHA-check the seven SMC-13C calibration assignments on 64 original calibration puzzles *before* constructing the SMC-17 test cohorts.

Fresh noncanonical uniquely solvable Sudoku families:
- \`SMC17_N30\`: 64 puzzles with exactly 30 holes and full-grid seed bases 202617030000..202617030063.
- \`SMC17_N34\`: 64 puzzles with exactly 34 holes and seed bases 202617040000..202617040063.

Use the identical randomized MRV source generator and dig(seed+1000000) method, and explicitly exclude every SMC-13, SMC-15 and SMC-16 train/calibration/test complete solution and input. A four-puzzle prefix per family, first primary A71→B501 SAE71 pair only, is an engineering pilot with no positive gate. Four registered pairs are A71→B501/B701 SAE71/89; eight visible-mask-chosen blank goal tokens per puzzle. Freeze all eligibility and best sham choices before accessing any ground-truth digit. Continue exactly three recurrent steps in FP64 after a single edited stage-one goal token, holding all other 80 token states fixed.

For each eligible goal and teacher, fix correct digit and highest *clean* non-target competitor after selection. The primary outcome is

\[
 d_{\rm best}=(m_{\rm clean}-m_{\rm kill})-(m_{\rm clean}-m_{\rm best\ sham})
 =m_{\rm best\ sham}-m_{\rm kill}.
\]

Record the original four-seed random tangent sham's paired effect d_baseline on the same tokens. Also retain clean-correct-to-wrong flip flags for kill, baseline and best. All summaries use every eligible goal and report clean-correct-only effects as secondary.

## Frozen uncertainty and pass/fail

Token-weighted mean of d_best is primary, supplemented by the average of each eligible puzzle's own mean. Resample 64 independent puzzle IDs (all eight goals together) with replacement 2,000 times, using random.Random(202617130 + 100*pair_index + len(family)). Use sorted 2.5th/97.5th percentiles by indexes 50 and 1949. Require all 2,000 draws to have eligible goals.

**Registered success per pair and family** requires at least 24 eligible goals in 20 distinct puzzles, a strictly positive primary paired-bootstrap lower bound, more clean-correct-to-wrong flips under kill than under the best sham, and a **minimum 10% pooled relative reduction** in inactive-encoder residual: (Σ ε_best)/(Σ ε_baseline)≤0.90 across eligible goals, with Σ ε_baseline>0. Overall pair passes only when both fresh cohorts pass. These gates are frozen even if the stronger control changes the sign or coverage.

Report all eight pair/family conditions, their full eligibility distributions and all fitted geometry diagnostics. Additional null distributions: 128 candidate ε values per eligible edit, original first-valid random sham, and the two signed least-squares candidate outcomes. Only the selected best sham receives task-margin computation; finite candidate geometry distributions are *not* treated as independent neural experiments. Per-goal ε, rho and matching ratio are conditioning variables for descriptive analysis only; do not filter on them after observing task margins.

## Interpretation and audit limits

A positive paired effect after improved inactive matching strengthens operational specificity conditional on the controlled edit and retained coverage. OLM-17 demonstrates that exact inactive matching can be impossible under a full-rank 92×48 encoder block. Margin attribution to the single matched feature requires additional gradient decomposition and curvature bounds; the current numerical PyTorch experiment is not an Arb interval proof. Distinguish a low-rank/small-rho geometry from the actual finite-budget sham's achieved ε. Independent stdlib replay must check all registered hashes, roster/goal masks, eligibility and candidate geometry receipts, nonnegative residual floors, paired task margins and bootstrapped gates without relying on the Torch scorer. The raw full receipt must be archived with a SHA-256 and the source exact weights remain unchanged.
