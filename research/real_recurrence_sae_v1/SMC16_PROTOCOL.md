# SMC-16: Selective active SAE feature kill with encoder-preserving sham

**Preregistered November 16, 2026, before any SMC-16 fresh-cohort intervention.** Stacked on [SMC-15 PR #176](https://github.com/MoralityLabAI/Compactification-Theory/pull/176), inheriting the frozen trained teachers, SAEs, SMC-13C mapping and prior SMC-14/15 source receipts. Prior heldout results motivated this protocol; this is a new prospective experiment.

## Objective

SMC-15 showed that subtracting a naturally active matched SAE decoder contribution changes Sudoku task margins and clean-correct-to-wrong rates, beyond an equally sized other-active decoder subtraction. That operation often changes adjacent SAE features because physical SAE decoder columns need not invert their encoder rows. SMC-16 isolates whether the receiving teacher's **selected matched SAE coefficient** carries a task-relevant causal signal that survives conditioning on the three other naturally active coefficients.

## Frozen sources, cohort and comparison roster

Recover exact original SMC-11C pretrained teacher A and SAE71, SMC-13A independently trained teachers B501/B701 and their SAE71/89, and the SMC-13C seven pairwise position+clue conditioned Hungarian feature matches and within-clue shuffles. Recompute and SHA-verify **all seven** assignments on the 64 original calibration puzzles **before constructing any SMC-16 test state**. The primary roster is the same four A71→B501/B701 SAE71/89 comparisons. The receiving SAE's matched feature j is determined by the source clean top-one feature i and frozen original mapping.

The new deterministic cohorts have distinct full Sudoku solutions from every SMC-13 and SMC-15 train, calibration and test sample:
- **SMC16_N30**: 64 independent noncanonical complete grids, seed base **202616030000**, with exactly 30 uniquely solvable blank cells each;
- **SMC16_N34**: 64 independent noncanonical complete grids, seed base **202616040000**, with exactly 34 uniquely solvable blanks.

Use the original SMC-13 randomized MRV full solution and source-matched unique-hole digger with dig seed=grid seed+1000000, preserve entire input/target/rectangle inventory and SHA-256 manifest. A four-puzzle prefix per family is a pilot only, and never counts as a positive full result. Eight selected blank positions per puzzle use the SMC-14 visible-mask rule and target labels select no features, controls or doses.

## Encoder-local selective physical intervention

At each goal token, freeze the FP32 stage-one hidden state h and original receiving SAE top-four ReLU code z. Require strict positive uniquely highest source feature i, receiving mapped j active, and **four strictly positive active receiving features** with j among them.

Let a_l=w_l⊘scale be the receiving SAE physical encoder covector. With O the three other active rows, project a_j orthogonally onto the nullspace of O:

\[
v=a_j-O^\top(OO^\top)^{-1}Oa_j.
\]

Abstain if O has deficient row rank, if \|v\|≤1e−8, or the numerical orthogonality check exceeds 1e−8. Set a *fixed* encoder-preactivation overshoot τ=0.02 and the selective subtraction

\[
\delta_{\rm kill}=\frac{z_j+\tau}{\|v\|^2}v,\qquad h_{\rm kill}=h-\delta_{\rm kill}.
\]

This is the minimum-norm physical state edit that moves j's affine encoder preactivation to −τ while leaving all three other clean active preactivations unchanged **inside the original affine cell**. Before looking at logits or puzzle labels, require \|\delta_{\rm kill}\|≤6× the SMC-15 original physical native decoder-contribution norm and re-encode the changed state with the original frozen SAE. Retain only cases where j now has zero top-four activation and each other originally active coefficient remains positive and within **1e−4×(1+z_l)** of its clean value. Log all ineligible reasons, initial/final top-four supports, physical doses and error residuals.

Construct a deterministic seeded Gaussian direction and project it into the nullspace of **all four** clean active physical encoder rows; normalize and rescale it to exactly \|\delta_{\rm kill}\|. Retry at most four predeclared seeds and abstain if the projection has norm ≤1e−8 or orthogonality error >1e−8. This **tangent sham** preserves all four active affine preactivations locally. Re-encode h−δ_sham and retain the episode only if each of the four originally active coefficients remains positive and within 1e−4×(1+z_l) of clean, including their top-four membership. Count ineligibility without consulting any digit label or model output.

Both arms change exactly one goal-token hidden state per rollout; the 80 others stay fixed at injection. Continue three steps in FP64 with the original frozen teacher. Target digit and highest non-target clean competitor are fixed once per teacher/puzzle/goal; report true-digit margin loss (clean minus ablated), and clean-correct-to-wrong flips, along with exact norm ratio and re-encoding diagnostics. All interventions are numerical (not Arb-certified recurrent-network behavior).

## Frozen primary estimand, uncertainty and acceptance

No sign orientation is learned. The primary per-goal treatment effect is:

\[
d_{p,g}=[m_{\rm clean}-m_{\rm kill}]-[m_{\rm clean}-m_{\rm sham}]
=m_{\rm sham}-m_{\rm kill}.
\]

Pool only the pre-outcome eligible goals within each source/receiving SAE pair and test cohort. Report the token-weighted mean of d and also the unweighted mean of per-puzzle eligible averages. Resample **64 independent puzzle IDs** with replacement 2,000 times, preserving all eight goal slots within each cluster; require all 2,000 draws have at least one eligible goal. Report descriptive 95% percentile bounds. Record every attempt (including filtered goals and all clean-correct flags) with fixed reasons.

The **predeclared pair-level success criterion** on **both** fresh 30-hole and 34-hole families requires ≥24 eligible goals across ≥20 distinct puzzles, 2,000 valid puzzle-level bootstrap draws, a strictly positive 95% paired-bootstrap lower bound for the mean true-digit margin loss difference, and a larger matched-kill *clean-correct-to-wrong flip rate* than sham among eligible goals. Report all four pairs regardless of success, never retune τ, dose limit, selection or primary gate after seeing test outcomes.

Secondary analyses: clean-correct-only margins, matched-kill versus sham flip counts and rates, sham top-four retention, target feature kill fidelity, active-other coefficient drift, coverage, norm ratio distribution, and posthoc comparison with prior SMC-15 **without reusing its heldout puzzles**. Distinguish every secondary result from the registered primary gate.

## Interpretation

A positive result shows task-output sensitivity to a **minimal-norm physical edit selectively suppressing a naturally encoded matched feature**, relative to a same-dose edit tangent to the original top-four affine encoder constraints. It still cannot identify a unique internal variable, because other inactive encoder preactivations and the teacher's hidden-state geometry can change. A low-coverage failure or nonpositive margin gate constrains how far SMC-15's decoder-erasure effect can be localized. This design tests previously trained teachers with substantial source/new capability disparity. The two receiving SAE fits and two teacher seeds are correlated comparisons, not independent population replications.

The companion OLM-16 mathematical result proves the projection minimum-norm theorem, rank obstruction, exact tangent-sham invariance on the affine encoder cell and sharp first/second-order margin sensitivity bounds. A separate stdlib receipt auditor must replay the eligibility, test roster, physical dose/encoder checks, puzzle bootstrap and preregistered gate. The original source artifact hashes and all seven SMC-13C assignments remain locked.


## Pretest implementation lock (November 16, 2026, before SMC-16 pilot/full evaluation)

This implementation detail amendment is appended to the already committed protocol before introducing fresh-cohort outcomes. It preserves all earlier selection thresholds and the primary success gate.

- Compute geometry in FP64 from physical encoder rows (encoder.weight divided elementwise by the frozen 48-state scale); compare SAE activations after conversion of edited FP64 states back to FP32, matching the original encoder computation.
- For rank, use the CPU PyTorch matrix_rank default SVD tolerance; reject singular solves. Require numerical projection norms >1e-8, relative orthogonality residual <=1e-8, and the original reencoding tolerance 1e-4*(1+z_l) for all protected coefficients.
- Take four deterministic sham Gaussian candidate seeds from the first 63 bits of SHA-256 of the literal tuple (SMC16, pair_index, family, full-grid seed, goal slot, attempt 0..3); the four draws are fixed before looking at labels.
- Each pair/family cluster bootstrap uses random.Random(202616130+100*pair_index+len(family)) and resamples the family puzzle count with replacement 2,000 times. Take sorted percentile indexes floor(0.025*2000) and floor(0.975*2000)-1, as in the earlier source-locked scoring style. Require 2,000 bootstrap draws with at least one eligible goal.
- Compare exactly four pretrained A SAE71 to B501/B701 SAE71/89 pairs. Source feature top-one must be positive and strictly greater than its runner-up. Matched receiving feature must belong to the four strictly positive receiving features. Outcome labels are accessed only after the candidate eligibility masks have been fixed.
- Scientific gate failures remain reported outcomes; execution errors, source hash mismatches, geometry invariance failures after an eligible edit and independent-audit inconsistencies fail the CI job.
