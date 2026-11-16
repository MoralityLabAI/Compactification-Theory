# SMC-17: Inactive-encoder-leakage-matched tangent sham

**Prospectively registered November 16, 2026; publish this file in a standalone commit before evaluating any SMC-17 fresh heldout states.** Stacked on [SMC-16 PR #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177). Reuse the precise source-frozen SMC-13C seven-feature mappings and the original SMC-11C/SMC-13A trained snapshots. The previous SMC-16 3/4 success and B501 SAE89 coverage failure motivated this investigation and remain fixed historical results.

## Question

SMC-16 selectively erased a mapped naturally active SAE atom while preserving the receiving SAE's other three active feature coefficients. Its tangent sham changed none of the four original active coefficients, but differed from the feature-kill edit in changes to the remaining 92 previously inactive **affine preactivations**. A causal effect can therefore arise from this unobserved displacement. SMC-17 constructs an **outcome-blind tangent sham that better matches this 92-dimensional inactive encoder displacement**, then tests the conditional task-margin difference.

## Frozen sources and fresh cohorts

Exactly the four A-SAE71→B501-SAE71, B501-SAE89, B701-SAE71 and B701-SAE89 primary comparisons. Rebuild and SHA-verify **all seven** original SMC-13C position/visible-digit-conditioned Hungarian and clue-shuffled assignment SHA-256 digests on the original 64 calibration Sudoku puzzles **before constructing any new heldout state**. Load immutable original pretrained and newly trained teachers and SAEs from the same three artifact receipts as SMC-16, checking file and parameter hashes.

Prepare two new 64-puzzle cohorts with the source-matched \`new_full_solution\` and \`dig_unique\` generators and exact unique-solution + positive intercalate checks. Disjointness is against every SMC-13, SMC-15 and SMC-16 full 64-puzzle cohort as well as the new families:
- \`smc17_novel_30\`: seeds **202617030000..202617030063**, 30 holes;
- \`smc17_novel_34\`: seeds **202617040000..202617040063**, 34 holes.

Full rosters and SHA-256 fingerprints are archived. The four-puzzle prefixes are smoke tests and cannot qualify as results. Exactly eight selected missing-goal positions per puzzle use the original visible-input deterministic rule. Target digits do not choose mapping, edits, eligibility, stopping or optimizer hyperparameters.

## Original fixed selective kill and seeded sham

Use SMC-16's original \`selective_vectors\` routine without modification to obtain \`δ_kill\` and \`δ_seeded\`. Only a goal that was already **eligible under SMC-16's complete pre-outcome strict source/target active-feature and full reencoding guards** can enter the SMC-17 comparison. The native dose cap remains 6× original target coefficient×physical decoder norm, kill overshoot \`τ=0.02\`, other-active coefficient drift ≤1e−4×(1+z), and the seeded sham must retain all four clean selected features. This keeps the baseline SMC-16 effect and eligibility count separately visible on fresh data.

## Frozen inactive-shift matching

At each SMC-16-eligible selected goal, let \`A\` contain the physical covectors of the four clean active receiving SAE features and \`I\` the 92 originally inactive physical encoder covectors, in numerical feature-index order. For an edit \`δ\` the *inactive affine preactivation displacement* is \`Iδ\`, which is independent of encoder bias. Let \`r=||δ_kill||=||δ_seeded||\`. Choose a sham from the sphere

\[
\mathcal S=\{\delta: A\delta=0,\ \|\delta\|_2=r\}
\]

by minimizing \`L(δ)=||I(δ−δ_kill)||²\` with the following deterministic, fixed-budget *approximate* solver. Compute an orthonormal nullspace basis \`N\` of A from a full FP64 SVD. In \`x∈R^{44}\`, let \`B=I N\`, \`y=Iδ_kill\`, \`x\` have norm r. Make exactly **four** starts: the SMC-16 seeded sham, \`Bᵀy\`, and two deterministic Gaussian seeds from \`SHA256("SMC17|pair|family|seed|slot|matched_feature|restart")\`, each projected and normalized onto the sphere. From each start take exactly **128** projected-gradient steps with

\[
g=2B^\top(Bx-y),\quad g_T=g-(x^\top g/r²)x,\quad
x\leftarrow r\frac{x-\eta g_T}{\|x-\eta g_T\|},
\qquad\eta=\frac{1}{4\|B\|_F^2}.
\]

Keep the original start and every iterate as an eligible candidate; select the candidate with lowest \`L\` **among those that pass the unchanged original SAE full reencoding check** (all four original active features remain positive and within 1e−4×(1+z)). Tie-break by start and iterate order, and require dose agreement ≤1e−8 relative. The original seeded sham is always a fallback candidate, so on every retained case the optimized inactive mismatch is at most the seeded baseline mismatch. This solver is a fixed-budget numerical search and carries **no global optimality claim**.

For each pre-eligible goal, record 48-dimensional FP64 kill, seeded and optimized physical edits, original active indices, all 96 physical encoder covectors (once per receiving SAE), the pre-outcome code, full 92-dimensional inactive shift differences, Euclidean leakage residuals, selected candidate identifier, reencoding error and finite status. The independent auditor recomputes the norm and inactive leakage from those raw vectors and the pinned mapping. No outcome-driven matching, feature selection or search retries.

## Causal test and preregistered gates

Intervene separately on **one stage-one Sudoku token** with each of the three immutable physical edits; all other 80 token states remain fixed at injection, then continue the unchanged teacher three FP64 recurrent steps. With the correct digit and strongest clean non-target competitor fixed per goal, measure margin loss \`clean_margin−edited_margin\`. Retain the existing SMC-16 seeded tangent sham and the new leakage-matched tangent sham alongside the kill.

On exactly the goals eligible for all three arms, compute per-goal \`D_optimized = drop_kill − drop_matched_sham\` and \`D_seeded = drop_kill − drop_seeded_sham\`. Average over eligible goals, with puzzle-cluster bootstrap over **64 distinct puzzle IDs**, 2,000 resamples, separate by pair and family; report unweighted per-puzzle means and clean-correct→wrong flip rates. Do not impute zero for ineligible cases.

**Frozen positive gate for each pair across both fresh cohorts:** (1) ≥24 jointly eligible goals in ≥20 puzzles; (2) all 2,000 puzzle-cluster resamples valid; (3) 95% lower bootstrap bound for mean \`D_optimized\` >0; (4) the kill correct→wrong flip count exceeds the optimized sham count; (5) pooled inactive mismatch RMS for optimized sham ≤**0.85** of the pooled seeded-sham mismatch RMS on the same eligible goals. Report individual family gate status and all four pairs, preserving any negative or low-coverage result. No retrospective changes to 0.85, optimizer steps, seed assignments, data split, or goal eligibility.

A 0.85 failure means the inactive-feature matching control was too weak for the intended scientific claim, even if the measured margin effect is positive. All bootstrap intervals are **descriptive finite-cohort** summaries; four comparisons share teachers, SAE fits and generators.

## Scientific limits

Matching the linear preactivation displacement of 92 inactive rows is generally infeasible exactly while satisfying the four active-row constraints and equal-norm sphere. The optimization is an approximation and a recorded residual remains; a positive result is a stronger *conditional intervention-specific* causal attribution, not proof of a uniquely identified feature or task algorithm. Other hidden-state directions, formerly inactive ReLU crossings, target-model capability mismatch and correlated puzzle families remain. The accompanying OLM-17 theorem proves a rank-based exact-matching obstruction and a computable spectral lower bound. An independent stdlib audit replays saved numerical observations and the registered gate; it does not constitute a separate recurrent-network forward pass.
