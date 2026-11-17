# SMC-15: frozen two-feature causal-tangent conditioning in recurrent Sudoku teachers

**Prospectively registered 17 November 2026**, before new SMC-15 test puzzles are generated or scored. This is the first file committed to the branch. Pilot runs use only the original SMC-13 calibration cohort. The newly seeded SMC-15 cohorts remain untouched until the pilot and source-integrity gates pass.

## Rationale and hypothesis

[SMC-14](SMC14_RESULTS.md) established source→independent-teacher signed causal margin transfer, though the corresponding target SAE feature was naturally active on only 8.6–16% of goals. [SAU-CURVE02](../curved_sae_gauge_v1/SAU_CURVE02_RESULTS.md) mathematically demonstrates near-singular tangent frames for coactive curved atoms. SMC-15 asks whether pairs of naturally active **linear** SAE features in frozen *trained recurrent Sudoku teachers* produce distinguishable causal effects after three recurrent steps and whether small local response singular values predict finite-dose indistinguishability. This measures nonlinear **downstream causal geometry**, with no trained curved decoder.

## Frozen source and cohorts

Use exactly the SMC-14 source pretrained A (44,001 parameters, checksum 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1), B501/B701 independently trained teachers, original A SAE71 and the four B teacher SAE71/89 frozen snapshots.

Required immutable SMC-14 artifacts, SHA-256:
- SMC-11C original pretrained SAE joint-full.json: 7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087
- SMC-13A independent teacher/SAEs full.json: 1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3
- SMC-13C clue-conditioned assignments conditioned.json: 3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07

Their source GitHub Actions runs were 43721281348, 43728006836 and 43729335868. All four original A→B matched/clue-shuffled correspondence hashes must reconstruct from SMC-13's 64-puzzle feature_calibration cohort before new target labels or hidden states enter memory.

A four-puzzle calibration-only pilot checks source compatibility, native self-patch parity and geometry. Full evaluation uses two independently generated, uniquely solvable, noncanonical Sudoku cohorts:
- smc15_novel30: 64 solutions with new_full_solution(seed), seeds 202615030000+i (i=0..63); dig 30 holes with dig_unique(solution, seed+1000000, 30)
- smc15_novel34: likewise 202615040000+i with 34 holes

Require positive rectangle/intercalate invariant, verified unique solution, exclusion of all original SMC-13 training, calibration and test full-solution hashes. Publish puzzle and target hashes. Never access ARG-03-N's 168 sealed cases.

## Feature selection and interventions

Every teacher and its fixed SAE operates on identical visible puzzles, independently of correctness. Use SMC-14 goal_positions() to select 8 evenly spaced visible blanks. At each selected stage-one token, choose two highest strictly positive SAE activities. Admit only when top-1 > top-2 > top-3 >= 0 and both physical decoder vectors are nonzero. Select pairs from these activations alone; no task labels, downstream logits or intervention outcomes enter selection. Exclude failed goals without replacement.

Normalize each physical decoder as SAE.decoder.weight times training scale, following SMC-14. Both unit directions u,v are FP64 and act on an unchanged FP32 stage-one anchor. Exactly three FP64 native recurrent steps follow. All arms use the same PyTorch eager CPU path with no compiled hooks. Check exact equality of clean versus zero-dose self-patched logits.

Use r_small=0.08 for a centered local finite Jacobian, r_verify=0.20 for independent finite-dose continuation and r_joint=0.20 per component for joint additions. For each feature, compute its local response in native nine-digit output space:
J_i = center_9((logits(h+0.08*d_i)-logits(h-0.08*d_i))/0.16)[digits 1..9].
For positive column norms, normalize each J_i and take sigma_min of the resulting 9x2 frame. Record physical 48x2 decoder-frame singular values separately. Zero causal columns must abstain.

Take right singular vectors of the normalized output response frame for maximal/minimal sensitivity. Construct the matching physical combinations with each normalized to the SAME L2 radius 0.20. Measure actual nine-logit centered changes against clean at h+delta_max and h+delta_min and their min/max norm ratio. Record task margin effects separately. Since singular vectors come from the test goal's local tangent, this is a within-goal two-dose validation of a predetermined intervention rule, not an independently calibrated predictor.

Also patch +0.20u, +0.20v, +0.20(u+v). Measure joint native correct-digit margin minus the clean-based additive extrapolation of the single patches, recording the physical norm of the joint edit. Add a seeded random 48-vector projected perpendicular to span(u,v), scaled to EXACTLY the joint edit norm, as a matched physical-dose nuisance control. Rank failures or sham norm mismatches abstain.

## Registered readouts and inference

Report eight-goal eligibility, column norms, physical and output-frame sigma_min, finite-dose min/max effect ratio, signed true-minus-strongest-CLEAN-rival Sudoku margins, joint nonadditivity, matched nuisance response and complete per-puzzle receipts, for all five SAE components in each new cohort. Choose the rival only from clean logits.

The low-conditioning threshold is normalized output-frame sigma_min < 0.10, frozen from the earlier SAU-CURVE02 toy. A teacher/cohort comparison is informative only if each conditioning group has at least 10 independent puzzles with an eligible goal and at least 30 eligible goals total. Report per-puzzle means, never treating eight goals as independent puzzles; bootstrap whole puzzles 2,000 times with seed 20261517 and give descriptive 95% intervals. A strict exploratory gate requires a positive cluster-bootstrap lower bound on the high-minus-low conditioned min/max native-response ratio, replicated on both novel cohorts for at least two independently trained teachers. Any gate using test-derived sigma categories remains explicitly conditional and does not establish predictive transfer.

The cross-checkpoint coactivation audit uses the FOUR frozen A→B source feature permutations from SMC-13C. At each of A's selected goals check whether both mapped features are active among B's top four at the same puzzle position. Report count out of 512 positions per comparison and family, plus abstentions. The cross-model coactive subset does not redefine local feature selection. No success is inferred from a too-small subset.

All source hashes, calibration assignments, interventions, target labels, puzzle rosters and runtime metadata must be emitted or checked. Original SMC-14 and ARG-03-N test gates remain untouched. The study measures native numerical FP64 effects, with no Arb-validated exact-real certificate.
