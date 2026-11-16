# SMC-20: Anchored task-gradient prediction of independent SAE refit native-erasure effects

**Prospective registration, November 16, 2026.** This protocol is committed by itself on a new branch before generating new SMC-20 test states or scoring any SMC-20 outcomes. It follows original source-locked [SMC-19 PR #184](https://github.com/MoralityLabAI/Compactification-Theory/pull/184), whose primary native-erasure concordance gate remains **2 of 4** refit pairs, including both failed pairs. Stacked on SMC-19 and preserves SMC-11 through SMC-19 checkpoints/experiments as immutable.

## Question

Two independently refitted SAEs can produce similar observed feature activity on a frozen recurrent Sudoku teacher while their native decoder interventions produce different task outcomes. Is the target refit's **finite native decoder-erasure task-margin effect** predictable from the original SAE's measured erasure effect plus the **clean task gradient projected onto the physical difference between the two edited contributions**?

This distinguishes a mechanistically motivated, parameter-free physical-gradient correction from merely transferring source native causal effects or using a different naturally active SAE decoder direction as a surrogate. A positive result predicts a *task-functional response* between two dictionary coordinate decompositions. The source/reference SAE erasure is an allowed observation, and the refit receiving erasure is the strictly reserved prediction target. This is not a semantic identity or an independent teacher-retraining experiment.

## Frozen source and preregistered inputs

Reuse **unchanged** original SMC-11C original pretrained A teacher and SAE, both independently trained SMC-13 B501 and B701 44,001-parameter four-loop teachers and old SAE71/89; verify original SMC-11C \`joint-full.json\` SHA-256 \`7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087\`, SMC-13 \`full.json\` SHA-256 \`1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3\`, and SMC-13C \`conditioned.json\` SHA-256 \`3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07\`, plus original embedded checkpoint/SAE fingerprints and **all seven** original clue/position-conditioned Hungarian mapping hashes, before any fresh state generation.

Reuse precisely the **four** SMC-19 newly trained SAE weights snapshots, trained on original SMC-13 B501/B701 hidden states (each new seed 131/167), from independent [SMC-19 frozen archive #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904). Its \`frozen.json\` SHA-256 is \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`, and archive ZIP SHA-256 \`43630f8b9e6e7fbd07f8c668dc607f8b53c3a6b78cd47ce43aefe33bc005cb9e\`. Source refit checkpoint file hashes and original 64-puzzle calibration-only matched/other arm orientations remain exactly as frozen in that receipt. No new SAE fitting, mapping, checkpoint choice, decoder scaling or outcome orientation occurs in SMC-20.

Four comparisons, original frozen order:
1. B501 SAE71→B501 SAE131.
2. B501 SAE71→B501 SAE167.
3. B701 SAE71→B701 SAE131.
4. B701 SAE71→B701 SAE167.

## Fresh data and fixed eligibility

After frozen model, SAE and all seven original and four refit calibration map SHA checks, generate **two new source-disjoint exact-unique noncanonical Sudoku cohorts**, each 64 puzzles:
- \`smc20_novel_30\`: full-grid seeds **202620030000..202620030063**, 30 missing clues.
- \`smc20_novel_34\`: full-grid seeds **202620040000..202620040063**, 34 missing clues.

Generate via the exact SMC-13 randomized MRV complete Sudoku generator and uniqueness-preserving dig with \`dig_seed=grid_seed+1000000\`. Require all 81-digit solutions valid, positive alternating-rectangle invariant (outside original canonical orbit), exact unique completion, exactly 30/34 blanks, and new complete solutions and input puzzles distinct from *every* original SMC-13, SMC-15, SMC-16, SMC-17, SMC-18, SMC-19 cohort, train/validation/calibration included. Persist complete rosters and SHA-256. The four-puzzle prefixes are smoke tests only, never scientific passes.

Select exactly 8 blank goal positions per puzzle using the original visible-mask evenly spaced rule. At each goal, the original SAE71 feature must be positive strict top-one, and its frozen SMC-19 matched refit feature must be naturally positive among the top-four with at least one **different naturally active refit feature** available. The comparison and reference, target and control physical decoder vectors are fixed *before* looking at the correct Sudoku digit or any continuation output. All eligibility reasons must be retained. No filter by true digit, observed source causal response, gradient magnitude or downstream effect.

## Three native physical edit directions and clean-gradient predictor

On the exact same original frozen target teacher and its FP32 post-first-loop 81×48 hidden state h, independently compute:
- \(\delta_A=z^{\mathrm{old71}}_i(h)d^{\mathrm{old71}}_i\) in original physical units;
- \(\delta_B=z^{\mathrm{refit}}_{\pi(i)}(h)d^{\mathrm{refit}}_{\pi(i)}\);
- \(\delta_O\), an *equal-physical-L2-dose* edit along the original SMC-19 source-blind deterministic choice of a **different naturally active refit SAE decoder column**. The refit target and \(\delta_O\) have identical physical dose, up to original FP64 tolerance.

Freeze the original exact physical edit vectors, their L2 norms and cosine/distance, selected feature IDs and SMC-19 source-independent alternative feature ID. For any selected goal, continue the frozen teacher three FP64 recurrent steps after editing **one goal token**, leaving the other 80 tokens fixed at injection. Let \(m(h)\) denote the **correct Sudoku digit minus the highest clean non-target competitor** logit, with the competitor fixed across all edits. A native causal erasure margin drop is

\[
E(v)=m(h)-m(h-v).
\]

The source anchor \(E_A=E(\delta_A)\) is observed before target inference. The target \(E_B=E(\delta_B)\) is the **evaluation label**. A different naturally active refit SAE edit \(E_O=E(\delta_O)\) remains a separately reported intervention control.

Calculate the **exact FP64 clean 48D token gradient** \(g=\nabla_h m(h)\) by original SMC-18 \`task_gradients\` using one differentiable replicated clean anchor per eligible goal, original model FP64 block/head and *no training*. Verify the clean autograd forward margin agrees with original frozen clean logits within \`2e−8\` and gradient is finite. Freeze three *prediction formulas* before reading the refit target-erasure outcome:
- **Anchored-gradient prediction (primary)** \(\widehat E_B^{\,G}=E_A+g^\top(\delta_B-\delta_A)\).
- **Reference-only predictor (baseline)** \(\widehat E_B^{\,A}=E_A\).
- **Other-active surrogate physical-gradient predictor (matched-dose null)** \(\widehat E_B^{\,O}=E_A+g^\top(\delta_O-\delta_A)\).
- Direct clean linear estimate \(g^\top\delta_B\) and physical decoder cosine/distance are descriptive only.

Use raw **unoriented signed task-margin drops** in this prediction experiment, because source and target erasures are in the SAME unchanged physical teacher and use the same clean digit/rival. Original SMC-19 calibration-only sign orientations are retained as provenance and not reused to change numerical target signs. Do not choose thresholds or a regression gain on SMC-20 heldout outputs.

## Primary per-goal errors and registered two-family success gate

For each eligible goal define:
\[
\ell_G=|E_B-\widehat E_B^{\,G}|,\qquad
\ell_A=|E_B-E_A|,\qquad
\ell_O=|E_B-\widehat E_B^{\,O}|.
\]
The two primary paired improvements are \(I_A=\ell_A-\ell_G\) and \(I_O=\ell_O-\ell_G\), on the **same eligible goals** with the **same true target**. A positive value means the task-gradient correction makes a more accurate target-erasure magnitude prediction than that control. Also report the sign-match rate \(\operatorname{sgn}(\widehat E_B)=\operatorname{sgn}(E_B)\), with the fixed \(|x|\le10^{-8}\) zero convention, separately for every predictor.

For each pair × novel family, retain all 64 puzzle IDs. Compute token-weighted means of \(I_A, I_O, \ell_G,\ell_A,\ell_O\), the relative pooled error ratio \(\sum\ell_G/\sum\ell_A\), raw source/refit drops and per-seed/per-puzzle summaries, with **2,000** puzzle-cluster bootstrap resamples of **64 puzzle IDs** with replacement. Seed Python \`random.Random(202620130+100\,\mathrm{pairIndex}+\mathrm{len(family)})\`. Use the sorted bootstrap 2.5% and 97.5% indices **50** and **1949**, respectively. Any draw without eligible goals is invalid; all 2,000 must be valid. Each cluster includes every selected goal.

**Preregistered positive gate per pair across BOTH fresh cohorts:**
1. ≥**24 eligible goals in ≥20 different puzzles**, all 2,000 resamples valid, and finite predictions/physical norms.
2. 95% lower bootstrap bound **strictly positive** for the mean \(I_A\) and also strictly positive for mean \(I_O\).
3. The pooled heldout absolute error ratio \(\sum\ell_G/\sum\ell_A\) is ≤**0.90**, with \(\sum\ell_A>10^{-8}\).
4. The heldout **sign accuracy** of anchored-gradient prediction is **at least** that of the original source-only predictor on all eligible goals, using the zero convention for both.

If the reference-only predictor already transfers erasure effects perfectly, \(\ell_A=0\) and the pair fails the ratio condition by abstention. Report all four pairs and their eight conditions, the failed gate components, the raw source outcomes and all nulls. The random-feature/within-clue SMC-19 Hungarian maps remain separate observational controls, since their eligible support differs and an unpaired comparison of their task signs would change the target subset. Source-refit positive gate and CI do not imply population extrapolation from this correlated synthetic task generator.

## Honest interpretation and planned mathematics

The physically natural anchored estimate \(\widehat E_B^{\,G}\) measures the difference between two distinct SAE feature-erasure *directions in one fixed recurrent teacher*. Even a successful prediction cannot establish that original/matched SAE labels correspond to a uniquely defined semantic mechanism. It tests local physical task-gradient transport of interventions.

OLM-20 will prove a **difference-sensitive anchored-gradient remainder bound**, using a Hessian operator norm M verified over the triangle between h, h−δ_A, h−δ_B:
\[
|E_B-E_A-g^\top(\delta_B-\delta_A)|
\le {M\over2}\big(\|\delta_A\|+\|\delta_B\|\big)\|\delta_B-\delta_A\|.
\]
This bound becomes small for physically similar edits, even when the individual edits themselves are large. When \(\|\delta_A\|=\|\delta_B\|=r\), it reduces to \(Mr\|\delta_B-\delta_A\|\). An associated **conditional sign certificate** arises if the predicted target effect exceeds the provable remainder bound in absolute value. SMC-20 provides numerical clean autograd gradients and edit distances, **not an independently verified global M**, so it cannot label any neural case a certified sign conclusion.

An independent **standard-library receipt auditor** must recompute original+refit frozen source hashes, four refit checkpoint *file byte* hashes, exact visible-mask-selected goals and original seven plus four new Hungarian mapping assignments, 48D physical edit distances/doses and gradient dot products, the three prediction formulas, all prediction loss measures and sign accuracy, original 2,000 puzzle bootstrap draws and frozen gates. It does not independently rerun the neural FP64 forward/gradient pass from checkpoint weights. Preserve raw per-goal gradients, physical edit vectors, margins, native control responses and abstentions in a full hosted artifact. The artifact must include the frozen checkpoint directory or reference its immutable separate download with SHA. CI is green if execution/audit succeed, regardless of scientific gate.
