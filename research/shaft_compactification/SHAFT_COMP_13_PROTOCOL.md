# SHAFT-COMP-13 registered protocol: observability as a predictor of delayed behavioral divergence

**Registration:** 2026-11-11, before the held-out GRU scores are computed.
**Branch:** `research/shaft-comp-05` in `MoralityLabAI/Compactification-Theory`.
**Scientific claim under test:** Does lower effective observability predict future behavioral disagreement between two naturally reached states that the present monitor regards as equivalent?

## Confirmed external baseline: LOOPSCOPE-01 Q5

Repository: `MoralityLabAI/LoopedTransformers`, branch `rmp-v0-phase2`, `experiments/recurrence_monitorability_v0/loopscope_toy_q5_tca/q5_report.json`.
The **fixed Q5 rule** is `F1/F8 < 0.05` predicts held-out length-extrapolation success (accuracy >=0.90). Published outcome: 9/12 correct, 8/12 positive; Spearman(F1/F8, held-out accuracy) = -0.718. Q5's F is the summed Frobenius norm of parameter-gradient fragility across core matrices, evaluated by TCA. **It is a different observable from a recurrent-state observability Gramian.** The experimental comparison must keep both definitions and outcome labels distinct.

Q5 checkpoint files listed as `experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt/<arm>.final.pt` are absent from the current Git tree. Cross-check the published SHA-256 receipts before running any checkpoint-based observability analysis. Static published metrics alone cannot establish whether new state-observability features predict Q5 extrapolation.

## Part A: preregistered local GRU test

Use the pinned existing `TaskMemoryGRU`, `fit`, `train_probe`, `analytic_jacobian`, and `observability_blocks` from SHAFT-COMP-06/11/12. The base model has 24 hidden dimensions, four probe outputs, and a 16-task four-bit truth-table family.

* **Calibration model seeds:** 101, 202 (previously studied).
* **Held-out model seeds:** 303, 404 (not included in prior SHAFT-COMP-11/12 experiments).
* **State depths:** 12 and 40.
* **Sampling:** 16 goals × 16 *pairs* of independently sampled, naturally reached recurrent states per model/depth (256 pairs per cell), each pair sharing its goal ID. Probe and base model are fitted from their normal fixed seeds, using no data from outcome continuations.
* **Eligibility:** use pairs whose **four current native teacher query answers are identical** and whose **four predicted probe bits are identical**. Report numerator/denominator and excluded cases. The primary analysis conditions on this equality.
* **Diagnostic input schedule:** independently seeded five-step continuation for each member state. These inputs are used solely to construct the analytic Jacobian cocycle and observability features. They are independent of the outcome continuation.
* **Outcome:** independently seeded common 32-step input continuation for the two states. Label (Y=1) if their native four-query output signatures differ after those 32 steps. These bits are obtained from the frozen GRU readout, not the linear probe.
* **Primary score, frozen now:** (S=24-rac12(r_A+r_B)), where each (r) is the numerical rank of (mathcal O_5=[W;WJ_0;ldots;WJ_4cdots J_0]) using relative singular-value threshold (10^{-3}). **Higher score is registered as predicting more disagreement**.
* **Primary metric:** ROC-AUC of (S) for (Y) on the pooled eligible observations from held-out seeds 303 and 404. Use average tie ranks; constant or single-class cases are undefined and reported as such. Also report each seed and depth.
* **Predefined secondary scores:** (i) mean per-state (-log_{10}(sigma_{24}/sigma_1)) for (mathcal O_5), clamped at (10^{-12}); (ii) negative mean minimum absolute current probe logit margin; (iii) Euclidean distance between the paired hidden states. Only the primary score is confirmatory.
* **Significance and uncertainty:** report a two-sided within-cell (seed × depth) 1,000-permutation test of the primary pooled AUC (test against chance AUC 0.5), plus 1,000 stratified bootstrap resamples within seed/depth, reflecting within-model sampling only. The two model seeds remain the unit of model replication and cannot support broad generalization.
* **Comparative analysis:** describe baseline AUCs and observational coverage. No feature or cutoff tuning on held-out model seeds.

Any failure or reversal of the directional hypothesis is a legitimate negative result. Preserve unrounded per-pair rows, fixed seed manifests, model-parameter hashes, input generators, full summary JSON, and exactly reproducible scripts.

## Part B: Q5 checkpoint bridge

Build an opt-in loader accepting a local, user-supplied Q5 checkpoint directory and Q5 result JSON; verify each checkpoint against its **published hash**. For the looped transformer, the recurrent state comprises **all token positions**, not only the answer position. Evaluate a finite-dimensional projected/sketched observability stack with predeclared projection dimension and readout site, and distinguish *sketched rank* from full-state rank. Use the published 12-arm Q5 report unchanged, retaining the registered `F1/F8` threshold and held-out success labels. Compare predictors across arms only after checkpoint availability is verified, with matched training-independent evaluation items and no tuning on held-out success.

Do not infer a 10T model's ability to persist, exfiltrate, or reconstruct itself from these toy experiments. All work stays in simulation and local consented checkpoints.
