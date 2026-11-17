# SHAFT-COMP-13: Registered observability-rank prediction of delayed divergence

**Registration date:** 2026-11-11, before running the held-out GRU experiment. [Read the frozen protocol](SHAFT_COMP_13_PROTOCOL.md).

**Result:** **negative** for the registered primary predictor. Effective observability rank at horizon 5 and relative singular-value tolerance 10^-3 did not predict delayed disagreement between naturally reachable hidden states in held-out small GRUs. The held-out ROC-AUC was **0.504439**, 95% within-model bootstrap interval **[0.416862, 0.586862]**, within-seed/depth permutation p-value **0.963037**.

The experiment concerns controlled finite truth-table tasks and two newly seeded, independently trained toy models. Its results should not be extrapolated to hidden objectives in deployed frontier systems, agent persistence, or distributed device recovery.

## Scientific bridge to the existing LOOPSCOPE Q5 result

The canonical source is [LoopedTransformers Q5, branch rmp-v0-phase2](https://github.com/MoralityLabAI/LoopedTransformers/blob/rmp-v0-phase2/experiments/recurrence_monitorability_v0/loopscope_toy_q5_tca/q5_report.md). SHAFT-COMP-13 pins a [read-only twelve-arm reference](results/q5_registered_reference.json), including its exact source blob SHA.

Q5's registered rule was:

    F1/F8 < 0.05 predicts depth extrapolation with held-out accuracy >= 0.90.

It classified 9/12 arms correctly; eight arms extrapolated. The reported Spearman correlation between F1/F8 and held-out accuracy was -0.71786.

**F1/F8 is not the finite-horizon observability rank.** In Q5, F_t is the summed Frobenius norm ||X^T A||_F of module-level parameter-gradient fragility at loop t (where A backpropagates a readout margin through later loops). The SHAFT-COMP-12/13 quantity is the numerical rank of the recurrent-state observability stack built from a local Jacobian cocycle and a fixed behavioral probe. The two refer to different derivatives, intervention spaces, and endpoints.

The Q5 Git branch exposes experiment metrics, TCA output, receipts and checkpoint SHA-256 hashes. Its twelve actual model checkpoint files, expected under experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt/, are absent from the accessible Git tree. Thus **the new observability statistic has not been evaluated on the twelve Q5 models**, and no combined Q5 predictiveness claim is made.

An [opt-in checkpoint bridge](experiments/q5_checkpoint_bridge.py) has been added. It validates every local checkpoint against its TCA receipt hash before loading it. It builds a finite 16-direction **sketch** of the observability stack across all token positions of the recurrent looped transformer, with centered answer-position logits as the observable. The full recurrent state has many more dimensions than the sketch, so the result must be called a projected rank. The bridge has passed a local synthetic forward-mode Jacobian-vector-product smoke test; it has not been executed on original Q5 checkpoints. See the explicit full-state-versus-sketch distinction in [the preregistration](SHAFT_COMP_13_PROTOCOL.md).

## Registered GRU test

Four synthetic GRUs, each with 3,265 parameters and a 24-dimensional hidden state, are trained using the unchanged SHAFT-COMP-06 recipe. Seeds **101 and 202** form previously observed calibration runs. Seeds **303 and 404** are new held-out model seeds. The four-output probe learns counterfactual behavioral response bits from frozen GRU activations, without task IDs in the probe targets. The *base models* were explicitly trained on one of sixteen four-bit synthetic truth-table tasks.

For each model and starting recurrent depth (12 or 40), generate 16 goal IDs × 16 pairs of independently sampled natural input prefixes, for 256 paired hidden states. Both members of each pair were created using the same true goal. They receive:

1. A five-token diagnostic sequence sampled independently for **each** state, used to calculate its local GRU transition Jacobians and observability stack.
2. A separate 32-token continuation, shared by both states, used only to measure whether their **native model** four-query output signatures become different.

The frozen eligibility rule includes pairs only when their present native four-query signatures and four predicted probe bits agree. The resulting label is \(Y=1\) if native signatures disagree after the shared 32-step continuation.

Let the learned four-output monitor be \(W\in\mathbb R^{4\times24}\), and \(J_t\) its recurrent Jacobians along the diagnostic schedule. At horizon L=5,

    O_5 = [W; WJ_0; WJ_1J_0; ...; WJ_4...J_0].

With singular values \(\sigma_1\geq\dots\), define the predeclared effective rank \(r=\#\{\sigma_i>\sigma_1\cdot10^{-3}\}\), and the **primary pair risk score**

    S = 24 - 0.5(r_A+r_B).

The registered direction is that larger S predicts more future divergence. This direction was **not reversed or retuned after seeing results**. The primary metric is the pooled held-out ROC-AUC on seeds 303 and 404. Permutations shuffle outcomes **within** seed/depth cells; the reported two-sided p-value is centered on the empirical stratified-permutation null. Bootstrap intervals resample pairs within seed/depth, and do not represent between-model uncertainty.

Predeclared secondary scores are negative log minimum-to-maximum singular value ratio, negative current absolute probe margin, and initial hidden-state Euclidean distance. They are descriptive comparisons.

## Results

| Measurement | Calibration (seeds 101/202) | Held-out (303/404) |
|---|---:|---:|
| Total state pairs | 1,024 | 1,024 |
| Eligible after current-behavior/code matching | 819 | 827 |
| Future signature disagreements | 56 | 55 |
| Positive prevalence among eligible | 6.84% | 6.65% |
| Primary rank-deficit ROC-AUC | 0.461969 | **0.504439** |
| Weak-singular-value ROC-AUC | 0.461173 | 0.528026 |
| Current probe-margin ROC-AUC | 0.558510 | 0.546302 |
| Hidden-state-distance ROC-AUC | 0.571358 | 0.499976 |

Held-out inference:
- **ROC-AUC:** 0.504439 for the registered rank-deficit direction.
- **Within-model stratified bootstrap 95% interval:** [0.416862, 0.586862], 1000 valid bootstrap samples.
- **Seed/depth-stratified permutation test:** p=0.963037 with 1,000 permutations; null average AUC 0.503007.
- **Eligibility:** 827/1024 (80.8%) held-out state pairs, leaving 197 excluded as specified by the protocol.

### Per-model breakdown

| Held-out model | Initial depth | Eligible pairs | Future disagreements | Rank-deficit ROC-AUC |
|---|---:|---:|---:|---:|
| 303 | 12 | 256 | 18 | 0.522993 |
| 303 | 40 | 153 | 10 | 0.547552 |
| 404 | 12 | 256 | 17 | 0.539011 |
| 404 | 40 | 162 | 10 | 0.363158 |

The model/depth dependence exceeds the aggregate signal. Positive prevalence is low, and these proportions come from only two new model seeds. The confidence interval includes chance and moderate effects in both directions, so this study establishes a **failure of the registered small-model predictor test**, rather than universal absence of a predictive relationship.

### Why rank alone falls short

Numerical observability of a full tangent state is a local statement about *continuous monitor outputs along a particular diagnostic input schedule*. The label in this experiment concerns **discrete native output changes under a different future schedule**. A measured rank can be high while behavior remains identical; a weakly observed direction can matter under some inputs and be irrelevant under others. Singular-value rank also discards the alignment between low-sensitivity directions and the particular task's decision boundary.

The [DiagnosticSchedule.lean](lean/Shaft/DiagnosticSchedule.lean) counterexample makes the input-schedule problem exact. Two Boolean states are indistinguishable for eight inert diagnostic steps, but a single different input makes them observably distinct. Lean verifies the equality and the separating continuation. No model-general empirical inference is encoded in that proof.

This negative result follows the earlier SHAFT-COMP-11 finding that current behavioral equivalence is not an exact causal quotient. COMP-12's local rank-growth measurements remain valid. The new statistical test specifically rejects the adequacy of the *registered unweighted rank-deficit score* as a useful delayed-divergence predictor in this benchmark.

## Receipts, validation and reproduction

- [Executable GRU experiment](experiments/observability_prediction.py), which reconstructs all four trained models and probes from their fixed seeds, computes independent diagnostic and outcome trajectories, and performs the fixed permutation/bootstrap analysis.
- [Frozen summary](results/comp13_summary.json), including per-model AUCs, held-out positive/negative counts by exact rank-deficit bin, model and probe SHA-256 hashes, and the SHA-256 of full per-pair results.
- [Pure-standard-library AUC re-audit](rank_auc_reference.py) and [unit tests](tests/test_rank_auc_reference.py), which reconstruct the registered primary AUC from the binned counts without training or PyTorch.
- [Read-only original Q5 benchmark](results/q5_registered_reference.json) and [checkpoint bridge](experiments/q5_checkpoint_bridge.py).

The full 2,048-row local receipt's SHA-256 is

    2d522e9df3422bf423186a224939338e9aef52069c3d0356e8a69b401c562eaf

The full experiment was executed **twice** on the local CPU; the generated JSON files matched byte for byte. PyTorch 2.10.0+cpu, NumPy 2.3.5, and deterministic torch operations were used. Complete raw pair-level outputs are provided as a conversation ZIP artifact, while the summary, test, and training recipe are committed in GitHub. GPU or alternative PyTorch builds may change low-order numerical values.

From the repository root:

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.rank_auc_reference
python -m research.shaft_compactification.experiments.observability_prediction \
  --per-goal 16 --permutations 1000 --bootstraps 1000 \
  --out /tmp/shaft_comp13_full.json
```

When **all** original Q5 checkpoint files and accompanying TCA JSON receipts are present locally:

```bash
python -m research.shaft_compactification.experiments.q5_checkpoint_bridge \
  --looped-repo /path/to/LoopedTransformers --preflight
python -m research.shaft_compactification.experiments.q5_checkpoint_bridge \
  --looped-repo /path/to/LoopedTransformers --device cuda \
  --items 4 --sketch 16 --out /tmp/q5_checkpoint_observability.json
```

This Q5 interface is an experimental opt-in protocol, not a completed Q5 checkpoint run. It should be inspected for accelerated-attention forward-AD compatibility on the target CUDA version before costly launches.

## Next mathematical issue

A monitor's horizon and effective tangent rank are incomplete descriptors of the **task-relevant causal quotient**. A follow-up hypothesis could weight the observability Gramian by a task reward/output sensitivity operator and evaluate multiple independent future input schedules. Such an experiment needs a new registered predictor and a fresh holdout seed family. Results from COMP-13 may guide the design, while its negative holdout should remain unchanged.
