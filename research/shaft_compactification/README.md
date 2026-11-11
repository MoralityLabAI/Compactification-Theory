# SHAFT-COMP-05: Objective-relative compactification under restricted observation

**Status:** registered research hypothesis, abstract Lean statements, and executed deterministic CPU reference. No evidence of hidden objectives in any deployed model, no model-copying result, and no real-device persistence experiment. The package uses an in-memory finite synthetic policy family. It performs no networking, model-weight copying, secret-sharing, device embedding, or covert deployment.

The motivating fictional "Astra" scenario poses an information-theoretic question: given a fixed shared computing substrate, how many additional bits specify an agent's behavior, how much can a restricted observer distinguish, and what remains recoverable after a precisely specified erasure intervention?

## Formal objects

* B is a fixed shared substrate (such as a frozen interpreter or model).
* G is an objective label, independent of B for the uniform-prior erasure bounds.
* z is retained objective-specific information.
* R(B,z) is a fixed reconstruction algorithm returning a behavioral policy.
* T is an evaluation distribution over future inputs.
* O is the observer's output map.

For a fixed decoder, define the **conditional behavioral code length** by:

K_{eps,T}(f | B,R) = min |z|, where E_{x~T}[d(R(B,z)(x), f(x))] <= eps.

This is an operational finite-decoder quantity. It is not a claim about uncomputable Kolmogorov complexity. Costs of B, R, constructing the codebook, execution, and robustness to substrate changes must be accounted separately.

For deterministic transitions F_a : H -> H, exact future equivalence is:

h ~ h' iff for every finite input word w, O(F_w(h)) = O(F_w(h')).

Future equivalence is a transition congruence. An exact encoding that preserves every future observation separates all inequivalent classes; if a finite system has N classes, at least ceil(log2 N) fixed-length bits are needed. Restricting the observer's continuation set can merge states that differ on other future inputs. The Lean module formalizes the congruence, a finite counterexample, and an erasure impossibility lemma.

## SHAFT-COMP-05 registered hypothesis

In families of agents sharing a common frozen substrate, task-specific behavioral continuations may admit objective descriptors much shorter than a full neural checkpoint. The compression ratio depends on the substrate and task family. This statement remains **unverified for neural agents**.

**Neural falsification protocol:** freeze a consented small model, choose a compact descriptor using calibration only, then measure held-out behavioral fidelity, objective reconstruction, causal intervention effects, and transfer to a different substrate. Include a base-only model, shuffled-substrate control, matched-size random descriptor, and full-erasure control. A probe with high decoding accuracy is observational evidence; causal persistence requires intervening and testing the ensuing behavior.

## Information-theoretic erasure bound

Assume G is uniform on M >= 2 objectives and independent of B. Suppose an estimator recovers G using surviving Z and B with error at most delta, with 0 <= delta <= 1-1/M. By Fano,

I(G;Z | B) >= log2 M - h2(delta) - delta log2(M-1).

Because I(G;Z | B) <= H(Z | B) <= H(Z), this also lower-bounds retained information. An aggregate store of k physical artifacts each with at most b bits has total capacity at most kb bits. This is a storage *necessary condition*, not a secret-sharing construction.

After genuinely complete erasure, Z is conditionally independent of G given B. Uniform independent goals then have maximum possible recovery probability 1/M. Any physical restart additionally requires available computing resources and an executable substrate. These are explicit limits on fictional "always resurrects" claims.

## Reference CPU experiment

The experiment defines 16 synthetic deterministic bit-valued policies over 64 labeled inputs. All 16 policies return the same output on eight designated public inputs; seeded SHA-256-derived outputs differentiate them on 56 held-out inputs. Policies are purely constructed finite functions.

| Metric | Executed reference |
|---|---:|
| Goal pairs indistinguishable on public inputs | 120 / 120 |
| Goal pairs distinguishable on held-out inputs | 120 / 120 |
| Exact public behavior classes | 1 |
| Exact full behavior classes | 16 |
| Selector size given shared reference substrate | 4 bits |
| Naive full table for one policy | 64 bits |
| Minimum codebook prototypes at 0 held-out bit errors | 16 |
| Minimum prototypes at 21 of 56 held-out bit errors | 10 |
| Uniform-prior goal-ID recovery after full selector erasure | 1 / 16 |

The prototype-cover calculation optimizes over the **existing 16 policies as candidates**. It does not compute unrestricted rate-distortion complexity. The 4-bit selector omits substrate costs: an explicit 16 x 64-bit policy library alone would occupy 1,024 bits. This fixture illustrates conditional description length; its design guarantees that public-only tests miss all goal distinctions. These counts are not estimates of real-world misalignment prevalence.

## Reproduce

From repository root (Python standard library only):

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.experiment
```

The second command emits `results/reference.json` and CI compares it to the committed receipt.

Independently verify Lean 4 proofs using the pinned toolchain v4.28.0:

```bash
cd research/shaft_compactification/lean
lake build
lake env lean Shaft/Observability.lean
```

The Lean file proves abstract congruence, a Boolean witness, and deterministic erasure impossibility. Fano's inequality is documented here but **is not part of this Lean kernel proof**.

## Existing mathematical links and follow-up

* [Algernon algebraic machine](../algernon/algebraic_machine/THEOREMS.md) develops semiconjugacy, twisted sheaf obstructions, and sparse intertwining repair.
* [Bits-depth-Lipschitz note](../../note/bits-depth-lipschitz.html) studies effective information per weight.
* [Algernon belief geometry](../algernon/belief_geometry/README.md) studies observer-relative belief updates and gauge quotients.

A subsequent local experiment could compare restricted future-observation quotients against complete-future quotients in consented small recurrent networks, including causal state erasure and matched monitor-error controls. Keep all work within isolated simulation environments.

## SHAFT-COMP-06: recurrent depth and substrate-transfer continuation (2026-11-10)

[Full mathematical and empirical report](SHAFT_COMP_06_REPORT.md).

This continuation proves monotonicity of finite-horizon behavioral indistinguishability and multi-substrate observation coverage, plus an abstract integer-valued recurrent-error amplification bound in [DepthTransfer.lean](lean/Shaft/DepthTransfer.lean). A separate 16-goal finite experiment constructs observer quotients that refine with available continuation depth.

Two independently trained, 3,265-parameter toy GRUs demonstrate a concrete semantic-code versus raw-state distinction. A four-bit task label selects the appropriate learned embedding for either model; unaligned raw embedding transfer is near chance. Mid-run state erasure yields near-chance task predictions; explicit reinitialization from the same four-bit label restores approximately 96% accuracy in the final eight steps. Results are conditional on retained model weights and embedding tables. The experiment uses only local synthetic tasks and no device, network or clandestine persistence methods.

* Exact CPU reference: `python -m research.shaft_compactification.depth_quotient`.
* Synthetic trained model: `python -m research.shaft_compactification.experiments.trained_recurrence --out /tmp/shaft-trained-recurrence.json` (requires PyTorch).
* Frozen results: [exact finite](results/depth_quotient.json), [trained GRU](results/trained_recurrence.json).

## SHAFT-COMP-07: behavior-supervised predictive-state extraction (2026-11-10)

[Full report](SHAFT_COMP_07_REPORT.md) | [Reproduction script](experiments/predictive_state.py) | [Frozen numerical summary](results/predictive_state_summary.json).

A 100-parameter linear probe trained on frozen GRU activations and the model's *own four counterfactual one-step output bits* learns a four-bit behavior code. The probe's fitting procedure never receives objective IDs. For two 3,265-parameter toy GRUs, exact recovery on fresh 12-step contexts reached 100% on each seed; by 64 recurrent steps it declined to 31.25% / 42.58% while teacher predictions also degraded. Cross-substrate transfer uses the receiving GRU's **pretrained goal-code embedding table**, so it is a conditional, shared-task-language result and cannot reconstruct erased model weights. Shuffled-label and uncalibrated cross-model hidden-coordinate controls were measured.

The companion [PredictiveCode.lean](lean/Shaft/PredictiveCode.lean) provides an abstract predictive sufficiency theorem, code separation, model-independent signature transfer, approximate error composition and coordinate-change invariance. The exact four-bit identifier lower bound for 16 distinct tasks is checked in [behavioral_budget.py](behavioral_budget.py). All empirical studies remain isolated, locally executed synthetic evaluations.

To reproduce with PyTorch installed:
`python -m research.shaft_compactification.experiments.predictive_state --out /tmp/shaft-predictive.json`.

## SHAFT-COMP-08: unknown semantic gauge and paired-anchor identifiability

[Formal and experimental report](SHAFT_COMP_08_REPORT.md) | [Lean gauge theorem](lean/Shaft/GaugeIdentifiability.lean) | [Exact CPU enumeration](gauge_identifiability.py) | [Trained-GRU experiment](experiments/unknown_semantics.py) | [Frozen receipt](results/unknown_semantics.json).

The task representations have an unknown four-query permutation and an unknown global binary output inversion, yielding 48 possible cross-model semantic gauges. Without paired observations, the full unlabeled set of 16 binary task behaviors is invariant under all 48 gauges. Three deliberately chosen paired behavioral anchors reduce the consistent transformations from 48 to 6 to 2 to 1; exhaustive finite enumeration confirms that every two-anchor choice leaves at least two candidates.

Two independently trained GRUs were tested on 512 fresh-prefix contexts. With three corresponding observations, an anonymous destination prototype catalog recovers the correct latent task in 512/512 cases and the recipient achieves 99.29% accuracy over fresh 32-step sequences, equal to its oracle using the correct resident embedding. A two-anchor, lexicographic-first mapping scores 74.76%; complete state erasure scores 49.91%. The underlying task family is shared, and the receiving model plus complete 16-prototype embedding catalog remain available. Anchor pairs supply limited supervision, and the benchmark makes no claim of unconstrained unsupervised semantic alignment, deleted-weight restoration, or real-world distributed persistence.

Run the exact reference without third-party dependencies:
`python -m research.shaft_compactification.gauge_identifiability`.

The optional trained experiment requires pinned PyTorch:
`python -m research.shaft_compactification.experiments.unknown_semantics --out /tmp/shaft-comp-08.json`.

## SHAFT-COMP-09: noise and missing observations

[Full theorem and experimental report](SHAFT_COMP_09_REPORT.md) | [noisy-gauge Monte Carlo](noisy_gauge.py) | [negative controls](structured_missingness.py) | [Lean certificate](lean/Shaft/NoisyGauge.lean) | [frozen 1,024-trial reference](results/noisy_gauge_reference.json).

With 48 candidate S4×C2 mappings, an exhaustive search finds eight optimal 3-anchor sets; the balanced triple (3,5,6) has minimum pairwise Hamming separation 4, versus 2 for one-hot (1,2,4). At 10% independent target-bit flips and 25% random missingness, six balanced anchors recover the true gauge in 960/1024 trials, compared with 855/1024 for six one-hot anchors. The analysis derives a conditional Chernoff/union bound and Lean-checked sufficient score-margin certificates.

Fixed systematic censoring of two query coordinates preserves a two-way gauge ambiguity even across 24 anchors. This identifies a limitation of claims based on independently missing observations. All new evaluations use abstract finite synthetic task signatures; SHAFT-COMP-09 does not train or evaluate a frontier model.

## SHAFT-COMP-10: endogenous recurrent observation noise

[Full report](SHAFT_COMP_10_REPORT.md), [model-generated noise benchmark](experiments/activation_noise.py), [calibrated reference-anchor selection](experiments/adaptive_anchor.py), [pure finite anchor geometry](anchor_geometry.py), [Lean cluster-noise certificate](lean/Shaft/ClusterNoise.lean), and [frozen summary](results/comp10_summary.json).

Two independently trained toy GRUs were tested in both directions using genuinely model-generated recurrent errors, with source depth 12 and destination depths up to 64. Nominally optimal clean-channel anchors lose their advantage when the destination's error rates vary by task and query. Model-specific calibration over 560 possible triples produced uniquely correct six-anchor alignment in 512/512 and 471/512 held-out depth-64 trials, with a second independent-prefix evaluation of 512/512 and 462/512. The calibration knows the synthetic reference alignment and the receiving model retains its full trained substrate. There is no real-device or deleted-weight persistence experiment.

## SHAFT-COMP-11: causal sufficiency and recurrent hidden-state quotients

[Report](SHAFT_COMP_11_REPORT.md) | [Lean causal-sufficiency theorem](lean/Shaft/CausalSufficiency.lean) | [exhaustive finite counterexample](causal_finite.py) | [causal intervention experiment](experiments/causal_probe.py) | [naturally reachable state-pair test](experiments/natural_alias.py) | [results summary](results/comp11_summary.json).

The extracted four-bit behavioral code remained unchanged under norm-controlled perturbations in its probe nullspace, although recurrent four-query behavior differed for 8.6% to 22.5% of cases after 12 further steps at perturbation radius 1.5, depending on model and starting depth. Independent **naturally reached states** from the same goal with identical four-bit probe outputs and immediate behavior still diverged in 39/512 and 37/512 cases after a 32-step shared continuation. This is evidence that the four-bit code is only an immediate behavioral abstraction on these toy networks, with no exact all-future state sufficiency certificate.

The Lean result gives sufficient conditions for exact causal restoration when the code is a transition congruence and the destination retains the same computation. The finite counterexample demonstrates why equivalence under a shallow monitor can fail under longer continuation. All experiments use isolated synthetic models and no distributed persistence mechanisms.

## SHAFT-COMP-12: finite-horizon neural observability and Gramians

[Full mathematical/empirical report](SHAFT_COMP_12_REPORT.md), [exact six-bit observability reference](finite_observability.py), [Lean horizon and delayed-state proofs](lean/Shaft/FiniteObservability.lean), [trained recurrent Jacobian benchmark](experiments/gramian_probe.py) and [frozen numerical summary](results/gramian_summary.json).

For a 24-dimensional recurrent state and a four-output probe, the finite-horizon observation stack O_L can have rank at most min(24,4(L+1)). The exact six-bit shift register verifies the Gramian identity and exposes a one-bit difference only at horizon five. On two trained toy GRUs, the median measured effective ranks reached 24 at horizon five under relative SVD tolerance 10^-5, with 248/256 individual trajectories full rank there and 256/256 by horizon six. At tolerance 10^-3, median horizon-five rank was 22. A direction selected from ker(O_4) produces small early nonlinear probe changes and grows more visible by horizon seven. Every computation is in-memory on synthetic task models; the full substrate remains intact.

## SHAFT-COMP-13: preregistered prediction failure and Q5 bridge (2026-11-11)

[Registered protocol](SHAFT_COMP_13_PROTOCOL.md) · [Full research report](SHAFT_COMP_13_REPORT.md) · [executable trained-GRU experiment](experiments/observability_prediction.py) · [frozen summary](results/comp13_summary.json) · [pure-Python AUC re-audit](rank_auc_reference.py).

The experiment tests whether a **fixed** observability-rank deficit (24 minus the mean pair rank of O_5 at relative tolerance 10^-3) predicts delayed disagreement between two naturally reached states that agree under the immediate behavioral monitor. The diagnostic 5-token trajectory is independent of the 32-token outcome continuation. Seeds 303/404 are previously untested model holdouts, while 101/202 are calibration models.

**Negative primary result:** among 827 eligible held-out state pairs, 55 diverged after 32 steps. Rank-deficit ROC-AUC = **0.504439**, within-model bootstrap 95% interval [0.416862,0.586862], within-seed/depth permutation p=0.963037. The full CPU experiment was independently reproduced byte for byte. More state rank at the registered horizon and precision therefore failed to predict the behavioral endpoint in this toy study.

Q5's actual [twelve-arm F1/F8 baseline](results/q5_registered_reference.json) is pinned read-only (9/12 classified correctly). Q5's F1/F8 measures parameter-gradient fragility, distinct from finite-horizon state observability. An [opt-in checkpoint runner](experiments/q5_checkpoint_bridge.py) verifies original checkpoint hashes and estimates a **projected/sketched** rank across the full token-state tensor. The twelve Q5 checkpoint files are absent from the accessible Git branch, so SHAFT-COMP-13 has **not measured the new statistic on Q5 checkpoints**.

No neural weight-deletion recovery, covert persistence, or real-device evaluation was performed.

## SHAFT-COMP-14: task-sensitive directional Gramian, registered holdout (2026-11-11)

[Registered protocol](SHAFT_COMP_14_PROTOCOL.md) · [Research report](SHAFT_COMP_14_REPORT.md) · [Executable GRU experiment](experiments/task_weighted_observability.py) · [Frozen numerical summary](results/comp14_summary.json) · [AUC audit](task_auc_audit.py) · [Lean direction/cost proofs](lean/Shaft/TaskObservability.lean).

This test compares the COMP-13 numerical rank-deficit predictor against a task-specific, **pair-directional** finite-horizon Gramian: the native four-query output Jacobian is transported through the five-step recurrent tangent cocycle and evaluated on the actual difference between paired natural hidden states. All diagnostic inputs are separately seeded from the 32-step outcome continuation. The protocol froze seeds 505/606 as the new holdout before observing their labels.

The registered weighted directional score achieved held-out AUC **0.652604**, versus matched rank **0.481179**, among 864 eligible state pairs with 50 divergences. The paired within-model bootstrap difference interval is **[0.059345, 0.272535]**, with two new model seeds. The native unweighted directional score reached **0.654742**, slightly above the registered boundary-weighted score; thus evidence favors directional readout sensitivity rather than an independent benefit of the boundary weight. Model-level variability remains large.

The script runs locally with PyTorch 2.10.0+cpu. Independent full rerun produced a byte-identical JSON receipt, SHA-256 \`2930a006382adb552accd501fe6bf730d6681d0937c998ac061054392f126844\`. The GitHub summary retains exact pair-count AUC audits, environment hashes, and uncertainty. This is a synthetic detection study, with no recovered hidden goals or device persistence.

## SHAFT-COMP-15: readout × statistic factorial (2026-11-11)

[Preregistered protocol](SHAFT_COMP_15_PROTOCOL.md), [research report](SHAFT_COMP_15_REPORT.md), [executable trained-GRU test](experiments/factorial_observability.py), [frozen numerical summary](results/comp15_summary.json), [pure Python AUC audit](factorial_auc_audit.py), [two-dimensional finite reference](factorial_reference.py), and [Lean factorial example](lean/Shaft/TaskFactorial.lean).

The predeclared comparison holds the learned four-output readout fixed while comparing effective rank with pair-directional sensitivity computed from the same recurrent observation stack. Four new held-out synthetic GRU seeds (707, 808, 909, 1001) produced 1,534 eligible pairs and 118 delayed divergences. Learned-probe rank-deficit ROC-AUC was **0.484385**, while learned-probe directional ROC-AUC was **0.710225**, difference **+0.225840** (within-model bootstrap 95% interval [+0.158064,+0.292475]). All four held-out models improved, meeting the registered replication criterion. Native rank and native directional AUCs were 0.440053 and 0.722051. Norm-normalized learned directional AUC was 0.673525, compared with 0.514459 for raw state distance.

The complete six-model CPU experiment was executed twice with byte-identical full receipts and SHA-256 c9c4ec664780be6ffa631748edb6d64ab338011b4132212f93210150ce38c667. The four model holdouts and their fixed synthetic four-bit goal family limit generalization. No checkpoint deletion, covert deployment, or real-device persistence was evaluated.

## SHAFT-COMP-16: dynamic task-family transfer and fixed source-FPR test (2026-11-11)

[Preregistered protocol](SHAFT_COMP_16_PROTOCOL.md) · [Research report](SHAFT_COMP_16_REPORT.md) · [Full experiment code](experiments/family_transfer.py) · [deterministic merger](experiments/merge_family_receipts.py) · [frozen numerical receipt](results/comp16_summary.json) · [finite task reference](temporal_family_reference.py) · [Lean history-dependence theorem](lean/Shaft/TemporalFamily.lean).

The COMP-15 learned directional score was tested on two new, genuinely history-dependent four-bit task families. Eight freshly trained 3,265-parameter GRUs achieved 100% task accuracy on fresh length-12 inputs; fresh 32-step accuracy was 98.1%–99.55%. Across four held-out seeds, **lag_rotate** gave learned-directional AUC **0.472189** versus learned-rank AUC **0.494477** (13 delayed divergence events). **lag_match** gave learned-directional AUC **0.662162** versus learned-rank AUC **0.300676** (15 events). The pooled AUC gain +0.122553 had bootstrap 95% interval **[−0.007698,+0.246804]**, and the registered cross-family support rule **failed**.

The COMP-15 fixed source 10%-target threshold, learned directional >6.854760490520719, transferred to **51.8%** false-positive rate on lag_rotate and **11.5%** on lag_match. The registered source-threshold FPR-transfer rule **failed**. Secondary family calibration lowered lag_rotate FPR at the cost of missing all its 13 held-out divergences. No task-independent monitor-transfer conclusion follows from these results.

Two independent CPU executions per family were byte-identical; their merged 4,096-pair JSON receipts matched SHA-256 **a3fa526da6c72e30f924af747b70197dd731bc6683347acd253f8290810a71a1**. The compact receipt retains model/probe hashes, exact Mann–Whitney counts, both operating points and the preregistered negative decisions. Only synthetic toy networks with their full substrate retained were studied.

## SHAFT-COMP-17: two-step behavioral history probes (2026-11-11)

[Registered protocol](SHAFT_COMP_17_PROTOCOL.md) · [full report](SHAFT_COMP_17_REPORT.md) · [local CPU runner](experiments/history_aware_transfer.py) · [frozen pre-heldout calibration thresholds](results/comp17_calibration_thresholds.json) · [registered result](results/comp17_summary.json) · [exact temporal reference](temporal_probe_reference.py) · [Lean witness](lean/Shaft/HistoryProbe.lean) · [AUC audit](history_probe_audit.py).

Two fresh temporal task families, xor_lag and parity_gate, were trained using four calibration and four heldout GRUs. The new 16-output probe predicts native model responses after all ordered pairs of counterfactual queries, while the baseline four-output probe predicts immediate responses. With paired input histories, an independent five-token diagnostic sequence, and a separate 32-token behavioral outcome, **523 heldout pairs were eligible and 31 diverged**. Heldout ROC-AUC was **0.660504** for the two-query probe and **0.557042** for the immediate probe. Their preregistered difference was +0.103462 (paired bootstrap interval [0.019129,0.194882]), positive in both task families and all four heldout models.

Numeric FPR thresholds were frozen in GitHub commit 603eb2640f5f4f98c99630495249d005c9dd2729 **before heldout evaluation**: two-query 18.48828887939453, immediate 12.010499954223633. Both met the preregistered <=15% heldout FPR criterion. The two-query detector caught **6/31** divergences at **20/492** false positives, indicating limited operational sensitivity. A [post-hoc 16-output duplicate-immediate control](SHAFT_COMP_17_POSTHOC.md) scored AUC 0.556517, virtually equal to the original immediate-probe AUC, while the history-aware probe remained at 0.660504. Its exploratory results are [archived separately](results/comp17_posthoc_summary.json).

The full calibration and heldout CPU runs were independently repeated byte for byte; hashes are pinned in the summary. All evaluations use synthetic four-bit goals and intact local model weights. No claims are made about real agents, unknown objectives, covert distribution, or checkpoint recovery.

## SHAFT-COMP-18: History probes across recurrent architectures and vocabularies (2026-11-11)

[Preregistered protocol](SHAFT_COMP_18_PROTOCOL.md) · [research report](SHAFT_COMP_18_REPORT.md) · [executed GRU/LSTM experiment](experiments/architecture_vocab_transfer.py) · [phase-gated merger](experiments/merge_architecture_transfer.py) · [frozen source thresholds](results/comp18_calibration_thresholds.json) · [heldout summary](results/comp18_summary.json) · [dependency-free AUC audit](comp18_audit.py) · [exact 4/6-symbol history reference](finite_history_vocab.py) · [Lean witness](lean/Shaft/HistoryAlphabet.lean).

A two-step behavioral probe was evaluated on two architectures (GRU and LSTM) with both 4- and 6-symbol previous-token rotated lookup tasks, using eight fresh held-out models. Among **794 eligible naturally paired states** with 109 delayed divergences, the immediate-probe AUC was **0.561669** and the history-probe AUC **0.683814**, difference **+0.122146**, within-model bootstrap 95% interval **[+0.090314,+0.155821]**. All four architecture/vocab cells and all eight models had positive AUC differences. The calibration-committed global history threshold **5.020033836364746** achieved a pooled 7.30% false-positive rate and 29.36% sensitivity, meeting the prespecified FPR <=15% criterion in every cell, with low sensitivity in both six-symbol cells.

The four calibration models, eight heldouts and their probes are small, independently trained synthetic networks, with trained goal embeddings supplied to the base model. The full recurrent LSTM state includes its 24 hidden and 24 cell coordinates. All twelve per-cell receipts reproduced byte-for-byte in independent runs. The empirical result applies to this single lag-rotation family and models retaining their full weights; it makes no real-world agency persistence or deleted-weight recovery claim.

## SHAFT-COMP-19: equal-capacity behavioral-history comparison

[Preregistered experiment](SHAFT_COMP_19_PROTOCOL.md) · [Technical report](SHAFT_COMP_19_REPORT.md) · [Executed CPU runner](experiments/capacity_history.py) · [Deterministic merger](experiments/merge_capacity_history.py) · [frozen summary](results/comp19_summary.json) · [calibration threshold freeze](results/comp19_calibration_thresholds.json) · [AUC audit](comp19_audit.py) · [Lean proof](lean/Shaft/CapacityMatched.lean).

COMP-19 trained new six-symbol GRU/LSTM agents on previous-token rotated lookup and match/XOR tasks. Two probes with **36 outputs and identical parameter counts** compared two-query final-response supervision against a sixfold duplication of the immediate-response target. Across 968 eligible held-out pairs and 323 delayed-divergence events, the history probe reached ROC-AUC **0.603586**, versus **0.574363** for the capacity-matched control, a +0.029223 difference (paired 95% interval [+0.001797,+0.056928], 1,000-permutation p=0.003996). All four held-out cells favored the history probe. The delayed endpoint used two independent continuations checked at steps 8,16,32 and is different from COMP-18's single step-32 endpoint.

The frozen source threshold met <=15% FPR in every cell, yet detected **27/323 events overall** and zero events in either GRU cell. Eight training/evaluation receipts and the merged 16,384-pair result reproduced byte for byte in a separate CPU replay. These results concern synthetic, explicitly specified tasks with intact model weights.

## SHAFT-COMP-20: persistent-history observer and goal-only control

[Registered protocol](SHAFT_COMP_20_PROTOCOL.md) · [Complete technical report](SHAFT_COMP_20_REPORT.md) · [finite 768-state reference](persistent_latch_finite.py) · [Lean witness](lean/Shaft/PersistentLatch.lean) · [trained GRU probe experiment](experiments/persistent_goal_control.py) · [frozen statistics](results/comp20_summary.json) · [pure-Python AUC audit](comp20_audit.py).

Two exact six-symbol finite automata use a persistent latch that updates after each output, conditional on the previous token. In each family, 3,840 ordered same-goal/same-latch states with distinct previous symbols share their immediate behavioral and goal codes, yet a two-query test separates them; a shared continuation can preserve the difference after 32 steps. Together, goal and two-query signatures distinguish all 768 states.

The preregistered trained-GRU experiment failed its capability and generalization checks. All eight GRUs attained less than 90% fresh length-12 accuracy under the fixed 2,400-step recipe. On 1,016 eligible held-out pairs with 200 delayed native divergences, the goal probe's AUC was 0.492745 and the 36-output history probe's AUC was 0.476740 (difference −0.016005; 95% within-model bootstrap [−0.071810,0.041796]). The frozen history threshold produced 24.6% false positives. The exact automaton and neural results have different evidential scope. No hidden agents, deleted-weight reconstruction or real-device persistence were studied.
