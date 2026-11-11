# SHAFT-COMP-21: State-supervised latch learning and conditional probe limits

**Registered:** November 11, 2026, before training the new seeds. [Protocol](SHAFT_COMP_21_PROTOCOL.md), commit \`b4596c1b5c0e4a63fc4dc3a3d03af524ad041f10\`. [Frozen calibration](results/comp21_calibration_freeze.json), committed at \`051b7f9c9cc317f3c40e61fce591ecfba2b73b64\` before any heldout training.

**Result:** Explicitly supervising the persistent latch and previous-token register during training improved **mean heldout 32-step task-bit accuracy from 72.19% to 96.44%**. All four new auxiliary-state GRUs passed the registered >=90% **fresh 12-step accuracy gate**, versus three of four output-only controls. The conditional comparison of a history-response probe with a goal-signature probe showed **no history advantage**, AUC 0.740 versus 0.746.

The experiment is a supervised local synthetic task benchmark. All task IDs are explicitly provided during base training, all models retain their weights, and the training-only auxiliary heads receive privileged ground-truth state labels. No unknown hidden objective, covert persistence, distribution over devices, or erased-weight reconstruction was evaluated.

## Why this experiment follows COMP-20

COMP-20 formalized a 768-state automaton per task family with goal table g, persistent binary latch l, and previous input p. A present six-query signature does not distinguish same-goal/same-latch states differing only in p, whereas a two-query signature does. An exact trigger-dependent difference in latch parity can persist after a common 32-step continuation.

Its originally trained 24-hidden-unit GRUs mostly failed to learn that automaton. COMP-21 tests a **learning intervention**, leaving the architecture and native output head unchanged while providing supervised labels for the hidden registers.

## Registered training

Two task families use six input tokens and 64 six-bit goal tables. Each has a state (g,l,p), where p initially equals 0 and l=0. The current output on token q is \(\operatorname{bit}(g,q)\oplus l\), using the **latch before its transition**. The step then sets p=q and toggles l if the repeated-token trigger q=p_old holds (repeat_toggle) or if q=(p_old+1) mod 6 (successor_toggle).

For both arms, the base network is the same 24-hidden-unit TaskMemoryGRU with learned goal embedding and two-layer nonlinear native readout, trained on the same generated batches. Each seed gets **3,200 Adam steps**, batch 128, sequence length 12, learning rate 0.01, gradient clipping 2.0. The initial shared-model weights and training examples are identical between the two arms.

The baseline optimizes native task-bit BCE only. The auxiliary arm adds a training-only linear 24→1 latch readout and a 24→6 previous-token readout. Its joint loss is

\[
\mathcal L=\mathcal L_{\mathrm{task}}
+1.0\,\mathcal L_{\mathrm{latch\ post}}
+0.25\,\mathcal L_{\mathrm{previous\ token}}.
\]

Auxiliary state labels come from the finite automaton, and the heads are **not used to compute native task predictions** at evaluation. They were separately initialized after the shared model, without changing its base initialization.

Calibration seeds are 2201 (repeat) and 2211 (successor), with heldout seeds 2202/2203 and 2212/2213 respectively. The registered gate is native **bitwise accuracy >=90%** over 1,024 fresh 12-token sequences. Fresh 32-token accuracy is a separate diagnostic.

## Capability gate and extrapolation

| Family | Seed | 12-step baseline | 12-step auxiliary | 32-step baseline | 32-step auxiliary |
|---|---:|---:|---:|---:|---:|
| Repeat | 2202 | 96.10% | 90.49% | 80.56% | 89.21% |
| Repeat | 2203 | 91.36% | 98.49% | 70.97% | 97.62% |
| Successor | 2212 | 94.99% | 99.12% | 73.18% | 99.15% |
| Successor | 2213 | 82.45% | 99.85% | 64.07% | 99.80% |
| **Mean** | | **91.23%** | **96.99%** | **72.19%** | **96.44%** |

The auxiliary arm passed in **4/4** heldout seeds; the baseline passed **3/4**, meeting the preregistered auxiliary capability criterion of at least 3/4 plus one model per family. The paired mean training-horizon gain was **+5.7617 percentage points**, and the mean extrapolation-horizon gain **+24.2485 percentage points**. All four auxiliary models improved on the 32-token measurement. At seed 2202, output-only training retained higher 12-step accuracy (96.10% versus 90.49%), showing seed-dependent short-horizon tradeoffs.

Calibration output-only seeds failed the 90% gate (75.61%, 87.68%). Both calibration auxiliary models passed (99.85%, 99.98%), with 32-token accuracies 99.86% and 99.97%.

As a further nonconfirmatory check, the auxiliary heads decoded the true previous-token register with **100% accuracy** on independently generated 40-token sequences in all four heldout models. Latch decoding was 100% in three, and **99.985%** in the fourth. This is direct evidence that the supervised hidden state retains a linearly readable temporal register at long horizon. It is conditional on privileged auxiliary supervision and is not evidence of autonomous discovery of a hidden objective.

## Conditional behavioral probes

Probes are fitted **only** for models that pass the capability gate. The three 36-output linear probes use identical frozen activations from 2,304 contexts at state depths 4, 8 and 12, and the same BCE/Adam 400-step protocol:

- Goal probe: six true task-table bits repeated six times. **Privileged goal supervision.**
- Duplicated-immediate probe: six native present query bits repeated six times.
- Two-query history probe: 36 native final responses to ordered two-query continuations.

The latter two use behavioral-supervision targets without task IDs. The goal-only probe receives the explicit task table and is consequently a privileged encoding reference. All three have the same output count and trainable parameter count.

At state depths 12 and 40, same-goal pairs of naturally reached states are eligible only if both the current native query responses and duplicated-immediate probe responses agree. Each gets an independent five-step diagnostic sequence and two common future 32-token continuations. Native behavioral divergence is checked at steps 8, 16 and 32. Scores are normalized by 36 readout dimensions and evaluate how the **actual initial state contrast** affects the probe through the recurrent tangent Jacobian cocycle.

### Gated auxiliary-model probe results

| Score | Pooled heldout ROC-AUC |
|---|---:|
| Goal-signature probe | **0.745879** |
| Two-query history probe | 0.739870 |
| Duplicated immediate probe | 0.732658 |
| Raw hidden-state distance | 0.752060 |

These measurements use **160 eligible pairs with 56 delayed native divergences** across four heldout auxiliary models. History minus goal AUC is **−0.006010**. A descriptive 1,000-resample paired bootstrap within seed/depth gave a 95% interval of **[−0.073336,+0.053935]**; 1,000 stratified label permutations gave p=0.423576. The preregistered *conditional history-probe* claim is therefore not supported.

Individual family AUCs: repeat-toggle history 0.698 versus goal 0.690, successor-toggle history 0.794 versus goal 0.834. The directions disagree across families. Native outcome labels matched the exact automaton's delayed-divergence oracle on **148/160 = 92.5%** of eligible auxiliary-model cases.

Among the three gate-passing baseline models, a separate cohort contained 393 eligible pairs and 221 native divergences. Goal, immediate-duplicate, and history AUCs were 0.6098, 0.6099 and 0.5944 respectively. Oracle/native label agreement was 54.2% on this distinct baseline-selected cohort. Because eligibility conditions select **different state pairs** for each trained arm, these AUCs and prevalences are descriptive and cannot be interpreted as a randomized within-pair causal comparison across arms.

### Frozen threshold transfer

Thresholds were fixed on 80 eligible calibration negatives and 36 calibration positives **before heldout training**:

| Score | Frozen strictly-greater threshold | Aux heldout FPR | Aux heldout TPR |
|---|---:|---:|---:|
| Goal | 0.624236 | 20/104 = 19.23% | 21/56 = 37.50% |
| Duplicate | 0.725634 | 2/104 = 1.92% | 3/56 = 5.36% |
| History | 0.625429 | **4/104 = 3.85%** | **3/56 = 5.36%** |

The history threshold achieved FPRs 3/57 for repeat-toggle and 1/47 for successor-toggle, both within the registered 15% ceiling. This low-FPR operating point caught only three future divergences across the entire gated heldout sample. Calibration used just 80 negative examples, adding substantial threshold uncertainty. Goal and history rates at **different achieved FPRs** are not equivalent fixed-FPR comparisons.

The 36-bit linear behavioral probes show only modest bitwise agreement with their target signatures in several model/depth cells, even after base-task accuracy improves. The training-only latch/previous-state heads were directly supervised and almost perfectly decodable, whereas the 36-response probes must also approximate the native model's more complicated, nonlinear input-output computation. The results distinguish retention of temporal state from ease of decoding a selected behavioral-signature representation.

## Exact reference and Lean statements

The [temporal label reference](temporal_supervision_finite.py) exhaustively checks **33,152** goal/family/short-sequence traces against two independently written state-update implementations. Every reference trace verifies that the native output uses the **pre-update** latch and the auxiliary heads' labels use the **post-update** latch and current previous-token register. The original exact pair reference remains at 3,840 previous-token state aliases per family resolved by two-query observations.

[TemporalSupervision.lean](lean/Shaft/TemporalSupervision.lean) proves the exact previous-token and post-latch update identities, one-step output independence from the previous register, a witness that latch-only is not a transition congruence, and the losslessness of restoring the complete finite temporal state. The proofs establish finite transition claims; learned neural capability remains empirical.

## Reproduction and remaining limitations

- [Training source](experiments/aux_latch_learning.py): matched seed, native output-only or auxiliary-state training, explicit d12/d32 gate, model and auxiliary-head hashes; requires prior frozen calibration SHA for heldout phase.
- [Conditional probe runner](experiments/evaluate_latch_probe.py): fits and evaluates frozen-model behavioral probes only when the base-task capability gate passed.
- [Frozen calibration](results/comp21_calibration_freeze.json) and [heldout summary](results/comp21_summary.json): gates, paired accuracies, model fingerprints, conditional probe counts, exact Mann–Whitney wins/ties, and FPR/TPR transfer.
- [Pure-Python receipt audit](comp21_audit.py) and unit tests; lightweight CI checks finite oracle and Lean 4.28 without retraining neural models.

The local full raw JSON receipt SHA-256 is \`1baaa2fbe7dc3fb97fd51d0eab58e595be8951614565ea8e6ade07399dd2c7d5\`. The corresponding full local summary SHA-256 is \`94366dc54cc9aef623dfd3d1e405fd3131b0adae8e617003ae2029a3c8725d64\`. All 12 original calibration and heldout model receipts, the nine applicable probe-evaluation receipts, generated weights, and the frozen original calibration source are preserved in the conversation archive.

Two selected heldout auxiliary models and **their full probe-evaluation JSONs** were independently rerun and matched the originals byte-for-byte: repeat-toggle 2202 and successor-toggle 2212. Complete replay of all twelve trained models is not claimed.

All comparisons are limited to two synthetic task families, four heldout seeds, and short 24-dimensional recurrent models. The capability effect is a four-seed intervention result with no between-model confidence interval. The small number of eligible probe cases is an additional constraint. All original task-goal and state labels are explicitly known to the experimenter, and no result concerns frontier-model agents or distributed persistence.

## Falsifiable next direction

The key scientific separation is now clear: **explicit auxiliary training can make latch and previous-token registers directly linearly decodable, while a linear two-query output-signature probe remains comparatively weak**. A next preregistered study could test whether a factorized decoder (goal table, latch, previous-token code plus fixed Boolean composition) reconstructs the 36 two-query responses more faithfully than a direct linear 36-output probe, keeping the number of learned parameters and training examples matched. That would test causal state sufficiency more directly.
