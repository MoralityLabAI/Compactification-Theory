# SHAFT-COMP-07: learned predictive-state code from recurrent activations

**Evidence:** two executed, seeded CPU GRU studies plus finite counting identity and Lean abstract sufficiency theorems. All observations concern synthetic four-query truth-table tasks. There is no real-device persistence, autonomous model resurrection, exfiltration, or frontier-model measurement.

## Registered question

Can a compact *behavioral signature* be recovered from the hidden state of a recurrent model, without providing the probe with explicit task identifiers, and used across independently trained models that share the same task language?

A frozen teacher RNN has hidden state h at time t. For each query q in {0,1,2,3}, compute the teacher's counterfactual one-step output O(F_q(h)). The four binary answers form the **behavioral target** sig(h) in {0,1}^4. The probe receives only h and learns to predict sig(h). Objective labels g, known to the synthetic data generator, enter the final audit alone; they never supervise the probe. This is supervised *behavioral self-distillation*, not unsupervised discovery.

The four output bits are a coordinate system shared by both model instances because their original tasks were deliberately matched. The probe has 24 x 4 + 4 = 100 trainable parameters. Its four-bit *output* is conditional on the retained 100-param probe, the target decoder, and the two underlying 3,265-param GRUs.

## Exact mathematics

Let b_B(h,q) be the behavior of substrate B. A code c_B(h) and decoder D are predictively sufficient on a query set Q when

  for every q in Q, D(c_B(h),q) = b_B(h,q).

If two states have different responses to some q, any exact sufficient code must separate them. If source and target codes agree and each is decoded correctly by the same D, their observed Q-responses also agree. In the approximate metric setting, independent decoder errors bounded by eps_S and eps_T yield cross-substrate behavior distance bounded by eps_S + eps_T via the triangle inequality.

The new Lean file [PredictiveCode.lean](lean/Shaft/PredictiveCode.lean) formalizes these conditional implications as well as invariance under observation-preserving coordinate changes. The code does not prove that any trained probe is exact.

For 16 uniformly likely distinct complete signatures, any k-bit deterministic identifier can correctly identify at most min(2^k,16) goals. At k=0,1,2,3,4 the upper bounds are 1/16,2/16,4/16,8/16,16/16. The bound requires a codebook/decoder paid for as part of the shared substrate. The Python finite-counting reference verifies it independently.

## Executed method

Two GRUs, seeds 101 and 202, were trained using the pre-existing [trained_recurrence.py](experiments/trained_recurrence.py) protocol, with truth-table goal g supplied to the *base model* through its original embedding at time zero. Goal g is one of 16 possible four-bit truth tables. The frozen RNN gets fresh prefixes of depths 4,8,12. Each model's own four counterfactual one-step outputs supply probe targets.

Training uses 48 independent contexts per goal per depth, totaling 2,304 examples per GRU. A separate calibration set contains 12 contexts per goal per depth. The linear 24-to-4 probe is trained with BCE and 400 Adam steps at learning rate 0.04. The fresh evaluation set has 32 contexts for each of 16 goals at depths 4,8,12,24,40,64, giving 512 examples per depth per model.

Controls include a separately fitted probe with shuffled behavior signatures, a source-trained probe applied directly to the other model's unaligned hidden representation, receiver initialization from a wrong goal, zeroed receiver state, and the correct-goal oracle. Every operation is in-memory and CPU only. Python and PyTorch versions, seeds, and methods are recorded in the script; exact PyTorch machine reproducibility can vary by platform.

## Results

| Probe measurement | Seed 101 | Seed 202 |
|---|---:|---:|
| Exact true-goal code on independent depth-12 prefixes | 100.0% | 100.0% |
| Exact code on depth-24 prefixes | 81.64% | 93.16% |
| Exact code on depth-40 prefixes | 56.84% | 67.19% |
| Exact code on depth-64 prefixes | 31.25% | 42.58% |
| Probe single-bit accuracy at depth 64 | 77.05% | 80.86% |
| Teacher's own single-bit accuracy at depth 64 | 79.54% | 77.29% |
| Shuffled-label probe exact code at depth 12 | 1.37% | 3.91% |

The teacher's own output behavior deteriorates with excessive recurrent depth, contributing to the code's apparent failure. The probe matches teacher responses on 93.21% / 88.43% of depth-64 query bits while agreeing less often with the synthetic task's original truth table. A code extracted from an inaccurate teacher cannot be treated as a guarantee of the original goal.

A probe trained on model 101 and applied *uncalibrated* to model 202's depth-12 hidden coordinates obtains 8.40% exact goal accuracy; the reverse yields 16.80%. A model-specific representation gauge remains substantial.

### Shared task-language transfer

Taking a source depth-12 hidden state, the trained probe emits four bits. These bits are interpreted as the original task's truth-table ID by the other model's existing embedding table. The receiver completes a fresh 32-step sequence.

| Transfer direction | Inferred-code accuracy | Correct-goal oracle | Wrong-goal control | Zero-state control |
|---|---:|---:|---:|---:|
| 101 → 202 | 99.13% | 99.13% | 53.25% | 50.65% |
| 202 → 101 | 98.84% | 98.84% | 52.98% | 49.77% |

The model and its trained goal-to-embedding table survive intact; the four-bit output selects a state from that table. These results do not establish recovery of erased model weights or of a previously unknown behavior family.

## Reproducibility

From the repository root with PyTorch 2.10 CPU installed:

```bash
python -m research.shaft_compactification.experiments.predictive_state \
  --out /tmp/shaft-predictive-state.json
python -m unittest discover -s research/shaft_compactification/tests -v
```

A compact frozen result lives in [results/predictive_state_summary.json](results/predictive_state_summary.json), including the SHA-256 of the full local JSON receipt. The complete local receipt can also be regenerated from the script. Lightweight CI checks source syntax, exact combinatorics, and receipt consistency; it intentionally avoids retraining PyTorch models on every PR.

## Next falsifiers

Probe held-out task *families* rather than only unseen token sequences; change the shared query language between substrates, and distinguish information about an objective from information about generic response statistics. Investigate learned latent gauge alignment and causal patching in pinned, isolated small recurrent networks. Compare to permuted behavioral targets and monitor-observation restrictions before claiming any stealth property.
