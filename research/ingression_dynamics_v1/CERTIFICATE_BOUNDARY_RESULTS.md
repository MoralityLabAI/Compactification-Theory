# QO-15: Frozen boundary-distillation experiment, completed

**November 4, 2026.** The frozen evaluation source was committed *before* writing or running the QO-15 trainer at \`c6ac3a1\`. The [reproducible receipt](../../results/certificate_boundary/receipt.json) stores the trained weights, source SHA-256 map, complete validation history, all 288 rollout checkpoints, and every one of the 96 trajectory hashes.

## Training source and learned architecture

This is a controlled 138-parameter tanh-RNN replication/augmentation of QO-14's exact two-action horizon-two Bellman student, using the same features, exact rational teacher, original squared-value loss, full BPTT, 65-epoch cap, and inference cost of 240 multiply–accumulates. Only the training evidence distribution changes.

| Source or outcome | QO-15 observed count |
|---|---:|
| Original usable QO-14 training examples | 689 |
| Unique counterfactual candidates evaluated for public frontier ranking | 5,760 |
| Frontier candidates selected for exact labels | 768 |
| Of those, informative unequal-value action pairs | 42 |
| Of those, at least one positive exact action value | 74 |
| Cell-swap training examples | 768 |
| Informative oversampling copies | 2,016 |
| Final training examples after resampling | 4,241 |
| Independent validation frontier examples | 192, including 4 informative |
| Validation-selected epoch | 57 |
| Learner weights | 138 |

One- or two-step branch successors were constructed from legitimate QO-07 ledgers generated only from the original train and validation source families. Candidate selection used the distance of the public confidence bounds to the tolerance. Their exact Q values were calculated **after** that selection. No frozen test rates, seeds or labels entered augmentation, training or epoch selection. QO-07's confidence certificate remained the only certificate authority.

## Independently frozen supervised test

The [frozen evaluation generator](certificate_boundary_test_contract.py) used eight new rate pairs, three seeds per rate, three fixed sampling schedules and ten checkpoints. It produced 720 unresolved examples, all with equal exact action values.

| Frozen-test metric | Original QO-14 RNN | Boundary-enriched QO-15 RNN |
|---|---:|---:|
| Action-value MAE | 0.00597587 | 0.00099294 |
| Predicted action-margin MAE | 0.00260466 | 0.00195449 |
| Informative teacher-choice states | 0/720 | 0/720 |
| Conditional teacher-action regret | undefined | undefined |
| Mean exact-teacher action regret | 0 | 0 |
| False engagement decisions (predicted max Q ≥ 0.05, exact max Q below) | 8 | 1 |

The recurrent model learned to output small Q values more accurately on this distribution, but **the primary decision-ranking test contained zero informative events**. There is no measured gain in choosing between probes, and the reduction in Q-error should not be described as decision-competence evidence.

## Frozen matched policy evaluation

Exactly 8 rate pairs × 3 new policy seeds × 4 registered policies × 80 queries = **96 full trajectories, 7,680 Bernoulli queries and 288 checkpoint records**. All policies were evaluated using the same per-cell Bernoulli stream prefix for each case/seed.

| Arm | Correct certificates at 80 queries | Independent teacher lookups | Student inference MACs | Model-chosen actions |
|---|---:|---:|---:|---:|
| QO-08 bottleneck | 0/24 | 0 | 0 | 0 |
| Frozen QO-14 RNN | 0/24 | 0 | 348,720 | 27 |
| Boundary-enriched QO-15 RNN | 0/24 | 0 | 348,720 | 68 |
| Exact QO-13 H2 teacher | 0/24 | 1,453 | 0 | 2 |

At the 32- and 56-query checkpoints every policy likewise had zero certificates. All 288 checkpoint records had zero wrong certificates. There was **one confidence-coverage miss per policy**, for the shared \`qo15_frozen_pair_0\` / seed \`1721\` case. These correspond to a single shared case-seed history family, not four statistically independent episodes. The observed miss does not invalidate the e-process under its stated finite-sample coverage guarantee; it must be preserved and reported.

## Interpretation and next objective

The enriched training generator succeeded in creating informative examples under the original exact Bellman definition, while the frozen rollout and supervised test distributions exposed an objective/dataset mismatch. The two-probe *probability of outright certification* was too sparse to distinguish actions in every one of the frozen supervised examples, and even the exact teacher earned no certificates in the 80-query trials. A trained student's low test Q-error chiefly measured its ability to reproduce negligible predicted certification probabilities.

The appropriate next registered experiment, QO-16, is a **dense proof-progress target**. Use QO-11/QO-12's exact expected decrease in the public robust certification gap as the Bellman reward, with the same QO-07 anytime certificate and independent frozen rollout. Preserve a risk- and compute-aware fallback, and compare a learned gap-progress recurrent controller against its exact teacher and bottleneck. The target should vary across undecided states where the binary near-term certificate indicator remains zero.

These are findings about a digital two-state one-action experiment. No uniform approximation guarantee, neural foundation-model transfer, hidden-state system-identification result, or biological claim is established.
