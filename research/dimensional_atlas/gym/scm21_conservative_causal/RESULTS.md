# SCM-21 results: causal consistency and clean preservation

Preregistered commit `f26b858`. Promoted causal checkpoint: True; lambda 0.25, common update budget 1000. The registered robust positive criterion did not pass.

Clean accuracy met the two-point final guard in all six seed/split comparisons; the worst change versus warm was -0.5 percentage points.

## Clean top-one mathematics

All entries use one neural argmax and independent exact post-prediction grading, without search or verification to choose a program. Warm means the frozen SCM-19 adapter; control uses the identical training objective and three-forward computation with causal coefficient zero. Each model sees the same 1024 ID / 1024 OOD fresh clean cases. Seeds share these cases, so their scores are not independent case samples.

| Seed | Split | Warm | Matched control | Causal | Causal change | Clean preserved |
|---:|:---|---:|---:|---:|---:|:---|
| 191 | ID | 674/1024 (65.8%) | 678/1024 (66.2%) | 676/1024 (66.0%) | +0.2 pp | yes |
| 191 | OOD | 440/1024 (43.0%) | 446/1024 (43.6%) | 438/1024 (42.8%) | -0.2 pp | yes |
| 1191 | ID | 688/1024 (67.2%) | 690/1024 (67.4%) | 690/1024 (67.4%) | +0.2 pp | yes |
| 1191 | OOD | 440/1024 (43.0%) | 435/1024 (42.5%) | 435/1024 (42.5%) | -0.5 pp | yes |
| 2191 | ID | 667/1024 (65.1%) | 668/1024 (65.2%) | 665/1024 (64.9%) | -0.2 pp | yes |
| 2191 | OOD | 460/1024 (44.9%) | 460/1024 (44.9%) | 456/1024 (44.5%) | -0.4 pp | yes |

![Fresh clean accuracy](figures/clean_accuracy.png)

## Development-only checkpoint choice

All six registered causal candidates are shown. A tie at zero mediation advantage is resolved by lower lambda and earlier step; selection is not confirmation of a mediation effect. The warm development baseline is fixed before updates. Admissibility requires a one-point development clean buffer and at least 64 common-eligible pairs per split. The lambda-zero pilot runs to 4,000 updates and provides the checkpoint at the selected budget; its selection-evaluation cost differs from the causal pilots.

| Lambda | Step | ID clean | OOD clean | Selection gate | ID native / random | OOD native / random |
|---:|---:|---:|---:|:---|---:|---:|
| 0.25 | 1000 | 65.3% | 43.9% | pass | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.25 | 2000 | 65.1% | 43.9% | pass | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.25 | 4000 | 64.8% | 44.3% | pass | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.5 | 1000 | 64.8% | 44.5% | pass | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.5 | 2000 | 64.1% | 43.7% | pass | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.5 | 4000 | 63.8% | 43.8% | pass | 0.0% / 0.0% | 0.0% / 0.0% |

## Primary causal-consistency confirmation

The first program operator is cycled while inputs, topology and the other operators stay fixed. Generation never uses model correctness. Eligibility requires recipient and donor to be exact-clean-correct under warm, control and causal models. The first 128 common-eligible pairs per split are assayed; all compared models use identical pair IDs. Own eligibility, common counts, selected predictions and outcomes are preserved in raw receipts.

| Seed | Split | Common / attempted | Assayed | Warm native | Control native | Causal native | Causal random | Positive |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 191 | ID | 186/512 | 128 | 1/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |
| 191 | OOD | 187/1024 | 128 | 0/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |
| 1191 | ID | 184/512 | 128 | 0/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |
| 1191 | OOD | 192/1024 | 128 | 0/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |
| 2191 | ID | 193/512 | 128 | 0/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |
| 2191 | OOD | 192/1024 | 128 | 0/128 | 0/128 | 0/128 (0.0%) | 0/128 (0.0%) | no |

A robust positive requires all seeds and both splits: ≥64 common-eligible pairs, ≥90% native-write fidelity, ≥15 percentage points over each of norm-matched random, warm native and matched-control native, and ≤2 percentage-point fresh clean loss versus warm. Selection must have promoted a candidate. Undefined fidelity at zero tests is not zero. Secondary directions/visits cannot rescue the primary endpoint. Wilson intervals are descriptive and unadjusted; no pooled independent-seed inference is made.

![Primary mediation](figures/primary_mediation.png)

![Methods across visits](figures/visit_mediation.png)

Every intervention acts only on CLS. Native writes lie in the same learned rank-four global head-fiber write span; random fiber patches match their norm separately for every pair/model/visit. Full CLS at the last visit is the donor-readout positive control. Sham and terminal fiber interventions are null controls. All controls are checked by independent exact replay.

## Training and resource accounting

![Separate loss components](figures/training_losses.png)

A sampled causal-loss value of zero can mean that no training pair in that batch passed the fixed warm-start eligibility mask. It does not imply perfect causal prediction. Per-update eligible counts are in the journal.

Only 1,552 adapter parameters per model were eligible for updates; the 231,120-parameter model, SAE and projector architecture were fixed. Protected frozen parameter hashes match the SCM-19 warm starts. All pilots, including unused updates/checkpoints, count toward 16,000 adapter updates, 74,448 model calls, 403,584 example forwards, 1,614,336 recurrent example-visits and 43,587,072 recurrent token-visits. Each training update runs three forward/backward paths; development-selection evaluation costs are separately included, so end-to-end compute is not claimed identical between arms.

Local RTX 3050 4GB, FP32, batches ≤8, four visits. Peak allocated/reserved 24.49/30.00 MiB; minimum sampled free 3245.7 MiB; active worker time 47.47 minutes. Enforced 3000 MiB allocation cap and 500 MiB free-headroom guard. No package installation, cloud compute or official EvalMath roster access.

Eleven unit tests pass. Independent audit regraded 138,240 recorded predictions, checked 720 mediation cells and 5,120 unique fresh signatures, verified protected weights/checkpoints/directions, common eligibility and pair identity, matched random norms, freeze chronology and cumulative journal/evaluator totals. Complete pure-source/cache counters and continuous journals are available. Worker totals above exclude this separate CPU replay; its own oracle/cache counts are in `audit.json`. No training followed final freeze. Weights, optimizer/RNG states, data cache, raw gold and journal remain private outside Git.

## Scope and limitations

This is a finite formal-series DSL and an explicitly supervised interchange task. Improved fidelity, if observed, demonstrates the tested intervention function on a clean-eligible subset; it does not establish a unique internal semantic variable, natural-language mathematical reasoning, or a topology/continuum certificate. The shared SAE and core were already trained in SCM-19. Development-selected loss/checkpoint results are not independent confirmation.

SCM-21 balances accepted cases across every family/topology cell and shuffles each pool deterministically. OOD clean and pair pools cover all 16 combinations equally. Exact coefficient-encoder inputs, as well as public signatures, are disjoint from the registered historical catalog and every earlier SCM-21 role. This changes coverage and freshness relative to SCM-20, so cross-study score changes do not isolate the new loss/optimizer settings. Within this study all compared models share cases. Algebraic and structural task relationships remain possible.

The independent audit rebuilt the historical/current pre-final encoder catalog and verified 5,120 mutually unique fresh encoder inputs with zero pre-final overlaps. It also checked every logged optimizer increment: maximum actual L2 0.00999952, registered cap 0.01, capped updates 5,371. The hard cap bounds one step, not total displacement, hidden-state changes or clean accuracy.

![Parameter update sizes](figures/parameter_steps.png)
