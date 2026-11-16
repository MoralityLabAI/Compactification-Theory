# NOR-24: train-only sequential coordinate auditing under structured hidden-future attacks

**Protocol frozen before the new data** in commit 780c822, with complete design at [REGISTRATION.md](REGISTRATION.md). Continues merged NOR-23 and the companion kernel-proved [MathCan NOR-24 PR #113](https://github.com/patrickdugan/MathCan/pull/113).

## What is being tested

NOR-23 proved and experimentally illustrated that an external honest exact-coefficient witness detects a hidden output alteration if a sampled coordinate intersects the changed set. NOR-24 asks whether public source structure and a train-only learned prior improve *where* a limited witness budget is spent. It retains the same exact sixteen-candidate RRC25 grammar and the existing NOR-21 streamed public verifier. The witness still supplies genuinely target-dependent information; all full-target-soundness claims remain scoped to those assumptions.

Four disclosed synthetic attack-location supports are defined entirely from public candidate output and header: the last 96 coefficients, the 96 largest absolute forecast coefficients, the 96 positions from delay+spacing, and all output coordinates as a uniform background. The policy learns the **four mixing weights**, not the attack templates themselves. Its fitted weights from 768 TRAIN labels were: tail 0.38212435, magnitude 0.31865285, lag 0.19818653, uniform 0.10103627.

## First prospective hosted outcome

[Original successful GitHub Actions run #43870686549](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43870686549) used **24 genuinely distinct public sources**, split into 12 TRAIN and 12 TEST, at observation degrees 512/1024 with three source families per degree per replicate. Query output lengths were 769 and 1537. All source-only fixtures excluded generating programs, hidden true outputs and target-derived witness values. Every public streamed mask was checked against the original native 16-template interpreter and every in-family full prediction matched native sealed truth.

There were exactly **1,024 challenge ordinals per TEST source and policy**, with k=4,16,64 nested audit budgets; r=1,4,16 altered-coordinate worlds; integer changes +1 and +4294049777; blind supports fixed before the challenge. A separate informed attacker observes the inspection set before changing output coordinates. Five policies are run on matched source/trial worlds: uniform, static learned prior, sequential Bayesian no-hit posterior, shuffled static ranking, and privileged support-informed oracle. The randomized controls share exact audit budgets.

The registered primary comparison was k=16,r=4,δ=1, with fixed family evaluation weights tail .4, magnitude .3, lag .2, uniform .1.

| Policy | Mean weighted detection / source |
| --- | ---: |
| Uniform | 6.145833% |
| Shuffled learned-risk ranking | 6.197103% |
| Static learned-risk ranking | 34.867350% |
| **Sequential posterior** | **34.969889%** |
| Privileged support oracle | 47.443034% |

**PASS:** posterior exceeded uniform on **11/12 TEST sources**, shuffled on **11/12**, and had a **+0.2882405599 absolute** source-mean weighted gain over uniform. Registered thresholds were 9/12 and +0.10. One degree-1024 source produced a negative paired posterior-minus-uniform difference (-0.0121). The **increment over static risk ranking was only +0.001025391**, or about **0.10 percentage points**. The result supports source-risk localization under the specified templates, with little evidence of incremental value from after-check Bayesian updates.

### Required negative controls

At the same k=16,r=4,δ=1, detections by attack family were:

| Attack location | Uniform | Static prior | Posterior | Support oracle |
| --- | ---: | ---: | ---: | ---: |
| Tail window | 5.92% | 51.52% | 52.29% | 52.22% |
| Largest forecast coefficients | 6.38% | 37.83% | 37.13% | 51.76% |
| Delay+spacing window | 6.27% | 13.53% | 13.56% | 52.13% |
| Uniform background | 6.07% | 2.03% | 2.03% | 6.01% |
| **Off-template** | **6.47%** | **0.00%** | **0.00%** | **7.93%** |

The off-template stressor alters only positions outside all three structured regions and is **excluded from the preregistered mixture**. Its zero posterior detection is a concrete assumption-break counterexample. For the informed adversary, all tests under k≤64 permit r≤16 unchecked positions inside each family support, yielding **0% detections** across every policy. Exact witness decisions were invariant to changing δ from 1 to 4,294,049,777.

## Original source-locked evidence

First source-only fixture: **4,300,923 bytes**, SHA-256:

    88285e2400566c633d211071ace59784987d6458d743db478321dc90f33ef001

Original complete first-run JSON: **737,700 bytes**, SHA-256:

    ed54eabd70637a42c34caf66d04bcadd50f890a1ced46cea0132f8767b476807

The exact byte streams are committed in [results/nor24_original_hosted.zip](results/nor24_original_hosted.zip), with the [frozen receipt](results/nor24_first_frozen_receipt.json). The separate independent replayer reconstructs all 24 original exact programs and hypothetical output masks, all **768** supervised family episodes, and **12,288** heldout 64-query clean-path schedules. It independently checks **2,764,800** paired TEST policy/family/radius/budget decisions using an implementation separate from the new evaluator. All three original CI jobs passed, including the separate-process audit.

The study is **conditional on the synthetic attack template family and the assumed authenticity of an exogenous exact target witness**. The policy's seed is deterministic for evidence replay and does not claim cryptographic unpredictability. Multiple challenge draws and counterfactual worlds share only 12 held-out source identities. A source-level transfer or adversarial guarantee in a real Lean theorem prover is outside these measurements.

The original execution separately recorded aggregate schedule generation cost (approximately 2.1–2.2 seconds per 1024 planned schedules at degree 512 and 3.9–4.0 seconds at degree 1024, including all policy arms). The preregistered separate CPU timing of witness comparisons was **not captured as its own field**. CPU comparisons between audit policies should not be inferred from this original receipt. The correctness and detection criteria were independently replayed and are reported without substituting a performance estimate.

## Reproduction

    python -m unittest research.nor24_adaptive_audits.test_experiment -v
    python research/nor24_adaptive_audits/experiment.py --make-fixture /tmp/nor24_fixture.json
    python research/nor24_adaptive_audits/experiment.py --check-parity --fixture /tmp/nor24_fixture.json --out /tmp/nor24_full.json
    python research/nor24_adaptive_audits/receipt_audit.py --check --fixture /tmp/nor24_fixture.json --full /tmp/nor24_full.json

The permanent read-only archive audit also checks both first-run byte hashes and replays the fixed program and decision grid.
