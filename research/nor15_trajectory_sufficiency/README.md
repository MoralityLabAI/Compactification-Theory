# NOR-15A: early-state observability and reconstructibility in frozen RRC-25

Registered 15 November 2026. Preceded by merged [NOR-14 #161](https://github.com/MoralityLabAI/Compactification-Theory/pull/161) and archival evidence [#162](https://github.com/MoralityLabAI/Compactification-Theory/pull/162).

## Question and mathematical scope

The 1000-parameter RRC-25 GRU computes nine recurrent states H1,...,H9 from nine public 8-dimensional feature rows. NOR-14's H4 basin feature modestly outscored matched early-state nonlinear controls without statistical support for a basin-specific advantage. NOR-15A measures whether H4 is reconstructible from H9 by a bounded decoder and how locally the frozen transition H4→H9 contracts, **holding each example's last five input rows fixed**.

Because the last five input rows vary across examples, even pointwise injectivity of each fixed-suffix transition need not imply that H9 determines H4 globally. Because each real-valued H9 is near-continuous, a finite table of distinct endpoints cannot prove positive conditional mutual information I(H4;Y|H9). We preregister quantitative reconstruction errors, singular spectra and near-endpoint witnesses as descriptive structural diagnostics; any conditional-information claim requires a separate statistical test with assumptions and confirmation.

## Fixed experimental design

- Source checkpoint SHA-256: \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`. Source nine-step GRU parity to original logits required within 1e-11. Source programs and withheld future query outputs play no role in reconstruction.
- Original 96 TRAIN / 36 DEV public-feature rosters reproduced through NOR-12 receipts. H9 and H4 standardized from TRAIN only.
- Fit a 12-output linear ridge reconstruction \`H9→H4\` at regularizers 0.001, 0.01, 0.1, 1, 10; fit a 24-feature frozen random-tanh expansion (37 inputs counting 12 original and bias), with the same ridge grid. Negative-label control permutes training H4 targets across source inputs; mean and direct H9-copy baselines carry zero reconstruction parameters.
- Choose regularizer per arm on the 36 original DEV public examples by minimizing standardized H4 MSE, then freeze. No test or future proof labels enter fitting, including early-state target construction.
- TEST comprises four fresh source cohorts of 72 examples, 12 from each of six families. Seed \`corpus.SEED+92715000+rep*1000003+family_index*10007\` is locked. This excludes all model-visible nonlate feature tensors and visible source examples from NOR-11/12/12B/13R/14.
- **Primary**: 240 distinct model-visible nonlate inputs; each reconstruction scored via the dimension-averaged squared error in TRAIN-normalized H4. **Secondary**: 48 source-distinct late_source cases with a shared endpoint input hash. These may not be treated as 48 independent model representations.
- Report MSE by arm and source family, exact case-level paired signs against predeclared controls, and the endpoint-nearest-neighbor H4 distances. These pairs establish finite-scale separation, not exact state collisions.
- For the first four samples per source family of the first independent test cohort, calculate the 12x12 central-difference Jacobian of the H4→H9 suffix map at epsilon 1e-5, keeping the last five model inputs fixed. Estimate all 12 singular values from symmetric Gram diagonalisation, logging near-zero values and condition estimates. Numerical eigenvalues are FP64 approximations and carry no interval certificate.

## Next inference

NOR-15B should test whether a 12-dimensional H4 residual after **training-only H9 reconstruction** improves native future-proof decisions over the same parameter and compute budgets. It must compare H4 residuals against reconstructed-H4 residuals, shuffled H4 assignments, and capacity-matched nonlinear H9 controls. More than 1,000 unseen nonlate source inputs should be preregistered, with independent evaluation groups and the same four exact native template checks.

The NOR-15A source audits are not evidence of task-relevant conditional information by themselves. An observed lower reconstruction loss merely supports a finite-decoder statement. The formal companion MathCan theorem bounds reconstruction errors on pairs of early states with nearby late states under an explicit Lipschitz constant.

## Reproduce

    python -m unittest research.nor15_trajectory_sufficiency.test_audit -v
    python research/nor15_trajectory_sufficiency/audit.py --check \
      --repetitions 4 --per-family 12 --out /tmp/nor15a_full.json

All serialized hidden-state records and source hashes remain in the hosted experiment artifact; future proof outcomes are deliberately absent. Full JSON must be archived with SHA-256 and a source-independent audit before merge. No NOR-15A result is claimed at registration time.


## First executed result (source-locked)

[First green NOR-15A Actions run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721123693) completed the 96/36 public-source reconstruction protocol and four newly seeded held-out cohorts. The complete 492,261-byte JSON is SHA-256-pinned to \`9dbe31960a1866b8f4062cb148ab5db9c4b7743ea1fa28689b8377a78459f92b\` in [the immutable first-result receipt](results/nor15a_frozen_receipt.json); [nor15a_receipt.py](nor15a_receipt.py) independently checks 1,440 per-case error values, all cohort/family arithmetic, the three exact paired signs, and 24 local spectral records.

| H9→H4 reconstruction | Primary normalized MSE /240 distinct inputs | DEV-selected ridge penalty |
| --- | ---: | ---: |
| Predict mean H4 | 0.919838 | n/a |
| Copy endpoint H9 | 1.039225 | n/a |
| Linear ridge | 0.320055 | 0.01 |
| Nonlinear frozen 24-tanh-feature ridge | **0.284538** | 0.001 |
| Shuffled-train linear | 0.940068 | 10 |

Nonlinear reconstruction improves on the linear fit in 177 primary inputs and loses in 63; unadjusted exact paired sign p≈1.02×10^-13. Linear versus shuffled succeeds on 211 inputs and loses on 29, p≈2.70×10^-35. These tests support **recoverable nonlinear structure in H9** for predicting H4 under the fixed finite decoder family.

The improvement varies by source family. The nonlinear model performs worse on the held-out rational family (about 0.975 MSE versus 0.909 for linear), indicating heterogeneous extrapolation across the generator's source types. All 48 late_source cases share one model feature hash and were excluded from the primary means and sign tests.

Across 24 fixed-suffix finite-difference Jacobians, minimum estimated singular values ranged from 6.13×10^-10 to 4.97×10^-6, with median approximately 2.97×10^-8. The condition-estimate median was approximately 2.79×10^7. Near-input pairs in the primary set had a median H4/H9 distance ratio about 2.34. These are source-dependent, floating-point local diagnostics; the result establishes neither certified global noninjectivity nor positive conditional mutual information.

The next registered task outcome experiment is [NOR-15B](NOR15B.md). Its new source inputs, parameter budget and decision criteria were fixed before reading the task outcomes.


### Source-controlled numerical evidence

The full original NOR-15A 288-case JSON is now preserved as [nor15a_full.zip](results/nor15a_full.zip), retaining the 492,261-byte source stream with SHA-256 `9dbe31960a1866b8f4062cb148ab5db9c4b7743ea1fa28689b8377a78459f92b`. The independent reproducer checks 1,440 reconstruction errors, 24 Jacobian spectra, feature-group uniqueness and all paired signs:

```sh
python research/nor15_trajectory_sufficiency/nor15a_receipt.py --check --archive research/nor15_trajectory_sufficiency/results/nor15a_full.zip
```
