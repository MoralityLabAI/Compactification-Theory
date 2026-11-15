# NOR-15B: native proof-template value of trajectory innovation

**Protocol frozen 15 November 2026, before reading NOR-15B held-out results.** Paired with NOR-15A [audit.py](audit.py) and the exact MathCan NOR-15 Lipschitz reconstruction theorem.

NOR-15A found 0.2845 standardized H4 reconstruction MSE under the public-only nonlinear 37-feature ridge predictor on 240 disjoint nonlate cases, versus 0.3201 for linear ridge and 0.9198 for the mean. This was a finite-decoder result. This continuation tests whether the part of H4 not recovered by an H9-only decoder can change native proof-template correctness.

## Frozen source, mechanism and controls

RRC25 original 1000-parameter GRU source SHA-256:
\`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`.

The recurrent weights and original sixteen-class head remain frozen. On the original 96 public training and 36 development source cases, reconstruct the NOR-15A public-only nonlinear H9→H4 ridge fit using the original preset 24 random tanh features, the original ridge grid, and DEV-only MSE selection. Let \(Z_9\), \(Z_4\) be train-standardized 12-dimensional hidden vectors, and \(\widehat Z_4(Z_9)\) the fixed public trajectory estimator.

Fit a 400-slot, 16-class residual correction to the original native ranker with input \((Z_9, U, 1)\), where U is one of:

| Readout | Twelve added coordinates | Why |
| --- | --- | --- |
| residual_early | \(Z_4-\widehat Z_4(Z_9)\) | Newly exposed trajectory innovation under finite decoder |
| raw_early | \(Z_4\) | Full trajectory reference |
| reconstructed_early | \(\widehat Z_4(Z_9)\) | Capacity-matched H9-only nonlinear decoding |
| endpoint_nonlinear | Twelve frozen seeded \(\tanh(W Z_9+b)\) features | H9-only nonlinear capacity control |
| shuffled_early | \(Z_{4,\pi}-\widehat Z_4(Z_9)\), group-preserving seeded permutation | Independence-oriented trajectory input negative control |
| hidden_only | Twelve identically zero coordinates | Linear H9-only reference |

All six heads allocate 16×25=400 parameters. The hidden-only arm has 208 active coefficients. Only the 96 training source cases and 16 exact **public observed-degree-192 template checks** per case supervise task fits; the 1,536 distinct public checks are shared across arms, with 1,536 label exposures per arm. Full-batch Adam, 150 epochs, step size .023, ridge penalty .075. No private source program or withheld degree-256 future coefficients enter the training objective. The 36 previously frozen development cases choose each arm's mixing amplitude from 0.25, 0.5, 1, 2, and a separately reported global policy including the original scorer. All inference policies retain up to four exact paid candidate-template checks per case.

## Newly seeded heldout protocol

\`corpus.SEED + 92815000 + replicate * 1000003 + family_index * 10007\`.

Each of four independently seeded cohorts has 250 distinct nonlate source inputs: 50 examples in each of signed, sparse, product, rational and squared families. The new test has **1,000 distinct model-visible nonlate feature tensors** and no overlap with previous NOR11/12/12B/13R/14/15A train, development or tested nonlate feature hashes or with their visible public source cases. Ten separate \`late_source\` cases per cohort are also evaluated as **one** model-visible aliased feature group among forty source cases; late cases are secondary only.

For the primary 1000 independent model-input cases we report native future-correct count, false accepts, paid checks, case-paired gains and losses, distinct-input exact two-sided sign test, and per-family/cohort breakdown. All 1,040 outcomes are preserved individually in the hosted artifact. The original frozen GRU is reported alongside all six locally DEV-selected arms and the globally DEV-selected policy.

**Primary causal-value criterion:** residual_early must exceed (i) original, (ii) reconstructed_early and (iii) endpoint_nonlinear on the 1000 nonlate cases, with a positive paired net gain and each two-sided exact paired p-value ≤ 0.05/3. Comparisons to raw_early, shuffled_early and hidden_only, and the \`late_source\` cluster, are secondary. The dataset still has family-specific structure, so these unadjusted sign p-values do not certify distributional universality. Any borderline positive result requires an independent fresh-seed replication before capability claims.

The empirical task effect remains distinct from an information-theoretic proof: even if a residual improves a finite readout, a potentially richer function of H9 could learn the same feature on restricted sources. A finite sample with continuously valued endpoints cannot identify I(Y;H4|H9) without extra assumptions.

## Reproduce

    python -m unittest research.nor15_trajectory_sufficiency.test_task_test -v
    python research/nor15_trajectory_sufficiency/task_test.py --check \
       --repetitions 4 --nonlate-per-family 50 --late-per-repetition 10 \
       --epochs 150 --out /tmp/nor15b_full.json

Source-lock tests audit public-label absence of future data, exact candidate budget, 400-slot models, and unique-input pairing. This result remains preregistered until the full independent run completes.
