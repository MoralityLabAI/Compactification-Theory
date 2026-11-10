# RRC-19: balanced source families for certified graph-class ranking

**Executed November 10, 2026, one CPU training seed.** The classifier architecture and finite graph language were held fixed from RRC-18; the new variable is the training-source mixture. All new programs are verified against two exact degree-12 demonstrations, and prediction through degree 32 is scored separately. Four checkpoints, a source-bound full replay, and two post-primary robustness batches are retained. Earlier RRC-18 scores informed the problem choice.

## Main positive test: 140 cases per source family

| Source family | Balanced GRU | Skewed GRU | Balanced FF | Shuffled GRU | Frozen RRC-18 GRU | Class-size prior | Full class search |
|---|---:|---:|---:|---:|---:|---:|---:|
| Signed, seen | 126 | **129** | 104 | 81 | 137 | 84 | 140 |
| Partition product, seen | **134** | 119 | 130 | 100 | 84 | 97 | 140 |
| Sparse signed, seen | 122 | **123** | 112 | 88 | 121 | 87 | 140 |
| Rational sparse reciprocal, **unseen** | 99 | **101** | 100 | 84 | 93 | 91 | 140 |
| Convolution-square, **unseen** | 116 | **125** | 110 | 86 | **125** | 82 | 140 |
| **All five groups /700** | **597** | **597** | 556 | 439 | 560 | 441 | 700 |
| **Unseen two /280** | 215 | **226** | 210 | 170 | 218 | 173 | 280 |

Counts are **correct future predictions**, and every positive-class acceptance on this roster also predicts the future correctly. The primary robustness gate, which required balanced recurrent training to exceed the skewed GRU on the *combined novel source families*, **fails**. The balanced GRU exceeds the feedforward and mass prior on that combination, but the skewed GRU retains eleven more successes. The zero-test-tuned six-neural/six-mass hybrid recovers 204/280 unseen and 124/140 product programs, below the balanced GRU in both groups.

The important conditional gain is on **partition products**, which had produced only 92/160 successes for RRC-18's signed-only GRU. Under matched fresh RRC-19 test instances, frozen RRC-18 finds 84/140 while the balanced GRU finds 134/140. The difference reflects *explicit product-source training*, which was absent from RRC-18. The matched skewed mix finds 119/140. Since both new models have the same parameters, initialization and optimization count, the 15-program advantage isolates the change in example weighting within this fixed training setup.

### Paired outcomes and arithmetic work

In the 140 product cases, balanced is exclusively correct on **17** cases versus **2** skewed-only. On rational inputs the exclusive outcomes are 7 versus 9; on squared series they are 4 versus 13. Balanced/FF exclusive results on products are 6 versus 2. On squared inputs, balanced exceeds the class-size prior on 40 cases while that prior exceeds balanced on 6.

| Family | Method | Mean exact queries | Mean nominal integer-loop work | Dense forward MACs |
|---|---|---:|---:|---:|
| Product | Balanced GRU | **2.88** | **762** | 53,216 |
| Product | Skewed GRU | 4.30 | 1,136 | 53,216 |
| Product | Balanced FF | 3.56 | 921 | 6,120 |
| Product | Class-mass prior | 6.93 | 1,512 | 0 |
| Rational, unseen | Balanced GRU | 6.43 | 1,846 | 53,216 |
| Rational, unseen | Skewed GRU | 6.89 | 1,958 | 53,216 |
| Rational, unseen | Balanced FF | 6.29 | 1,754 | 6,120 |
| Rational, unseen | Class-mass prior | 7.59 | 1,611 | 0 |

The proxy units are deliberately separate. The GRU's inference is approximately 8.7 times the feedforward MAC count and includes no interpretation cost; the exact symbolic verifier and optimizer-backpropagation costs remain separate. The saved receipt records the full precision values and per-topology splits.

## Two post-primary independent-seed checks

Additional *post-primary* seeds, with frozen weights and unchanged candidate ordering, sampled 80 cases per source family per seed. No gradient updates or model selection followed their results.

| Source, 160 cases across two new seeds | Balanced GRU | Skewed GRU | Balanced FF | Frozen RRC-18 GRU | Class-size prior |
|---|---:|---:|---:|---:|---:|
| Partition products | **150** | 135 | 147 | 95 | 94 |
| Rational reciprocals, unseen | 106 | 110 | **114** | 95 | 107 |
| Convolution squares, unseen | 137 | 145 | 125 | **149** | 103 |

These checks support the product improvement and corroborate the absence of a robust cross-family advantage against skewed GRU training. The extra seeds are a robustness measure conducted *after* the primary test, not a predeclared blind confirmatory study.

## Exact boundaries and training resources

All **20** deliberately inconsistent demonstration cases are rejected by every bounded and full-class verifier. Eight paired future-collision inputs produce identical predictions within each arm; complete search accepts all 16 visible prefixes and makes precisely eight false future predictions. On the 700 main positive cases the full certified finite-class solver accepts 700/700, using a larger (up to 33-program) budget. The 12-candidate neural methods receive no interpreter previews.

Each trained GRU has **6,240 FP32 parameters** and 32 FP32 recurrent coordinates. The matched FF has **6,235 parameters**. Training used 3,600 examples, balanced over 12 topology/source cells, and 420 development cases. Each of the four arms trained for 700 Adam updates with batch 128 and learning rate .003, for 89,600 training-example visits. Balanced sampling made **29,668 signed, 29,941 product and 29,991 sparse draws**. Skewed sampling made **71,523 signed, 13,647 product and 4,430 sparse draws**. Its initial GRU parameters match the balanced model exactly. Shuffled targets were permuted within training cells to preserve class frequencies.

All four training jobs together require about **14.85 billion estimated forward-only dense MACs**. Backpropagation, source generation, optimizer state, and model memory transfers are omitted. The RRC-18 exact quotient requires 256 offline degree-12 graph normalizations (certified polynomial classes), with interpreter code and class tables separately provisioned. Every verification checks both public demos with exact integer arithmetic; no hidden graph or future is read by the policy. Historical frozen RRC-18 training costs are additional and uncharged in the RRC-19 training total.

## Reproducibility and decision

The source-bound receipt records hashes of all defining files, four checkpoints, two frozen RRC-18 checkpoints, train/dev/test rosters, per-method execution traces, per-topology scores and the two robustness seeds. **17 RRC-19 unit tests and 36 RRC-17/RRC-18 regression tests pass**. A fresh independent training run produced byte-identical JSON checkpoints for all four trained roles. The entire receipt replays deterministically with Python standard library and the frozen model files.

This is an exploratory single-seed result. Graph classes and the interpreter were provided, all four topologies were familiar during training, and the model generated candidate orderings rather than discovering algebraic primitives. Stronger next tests should withhold a topology *and* its graph-class prior, or train a representation that can predict operator semantics under true family shifts, while comparing against the proven finite-degree symbolic baseline and its offline compilation costs.
