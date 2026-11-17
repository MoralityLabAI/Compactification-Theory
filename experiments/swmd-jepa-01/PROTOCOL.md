# SWMD-JEPA-01 preregistration and scaling gate

**Paper:** Balestriero & LeCun (2026), arXiv:2610.39302.
**Question:** Can positive pairs drawn from independent views of one symbolic
state improve out-of-world representation transfer and latent dynamical closure?

## Definitions

For state `s`, action `a`, and renderings `x_A(s)` and `x_B(s)`, an encoder
`f_theta(x)` produces `z`. The controlled condition minimizes a non-collapse
VICReg surrogate between `f(x_A(s))` and `f(x_B(s))`. In the bad-view condition,
both positives are independent masks of a *single randomly selected* caption.
The next-token condition trains the identical encoder backbone through a
language-model head. The dynamics arm augments good-view training by repeatedly
applying an action-conditioned recurrent cell `G(z,a)` toward a stop-gradient
embedding of `T(s,a)`.

The Platonic Representation Theorem assumes an idealized finite world,
injective observation functions, suitably connected good-view pairs and
appropriate population optima. Our shuffled text views, small GRU, finite
sample training and VICReg loss do **not** satisfy all those conditions. The
experiment is an empirical test of a training principle.

## Fixed comparison

- Dataset: 96 synthetic worlds × 12 deterministic transitions; `data-seed=2026`.
- Assign entire `world_id` groups to 70/15/15 train/validation/test with a
  separate seeded shuffle. Test once after fixed training epochs; hold validation
  for later, explicitly documented model-selection studies.
- Training: seeds `11,29,47`, all four arms; embedding width 48; AdamW
  `lr=0.001`, `weight_decay=0.0001`; 10 epochs; batch 64; gradient norm 5.
- Good/bad-view JEPA uses VICReg weights 25 (invariance), 25 (variance), 1
  (covariance); dynamics arm adds weight 5 to successor latent MSE.
- Decoder: identical float64 ridge probe (`alpha=1`) on train embeddings;
  linear transition features include the state vector, action one-hot and
  their action-specific interactions. No test-label gradient updates.
- Metric directions: retrieval and state accuracy **higher**; closure NMSE
  and view inconsistency **lower**, conditional on adequate effective rank.
- Freeze all source, data, and training seeds before looking at final metrics;
  archive SHA-256 fingerprints, raw per-arm receipts and summary.

## Primary planned contrasts

1. `good_view - bad_view` in held-out-world retrieval R@1 and macro attribute
   accuracy. Demand positive mean paired differences for both; assess uncertainty
   from paired seeds, and compare retrieval to the measured chance baseline.
2. `good_view - next_token` on those same scores, with trainable-head and
   token-budget differences explicitly recorded.
3. `good_view_transition - good_view` in closure-probe NMSE, where a **negative**
   paired difference is favorable. Separately report direct recurrent-predictor
   NMSE for the dynamics arm. This contrast is an architectural treatment,
   not an isolated effect of good positive pairs.

A seed bootstrap resamples *paired seeds* (2000 resamples, seed 2026); with
three training seeds its intervals are descriptive and underpowered. World-group
bootstrap and more seeds are required before strong statistical claims. The
validation set is reserved and never used by the present runner.

## Interpretation and failure guards

- Cross-world splits can reuse symbolic state values, so report
  `unseen_state_fraction` and the unseen-subset retrieval/closure metrics.
- A trivially collapsed encoder has low view discrepancy. Check effective
  embedding rank, retrieval against class-prevalence chance and state decoding.
- Probe closure reflects both representation quality and probe capacity.
  High closure NMSE may reflect a limited **linear** readout; low NMSE alone
  does not prove controlled Markov sufficiency or causal closure.
- D sees successor captions and a recurrent auxiliary head; it receives more
  tokens and gradient work than C. Inspect `tokens_seen` before cost claims.
- Generated toy dialects may permit rote lexical mapping. The next step needs
  controlled paraphrases, distractors, partial observations, and paired
  environment interventions.
- This is an exploratory pilot without claims of replicating the paper's 42×
  sample-efficiency figure. That figure concerns sample counts under a different
  experiment, and should not be converted into GPU-hour savings.

## Escalation (spend gate)

Proceed to a pre-registered, matched-token and matched-FLOP study with genuine
SWMD state traces only after all three-seed primary comparisons are recorded
and the good-view variant improves **both** primary representation metrics
with effective rank above trivial collapse. Compare recurrent predictors at
matched extra gradient compute, add world-level bootstrap and unseen-world
interventions, then consider ~100M-scale encoders on a rental GPU. If the
representation gains vanish under matched-budget controls or held-out
interventions, preserve the negative result and stop the scale-up.
