# FHC-01: finite-horizon compactification gap

Research started 2026-11-13. Sources: OpenAI Math [#094](https://github.com/openai/math/blob/main/lean/docs/094.md) and [#098](https://github.com/openai/math/blob/main/lean/docs/098.md).

## Main theorem

Conditional on the two published source results, a compact doubling Hilbert subset has N-point dense prefixes whose distortion-D embedding dimension must diverge with N, while l4 admits the upper bound exp(C_D sqrt(log N)).

[THEORY.md](THEORY.md) supplies the Gaussian transfer, finite obstruction compactness, robust epsilon-net obstruction, additive coarse embedding and exact linear shift-register observability rank min(T,m).

## Repeatable finite controls

Python 3.10+ and standard library:

    python -m unittest discover -s research/finite_horizon_gap_v1 -p 'test_*.py' -v
    python research/finite_horizon_gap_v1/reference.py --check --out /tmp/fhc01_control.json

The finite controls check Gaussian fourth moments, equilateral packing, distortion-transfer tolerance epsilon=delta/6, doubling cover arithmetic, rational finite l2->l4 maps and shift observability ranks.

The source papers' compact doubling witness is not reproduced by these finite checks, and no trained neural checkpoint is tested.

[PROTOCOL.md](PROTOCOL.md) defines the future-observation metric experiment, owned separately from RC-01, IC-01, IRC-02 and SUD-01.

The independently kernel-checked discrete bridge is in [MathCan PR #31](https://github.com/patrickdugan/MathCan/pull/31). It covers the integer one-sixth clearance and full-prefix recovery lemmas only.


## Executed extensions: RC-01B and trained RRC-25

[FHC-01B results](RESULTS_B.md) derive and measure the gap between
exact semantic binary memory and the Euclidean dimension required
for preserving full future-observation distances. The six-bit
primitive looper needs just six predictive bits at horizon 63,
while its exact 64-state future-trace isometry requires 63 real
coordinates. A 48/16 calibration/heldout PCA split has the
full-regular-simplex unseen-state collapse proved by FHC-8.
The frozen CI pilot includes PCA, Gaussian projection and exact
binary quotient comparisons with held-out pair witnesses.

[FHC-01C results](RESULTS_C.md) evaluate the existing source-locked
1,000-parameter RRC-25 trained recurrent GRU checkpoint. The
benchmark [protocol](PROTOCOL_RRC25.md) defines a fixed zero-input
recurrent continuation and measures the geometry of future logits.
At horizon 16 the native hidden-state 12-coordinate norm exhibits
39.528 worst-pair distortion on the frozen held-out cases, versus
1.230 for a 12-coordinate PCA of future-logit traces fitted on
separate dev cases. The latter has a significant future-computation
and projection-description cost, which is charged in the receipt.

All results retain distinct evidence labels: exact finite theorem,
synthetic recurrent pilot, and trained-model retrospective geometry.
They are not a construction of the #098 compact doubling counterexample.
The companion [MathCan Lean PR #31](https://github.com/patrickdugan/MathCan/pull/31)
checks the discrete one-sixth tolerance lemma, full-horizon prefix
recovery and an explicit six-bit observable-mask orbit.
